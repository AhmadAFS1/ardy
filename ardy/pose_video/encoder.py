# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic raw-RGB to deterministic H.264/yuv420p encoding."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


class FFmpegEncodingError(RuntimeError):
    """Raised when ffmpeg rejects or cannot encode the rendered stream."""


@dataclass(frozen=True)
class EncodingResult:
    output_path: Path
    frame_count: int
    raw_rgb_sha256: str
    mp4_sha256: str
    verified_boundary_frames: int


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_frame_hashes(path: str | Path, ffmpeg_binary: str | Path | None = None) -> list[str]:
    """Return ffmpeg framemd5 hashes of the decoded video planes."""

    if ffmpeg_binary is None:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise FileNotFoundError("ffmpeg was not found on PATH")
    else:
        executable = str(ffmpeg_binary)
    command = [
        executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "framemd5",
        "pipe:1",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegEncodingError(
            f"could not decode boundary frames from {path}: {result.stderr.strip()}"
        )
    hashes: list[str] = []
    for line in result.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 6:
            hashes.append(fields[-1])
    if not hashes:
        raise FFmpegEncodingError(f"ffmpeg decoded no frames from {path}")
    return hashes


def verify_decoded_boundary_blocks(
    path: str | Path,
    boundary_frames: int,
    ffmpeg_binary: str | Path | None = None,
) -> None:
    """Require decoded first/last frame blocks to be pixel-identical.

    The comparison hashes ffmpeg's decoded YUV planes, not packets or source
    RGB, so it catches codec-context differences at the actual switching point.
    """

    if boundary_frames <= 0:
        raise ValueError("boundary_frames must be positive")
    hashes = decoded_frame_hashes(path, ffmpeg_binary=ffmpeg_binary)
    if len(hashes) < boundary_frames * 2:
        raise FFmpegEncodingError(
            f"video has {len(hashes)} frames; need at least {boundary_frames * 2} "
            f"to verify {boundary_frames}-frame boundaries"
        )
    first = hashes[:boundary_frames]
    last = hashes[-boundary_frames:]
    if first != last:
        differing = next(index for index, pair in enumerate(zip(first, last)) if pair[0] != pair[1])
        raise FFmpegEncodingError(
            "decoded boundary blocks are not pixel-identical; "
            f"first mismatch is boundary frame {differing}"
        )


class FFmpegH264Encoder:
    """Write fixed-size RGB arrays into an atomic MP4 output."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        width: int,
        height: int,
        fps: float,
        overwrite: bool = False,
        ffmpeg_binary: str | Path | None = None,
        crf: int = 18,
        preset: str = "medium",
        all_intra: bool = True,
        verify_boundary_frames: int = 0,
    ) -> None:
        self.output_path = Path(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.overwrite = bool(overwrite)
        self.crf = int(crf)
        self.preset = str(preset)
        self.all_intra = bool(all_intra)
        self.verify_boundary_frames = int(verify_boundary_frames)
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("encoder width and height must be positive even integers")
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("encoder fps must be a finite positive number")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be in [0, 51]")
        if self.verify_boundary_frames < 0:
            raise ValueError("verify_boundary_frames cannot be negative")

        if ffmpeg_binary is None:
            resolved = shutil.which("ffmpeg")
            if resolved is None:
                raise FileNotFoundError("ffmpeg was not found on PATH")
            self.ffmpeg_binary = resolved
        else:
            self.ffmpeg_binary = str(ffmpeg_binary)
            if not Path(self.ffmpeg_binary).is_file() and shutil.which(self.ffmpeg_binary) is None:
                raise FileNotFoundError(f"ffmpeg executable not found: {self.ffmpeg_binary}")

        self._process: subprocess.Popen[bytes] | None = None
        self._temporary_path: Path | None = None
        self._raw_digest = hashlib.sha256()
        self._frame_count = 0

    def __enter__(self) -> "FFmpegH264Encoder":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.abort()

    def open(self) -> None:
        if self._process is not None:
            return
        if self.output_path.exists() and not self.overwrite:
            raise FileExistsError(f"output already exists: {self.output_path}; pass overwrite=True to replace it")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{self.output_path.stem}.",
            suffix=".partial.mp4",
            dir=self.output_path.parent,
        )
        os.close(handle)
        self._temporary_path = Path(temporary_name)

        x264_params = ["threads=1", "lookahead_threads=1", "sliced_threads=0"]
        if self.all_intra:
            # Every frame is independently decoded. Consequently, identical
            # canonical source frames have identical decoded YUV pixels even
            # when separated by a different behavior interior.
            x264_params.extend(("keyint=1", "min-keyint=1", "scenecut=0", "bframes=0", "open-gop=0"))

        command = [
            self.ffmpeg_binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            f"{self.fps:.12g}",
            "-i",
            "pipe:0",
            "-map_metadata",
            "-1",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-movflags",
            "+faststart",
            # Single-threaded x264 and stripped metadata make identical input
            # frames byte-reproducible on the same ffmpeg/x264 build.
            "-threads",
            "1",
            "-x264-params",
            ":".join(x264_params),
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            str(self._temporary_path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._process is None:
            self.open()
        assert self._process is not None and self._process.stdin is not None
        frame = np.asarray(frame)
        expected_shape = (self.height, self.width, 3)
        if frame.shape != expected_shape:
            raise ValueError(f"frame must have shape {expected_shape}, got {frame.shape}")
        if frame.dtype != np.uint8:
            raise TypeError(f"frame must have dtype uint8, got {frame.dtype}")
        contiguous = np.ascontiguousarray(frame)
        payload = memoryview(contiguous).cast("B")
        self._raw_digest.update(payload)
        try:
            self._process.stdin.write(payload)
        except BrokenPipeError as exc:
            error = self._collect_error()
            self.abort()
            raise FFmpegEncodingError(f"ffmpeg closed its input early: {error}") from exc
        self._frame_count += 1

    def _collect_error(self) -> str:
        if self._process is None or self._process.stderr is None:
            return "unknown ffmpeg error"
        payload = self._process.stderr.read().decode("utf-8", errors="replace").strip()
        self._process.stderr.close()
        return payload

    def finish(self) -> EncodingResult:
        if self._process is None:
            raise RuntimeError("encoder was never opened")
        process = self._process
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        error = self._collect_error()
        return_code = process.wait()
        self._process = None
        if return_code != 0 or self._frame_count == 0:
            self._remove_temporary()
            if self._frame_count == 0 and return_code == 0:
                error = "no frames were provided"
            raise FFmpegEncodingError(f"ffmpeg exited with status {return_code}: {error}")

        assert self._temporary_path is not None
        if self.verify_boundary_frames:
            try:
                verify_decoded_boundary_blocks(
                    self._temporary_path,
                    self.verify_boundary_frames,
                    ffmpeg_binary=self.ffmpeg_binary,
                )
            except Exception:
                self._remove_temporary()
                raise
        os.replace(self._temporary_path, self.output_path)
        self._temporary_path = None
        return EncodingResult(
            output_path=self.output_path.resolve(),
            frame_count=self._frame_count,
            raw_rgb_sha256=self._raw_digest.hexdigest(),
            mp4_sha256=sha256_file(self.output_path),
            verified_boundary_frames=self.verify_boundary_frames,
        )

    def _remove_temporary(self) -> None:
        if self._temporary_path is not None:
            self._temporary_path.unlink(missing_ok=True)
            self._temporary_path = None

    def abort(self) -> None:
        if self._process is not None:
            process = self._process
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if process.stderr is not None and not process.stderr.closed:
                process.stderr.close()
            self._process = None
        self._remove_temporary()
