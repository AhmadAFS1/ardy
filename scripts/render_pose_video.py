#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convenience wrapper for :mod:`ardy.pose_video`."""

import sys

from ardy.pose_video.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["render", *sys.argv[1:]]))
