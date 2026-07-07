# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the judge CLI argument parsing."""

from __future__ import annotations

from pathlib import Path

from evaluation.judge.main import _parse_args


def test_defaults() -> None:
    args = _parse_args([])
    assert args.input_dir == Path("input")
    assert args.output_dir == Path("output")
    assert args.workers == 1


def test_overrides() -> None:
    args = _parse_args(
        ["--input-dir", "csv_in", "--output-dir", "csv_out", "--workers", "4"]
    )
    assert args.input_dir == Path("csv_in")
    assert args.output_dir == Path("csv_out")
    assert args.workers == 4
