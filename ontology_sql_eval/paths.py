# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the repo keeps its data, relative to the repo root.

Kept in one place so that moving a directory is one edit rather than a search
across the ingestion code, the Gym task builder and both resources servers.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Relative to the repo root.
DATASETS_DIR = Path("data", "datasets")
ANNOTATIONS_DIR = Path("data", "annotations")
