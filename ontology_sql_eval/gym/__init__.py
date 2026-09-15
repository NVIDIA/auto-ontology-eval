# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Gym environment support for the ontology eval.

Holds the pure, testable pieces the Gym resources servers under
``resources_servers/`` depend on: schema dumping (:mod:`ddl`) and dataset
conversion (:mod:`tasks`). The servers themselves stay thin so that Gym's
workspace layout is satisfied without putting logic outside the linted package.
"""
