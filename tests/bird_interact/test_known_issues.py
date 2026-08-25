# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ADK p1snap-collision workaround (known_issues.py)."""

from __future__ import annotations

from ontology_sql_eval.bird_interact.known_issues import (
    corrected_phase1_result,
    is_p1snap_collision,
)

_P1SNAP_MESSAGE = (
    "Error: Command ['createdb', '-h', '127.0.0.1', '-p', '5433', '-U', "
    "'root', 'labor_certification_applications__labor_certification_"
    "applications_2__p1snap', '--template', 'labor_certification_"
    "applications__labor_certification_applications_2'] returned "
    "non-zero exit status 1."
)


def test_recognizes_known_p1snap_collision_message():
    assert is_p1snap_collision(_P1SNAP_MESSAGE) is True


def test_does_not_match_unrelated_errors():
    assert is_p1snap_collision("SQL failed Phase 1. Your SQL is not correct.") is False
    assert (
        is_p1snap_collision("function pg_catalog.btrim(enum_visa_class) does not exist")
        is False
    )
    assert is_p1snap_collision("") is False
    assert is_p1snap_collision(None) is False


def test_does_not_match_createdb_errors_unrelated_to_p1snap():
    """Only the specific p1snap-suffixed collision is a known, safe-to-correct
    issue — any other createdb failure (disk full, permission denied, a
    genuinely different naming bug) must NOT be silently reinterpreted as a
    pass.
    """
    other = (
        "Error: Command ['createdb', '-h', '127.0.0.1', '-p', '5433', '-U', "
        "'root', 'some_db'] returned non-zero exit status 1."
    )
    assert is_p1snap_collision(other) is False


def test_corrected_result_first_try_uses_first_try_reward():
    fix = corrected_phase1_result(_P1SNAP_MESSAGE, is_first_try=True)
    assert fix == {"reward": 0.7, "phase_completed": 1}


def test_corrected_result_debug_retry_uses_debug_reward():
    fix = corrected_phase1_result(_P1SNAP_MESSAGE, is_first_try=False)
    assert fix == {"reward": 0.5, "phase_completed": 1}


def test_corrected_result_none_for_unrelated_message():
    assert (
        corrected_phase1_result("SQL failed Phase 1. Your SQL is not correct.", True)
        is None
    )
