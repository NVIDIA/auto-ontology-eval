# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spec for patches/adk_numeric_type_compare.patch (ADK process_decimals_recursive).

Keep this function in sync with the patch. The ADK checker used to round
Decimal and float separately, then compare with ``==``, so Decimal('0.01')
failed against 0.01. Coercing both to Decimal after rounding is the fix —
not rewriting Auto Ontology SQL.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

PATCH = (
    Path(__file__).resolve().parents[2]
    / "patches"
    / "adk_numeric_type_compare.patch"
)


def process_decimals_recursive(item, decimal_places: int = 2):
    quantizer = Decimal(1).scaleb(-decimal_places)
    if isinstance(item, bool):
        return item
    if isinstance(item, Decimal):
        return item.quantize(quantizer, rounding=ROUND_HALF_UP)
    if isinstance(item, float):
        return Decimal(str(item)).quantize(quantizer, rounding=ROUND_HALF_UP)
    if isinstance(item, int):
        return Decimal(item).quantize(quantizer, rounding=ROUND_HALF_UP)
    if isinstance(item, (list, tuple)):
        return type(item)(process_decimals_recursive(x, decimal_places) for x in item)
    if isinstance(item, dict):
        return {k: process_decimals_recursive(v, decimal_places) for k, v in item.items()}
    return item


def test_patch_file_coerces_float_via_decimal():
    text = PATCH.read_text()
    assert "Decimal(str(item))" in text
    assert "isinstance(item, float)" in text
    assert "isinstance(item, bool)" in text


def test_decimal_and_float_same_value_compare_equal():
    assert process_decimals_recursive(Decimal("0.01")) == process_decimals_recursive(
        0.01
    )
    assert process_decimals_recursive(Decimal("53.27")) == process_decimals_recursive(
        53.27
    )


def test_extra_float_precision_matches_gold_rounded_to_2dp():
    gold = Decimal("0.013369130432692595")
    pred = 0.013369130432692595
    assert process_decimals_recursive(gold) == process_decimals_recursive(pred)


def test_int_matches_numeric_decimal():
    assert process_decimals_recursive(53) == process_decimals_recursive(Decimal("53"))


def test_bool_is_not_coerced_to_decimal():
    assert process_decimals_recursive(True) is True
    assert process_decimals_recursive(False) is False


def test_old_checker_would_reject_decimal_vs_float():
    """Guard: the bug we are patching still exists in raw Python."""
    gold = Decimal("0.01").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    pred = round(0.01, 2)
    assert gold != pred
