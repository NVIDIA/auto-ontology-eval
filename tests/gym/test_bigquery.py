# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spider2-lite on BigQuery: connection strings, SQL rewriting and scoring.

BigQuery itself is replaced by a fake client; what is under test is what the
verifier sends it and how it scores what comes back.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pandas as pd
import pytest

from ontology_sql_eval.bigquery import (
    DEFAULT_MAXIMUM_BYTES_BILLED,
    MAXIMUM_BYTES_BILLED_ENV,
    bigquery_url,
    make_client,
    parse_bigquery_url,
    parse_byte_size,
    qualify_datasets,
    script_write_keywords,
    targets_from_connection_strings,
)
from ontology_sql_eval.gym.exec_match import (
    BigQueryExecutor,
    GoldResults,
    execute_and_compare,
)
from ontology_sql_eval.gym.spider2 import table_from_csv_text

BILLING = "my-billing"
URL = (
    f"bigquery://{BILLING}/ga4"
    "?dataset=bigquery-public-data.ga4_obfuscated_sample_ecommerce"
)


# --- connection strings --------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1024", 1024),
        ("500MB", 500 * 10**6),
        ("20GiB", 20 * 1024**3),
        ("1.5 TB", int(1.5 * 10**12)),
        ("0", None),
        ("unlimited", None),
        (None, None),
    ],
)
def test_parse_byte_size(text: Optional[str], expected: Optional[int]) -> None:
    assert parse_byte_size(text) == expected


def test_parse_byte_size_rejects_unknown_units() -> None:
    with pytest.raises(ValueError):
        parse_byte_size("20 parsecs")


def test_url_round_trip_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MAXIMUM_BYTES_BILLED_ENV, raising=False)
    url = bigquery_url(
        BILLING, "noaa", ["bigquery-public-data.noaa_gsod", "other-proj.ghcn_d"]
    )
    target = parse_bigquery_url(url)
    assert target.db_id == "noaa"
    assert target.billing_project == BILLING
    assert target.datasets == (
        ("bigquery-public-data", "noaa_gsod"),
        ("other-proj", "ghcn_d"),
    )
    assert target.maximum_bytes_billed == DEFAULT_MAXIMUM_BYTES_BILLED
    assert target.default_dataset is None
    assert parse_bigquery_url(URL).default_dataset == (
        "bigquery-public-data.ga4_obfuscated_sample_ecommerce"
    )


def test_byte_cap_matches_auto_ontology_not_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MAXIMUM_BYTES_BILLED_ENV, "5GB")
    assert parse_bigquery_url(URL).maximum_bytes_billed == DEFAULT_MAXIMUM_BYTES_BILLED
    assert (
        parse_bigquery_url(URL + "&maximum_bytes_billed=0").maximum_bytes_billed is None
    )


@pytest.mark.parametrize(
    "url",
    [
        "bigquery:///db?dataset=p.d",
        "bigquery://b/db?dataset=p1.same&dataset=p2.same",
    ],
)
def test_bad_urls_are_refused(url: str) -> None:
    with pytest.raises(ValueError):
        parse_bigquery_url(url)


def test_auto_ontology_url_variants_are_supported() -> None:
    assert parse_bigquery_url("bigquery://billing").db_id == "billing"
    target = parse_bigquery_url(
        "bigquery://billing/db?dataset=local&dataset=public.sales&schema=local"
    )
    assert target.datasets == (("billing", "local"), ("public", "sales"))
    assert target.scoped_datasets == (("billing", "local"),)
    assert target.default_dataset == "billing.local"


def test_parser_matches_pinned_auto_ontology_contract() -> None:
    from auto_ontology.connectors.bigquery import _parse_connection_string

    key = {
        "type": "service_account",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    encoded = base64.urlsafe_b64encode(json.dumps(key).encode()).decode()
    urls = [
        "bigquery://billing",
        "bigquery://billing/db?dataset=local&location=EU&schema=local",
        "bigquery://billing/db?dataset=public.sales&maximum_bytes_billed=0",
        f"bigquery://billing/db?dataset=p.d&credentials={encoded}",
        "bigquery://billing/db?credentials_file=%2Fsecrets%2Fsa.json",
    ]
    for url in urls:
        ours = parse_bigquery_url(url)
        theirs = _parse_connection_string(url)
        assert ours.billing_project == theirs["project"]
        assert ours.db_id == theirs["database"]
        assert list(ours.datasets) == theirs["datasets"]
        assert ours.maximum_bytes_billed == theirs["maximum_bytes_billed"]
        assert ours.location == theirs["location"]
        assert ours.schema == theirs["schema"]
        assert ours.credentials == theirs["credentials"]
        assert ours.credentials_file == theirs["credentials_file"]


def test_explicit_credentials_build_the_client() -> None:
    target = parse_bigquery_url(
        "bigquery://billing/db?dataset=p.d&location=US&credentials="
        + base64.urlsafe_b64encode(
            json.dumps({"type": "service_account"}).encode()
        ).decode()
    )
    with (
        patch(
            "google.oauth2.service_account.Credentials.from_service_account_info"
        ) as from_info,
        patch("google.cloud.bigquery.Client") as client_cls,
    ):
        make_client(target)
    from_info.assert_called_once()
    assert client_cls.call_args.kwargs == {
        "project": "billing",
        "credentials": from_info.return_value,
        "location": "US",
    }


def test_targets_skip_other_schemes() -> None:
    raw = f"sqlite:///x.sqlite,\n{URL}\n"
    assert list(targets_from_connection_strings(raw)) == ["ga4"]


# --- qualification -------------------------------------------------------------


def test_qualify_datasets_adds_the_owning_project() -> None:
    projects = {"noaa_gsod": "bigquery-public-data"}
    assert qualify_datasets("SELECT stn FROM noaa_gsod.stations s", projects) == (
        "SELECT stn FROM `bigquery-public-data`.noaa_gsod.stations AS s"
    )


def test_qualify_datasets_leaves_qualified_sql_byte_for_byte() -> None:
    sql = "SELECT  *\nFROM `bigquery-public-data.noaa_gsod.gsod*`  -- keep\n"
    assert qualify_datasets(sql, {"noaa_gsod": "bigquery-public-data"}) == sql
    assert qualify_datasets("SELECT * FROM other.t", {"noaa_gsod": "p"}) == (
        "SELECT * FROM other.t"
    )


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("DECLARE d STRING DEFAULT '2017'; SELECT d", []),
        ("CREATE TEMP FUNCTION f(x INT64) AS (x); SELECT f(1)", []),
        ("CREATE OR REPLACE TEMPORARY TABLE t AS SELECT 1; SELECT * FROM t", []),
        ("SELECT 'DELETE' AS s, `update` FROM t", []),
        ("DECLARE x INT64; DELETE FROM p.d.t WHERE TRUE; SELECT 1", ["DELETE"]),
        ("CREATE TABLE p.d.t AS SELECT 1; SELECT 1", ["CREATE"]),
        ("DECLARE q STRING DEFAULT 'x'; EXECUTE IMMEDIATE q", ["EXECUTE"]),
    ],
)
def test_script_write_keywords(sql: str, expected: list[str]) -> None:
    assert script_write_keywords(sql) == expected


# --- scoring -------------------------------------------------------------------


class _Job:
    def __init__(self, statement_type: str, total_bytes: int, frame: Any) -> None:
        self.statement_type = statement_type
        self.total_bytes_processed = total_bytes
        self._frame = frame

    def result(self, timeout: Optional[float] = None) -> Any:
        if isinstance(self._frame, BaseException):
            raise self._frame
        frame = self._frame

        class _Rows:
            def to_dataframe(self, **_: Any) -> pd.DataFrame:
                return frame

        return _Rows()

    def cancel(self) -> None:
        pass


class _FakeClient:
    """Answers every query with one canned result; records what was run."""

    def __init__(
        self,
        frame: Any,
        *,
        statement_type: str = "SELECT",
        total_bytes: int = 10**6,
    ) -> None:
        self.frame = frame
        self.statement_type = statement_type
        self.total_bytes = total_bytes
        self.ran: list[tuple[str, Any]] = []

    def query(self, sql: str, job_config: Any = None, location: Any = None) -> _Job:
        self.ran.append((sql, job_config))
        return _Job(self.statement_type, self.total_bytes, self.frame)


def _executor(client: _FakeClient, url: str = URL) -> BigQueryExecutor:
    targets = targets_from_connection_strings(url)
    executor = BigQueryExecutor(targets, timeout_s=5)
    executor._clients[executor._client_key(next(iter(targets.values())))] = client
    return executor


def _gold(tmp_path: Path, text: str) -> GoldResults:
    path = tmp_path / "gold" / "bq900.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return GoldResults.from_task(tmp_path, ["gold/bq900.csv"], None, True)


def _score(client: _FakeClient, gold: GoldResults, sql: str = "SELECT 1"):
    return asyncio.run(
        execute_and_compare(_executor(client), "ga4", "", sql, "spider2_lite_bq", gold)
    )


def test_matching_frame_scores(tmp_path: Path) -> None:
    gold = _gold(tmp_path, "name,revenue\na,1.5\nb,2\n")
    frame = pd.DataFrame({"n": ["b", "a"], "r": [2.0, 1.5]})
    client = _FakeClient(frame)
    assert _score(client, gold)[:2] == (True, None)
    dry, real = client.ran
    assert dry[1].dry_run and not real[1].dry_run
    assert dry[1].labels == {"source": "auto-ontology"}
    assert real[1].labels == {"source": "auto-ontology"}
    assert real[1].maximum_bytes_billed == DEFAULT_MAXIMUM_BYTES_BILLED
    assert real[1].default_dataset is not None


def test_wrong_frame_misses(tmp_path: Path) -> None:
    gold = _gold(tmp_path, "name,revenue\na,1.5\n")
    assert _score(_FakeClient(pd.DataFrame({"n": ["a"], "r": [9.0]})), gold)[:2] == (
        False,
        None,
    )


def test_empty_result_scores_zero_even_against_empty_gold(tmp_path: Path) -> None:
    gold = _gold(tmp_path, "name\n")
    assert _score(_FakeClient(pd.DataFrame({"name": []})), gold)[0] is False


def test_non_select_is_refused_before_it_runs(tmp_path: Path) -> None:
    gold = _gold(tmp_path, "x\n1\n")
    client = _FakeClient(pd.DataFrame({"x": [1]}), statement_type="DELETE")
    assert _score(client, gold, "DELETE FROM t WHERE TRUE")[:2] == (
        False,
        "pred_sql_error",
    )
    assert len(client.ran) == 1 and client.ran[0][1].dry_run


def test_read_only_script_runs(tmp_path: Path) -> None:
    gold = _gold(tmp_path, "x\n1\n")
    client = _FakeClient(pd.DataFrame({"x": [1]}), statement_type="SCRIPT")
    assert _score(client, gold, "DECLARE a INT64 DEFAULT 1; SELECT a AS x")[0] is True


def test_query_over_the_cap_is_not_run(tmp_path: Path) -> None:
    gold = _gold(tmp_path, "x\n1\n")
    client = _FakeClient(pd.DataFrame({"x": [1]}), total_bytes=10**15)
    assert _score(client, gold)[:2] == (False, "pred_sql_bytes_cap")
    assert len(client.ran) == 1


def test_infrastructure_errors_raise(tmp_path: Path) -> None:
    from google.api_core.exceptions import Forbidden

    gold = _gold(tmp_path, "x\n1\n")
    denied = Forbidden("User does not have bigquery.jobs.create permission in project")
    with pytest.raises(Forbidden):
        _score(_FakeClient(denied), gold)


def test_sql_errors_are_misses(tmp_path: Path) -> None:
    from google.api_core.exceptions import BadRequest

    gold = _gold(tmp_path, "x\n1\n")
    client = _FakeClient(BadRequest("Unrecognized name: totals"))
    assert _score(client, gold)[:2] == (False, "pred_sql_error")


def test_csv_round_trip_types_like_pandas() -> None:
    frame = pd.DataFrame({"n": pd.array([1, None], dtype="Int64"), "s": ["a", "b"]})
    table = table_from_csv_text(frame.to_csv(index=False))
    assert table.kinds == ("float64", "object")
