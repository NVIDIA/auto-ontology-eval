"""Replay the intent validator's new "Dropped Conditions" check on a finished run.

Run from the GSF repo with its venv so the real prompt/model are imported:

    cd GSF && uv run python ../auto-ontology-eval/scripts/regress_filter_check.py \
        RESULTS.jsonl ADAPTER.log OUT.json [--repeat 2]

For each task the final Phase-1 SQL is shown to the validator together with the
original question and the evidence block the pipeline built (taken from the
adapter log). A task whose SQL passed in the run but now gets ``filter_issues``
is a regression candidate; a failed task that gets flagged is a possible rescue.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

GOLD = (
    Path(__file__).resolve().parents[1]
    / "datasets/bird_interact_full/bird_interact_data_with_gt.jsonl"
)
ENV = Path(__file__).resolve().parents[1] / ".env"
MAX_EVIDENCE = 6000


def load_env() -> None:
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if k.startswith("DEFAULT_MODELS_") and k not in os.environ:
                os.environ[k] = v.strip().strip('"').strip("'")


def evidence_blocks(log: str) -> dict[str, str]:
    """First 'SQL gen — Evidence:' block per instance id."""
    out: dict[str, str] = {}
    lines = log.splitlines()
    ts = re.compile(r"^\d\d:\d\d:\d\d ")
    i = 0
    while i < len(lines):
        m = re.match(r"^\d\d:\d\d:\d\d INFO \[([^\]]+)\] SQL gen — Evidence:(.*)", lines[i])
        if m and m.group(1) not in out:
            body = [m.group(2).strip()]
            i += 1
            while i < len(lines) and not ts.match(lines[i]):
                body.append(lines[i])
                i += 1
            out[m.group(1)] = "\n".join(body).strip()
            continue
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--ids", default="")
    ap.add_argument("--reasoning", action="store_true")
    args = ap.parse_args()

    load_env()
    from auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation import (
        IntentValidationModel,
    )
    from auto_ontology.retrieval.text_to_sql.prompts import (
        INTENT_VALIDATION_SYSTEM_PROMPT,
        create_intent_validation_prompt,
        format_authoritative_evidence,
    )
    from auto_ontology.utils import llm_invoke

    reasoning_client_name = "get_reasoning_llm_client"
    get_reasoning_llm_client = getattr(llm_invoke, reasoning_client_name, None)
    llm = (
        get_reasoning_llm_client()
        if args.reasoning and get_reasoning_llm_client is not None
        else llm_invoke.get_non_reasoning_llm_client()
    )

    def load_jsonl(path: Path) -> dict:
        rows = {}
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows[row["instance_id"]] = row
        return rows

    if "filter_issues" not in IntentValidationModel.model_fields:
        print(
            "warning: IntentValidationModel has no filter_issues; filter flags stay empty",
            file=sys.stderr,
        )

    gold = load_jsonl(GOLD)
    res = load_jsonl(Path(args.results))
    evid = evidence_blocks(Path(args.log).read_text(errors="replace"))
    ids = [i for i in args.ids.split(",") if i] or list(res)

    def run(iid: str) -> dict:
        r = res[iid]
        sql = r.get("p1_debug_sql") if r.get("phase1_passed") and r.get("p1_debug_sql") else r.get("p1_sql")
        if not sql:
            return {"id": iid, "skipped": "no sql"}
        question = gold[iid]["amb_user_query"]
        evidence = evid.get(iid, "")[:MAX_EVIDENCE]
        prompt = create_intent_validation_prompt(
            question, question, question, sql, has_evidence=bool(evidence)
        )
        msgs: list[BaseMessage] = [SystemMessage(content=INTENT_VALIDATION_SYSTEM_PROMPT)]
        if evidence:
            msgs.append(SystemMessage(content=format_authoritative_evidence(evidence)))
        msgs.append(HumanMessage(content=prompt))
        flagged = []
        other = []
        for _ in range(args.repeat):
            try:
                v = llm_invoke.invoke_with_structured_output(llm, msgs, IntentValidationModel)
            except Exception as e:  # noqa: BLE001
                return {"id": iid, "error": str(e)[:200]}
            if v is None:
                continue
            # Current IntentValidationModel has no filter_issues field.
            # model_dump keeps this working if a newer package adds it.
            flagged.append(list(v.model_dump().get("filter_issues") or []))
            other.append(
                bool(v.join_issues or v.aggregation_issues or v.evidence_issues)
            )
        return {
            "id": iid,
            "passed_in_run": bool(r.get("phase1_passed")),
            "filter_flags": flagged,
            "flagged_runs": sum(1 for f in flagged if f),
            "other_issue_runs": sum(other),
            "n": len(flagged),
            "has_evidence": bool(evidence),
        }

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(run, ids))
    Path(args.out).write_text(json.dumps(results, indent=2))
    for x in results:
        if x.get("flagged_runs"):
            tag = "PASSED-IN-RUN" if x["passed_in_run"] else "failed-in-run"
            print(f"{tag:14} {x['id']:40} {x['flagged_runs']}/{x['n']}  {x['filter_flags'][0][:2]}")
    print(
        "total",
        len(results),
        "| flagged among passed:",
        sum(1 for x in results if x.get("passed_in_run") and x.get("flagged_runs")),
        "of",
        sum(1 for x in results if x.get("passed_in_run")),
        "| errors:",
        sum(1 for x in results if x.get("error")),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
