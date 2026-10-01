"""Write both files the eval takes on the command line, in one run.

``eval_chatbot`` is given ``--value-anchors`` and ``--sql-examples``, and
producing them took three commands across two scripts: build the value index,
derive the anchors from it, then have the LLM predict a query per dev question
and match its shape against train. Which of those to run, and in what order,
was the whole difficulty. This runs all of them.

Each step is invoked through its own script's command line rather than by
importing it, so the defaults that apply here are the same ones that apply when
those scripts are run directly, and nothing has to be kept in sync by hand.

The LLM answers are never replayed. ``predict`` caches its responses and will
reuse them when the cache it is pointed at already holds the request, which
reproduces the previous exemplar file byte for byte while spending nothing and
looking, from the outside, like a real run. Since the point of running this is
to *produce* the inputs, any existing cache is archived before the step starts
and the answers are always resampled.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
VALUE_INDEX = ROOT / "scripts/value_index.py"
EXEMPLARS = ROOT / "scripts/llm_structural_exemplars.py"
# Not output/, whose contents are gitignored: these two files are eval inputs
# and are committed, so a run can be reproduced without regenerating them.
OUT_DIR = ROOT / "prompt_inputs/bird"


def child_env() -> dict[str, str]:
    """The parent environment, with PYTHONPATH from .env if it is not set.

    The exemplar step imports ``gsf``, which lives outside this repo and is
    reached through PYTHONPATH. The launch configs supply it via ``envFile``,
    but a plain shell does not, and the child calling ``load_dotenv`` itself is
    too late: the interpreter reads PYTHONPATH before any of that code runs.
    """
    env = dict(os.environ)
    if not env.get("PYTHONPATH"):
        from_file = dotenv_values(ROOT / ".env").get("PYTHONPATH")
        if from_file:
            env["PYTHONPATH"] = from_file
    return env


def run(step: str, argv: list[str], dry_run: bool = False) -> float:
    printable = " ".join(str(a) for a in argv[1:])
    print(f"\n=== {step} ===\n$ {Path(argv[0]).name} {printable}", flush=True)
    if dry_run:
        return 0.0
    start = time.time()
    result = subprocess.run([sys.executable, *argv], cwd=ROOT, env=child_env())
    if result.returncode != 0:
        raise SystemExit(f"{step} failed with exit code {result.returncode}")
    elapsed = time.time() - start
    print(f"--- {step} finished in {elapsed:.0f}s", flush=True)
    return elapsed


def rows(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    with path.open(encoding="utf-8", newline="") as fh:
        # Header excluded; quoted SQL spans lines, so this counts records only
        # approximately and is a sanity check, not a measurement.
        return f"{sum(1 for _ in fh) - 1:,} lines"


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--only",
        choices=("anchors", "exemplars"),
        help="Write just one of the two. Default: both.",
    )
    ap.add_argument(
        "--anchors-out", type=Path, default=OUT_DIR / "value_anchors.csv"
    )
    ap.add_argument(
        "--exemplars-out",
        type=Path,
        default=OUT_DIR / "predicted_structural_exemplars.csv",
    )
    ap.add_argument("--k", type=int, default=5, help="Exemplars kept per question.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--limit", type=int, default=0, help="First N dev questions, for a smoke test."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run, and run none of them.",
    )
    args = ap.parse_args()

    want_anchors = args.only != "exemplars"
    want_exemplars = args.only != "anchors"
    timings: list[tuple[str, float]] = []

    if not args.dry_run:
        for out in (args.anchors_out, args.exemplars_out):
            out.parent.mkdir(parents=True, exist_ok=True)

    if want_anchors:
        # The index is rebuilt every time: it is derived wholly from the dev
        # databases, and a stale one would quietly produce stale anchors.
        timings.append(
            ("value index", run("value index", [str(VALUE_INDEX), "build"], args.dry_run))
        )
        timings.append(
            (
                "value anchors",
                run(
                    "value anchors",
                    [str(VALUE_INDEX), "anchors", "--out", str(args.anchors_out)],
                    args.dry_run,
                ),
            )
        )

    if want_exemplars:
        out = args.exemplars_out
        # Derived from the output path, never from predict's own default, which
        # is a fixed name: pointing --out somewhere new while that default
        # still resolves to a populated cache is what replays stale answers.
        cache = out.with_name(f"{out.stem}_cache.jsonl")
        if cache.exists():
            # Moved aside rather than deleted, and stamped with its own mtime
            # so the name records when those answers were generated. The fresh
            # run then writes this same canonical name, which keeps the cache
            # sitting beside a CSV always the one that produced it.
            stamp = datetime.fromtimestamp(cache.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
            archived = cache.with_name(f"{cache.stem}_{stamp}.jsonl")
            if not args.dry_run:
                cache.rename(archived)
            print(f"archived previous cache -> {archived.name}", flush=True)
        argv = [
            str(EXEMPLARS), "predict",
            "--out", str(out),
            "--cache", str(cache),
            "--k", str(args.k),
            "--workers", str(args.workers),
        ]
        if args.limit:
            argv += ["--limit", str(args.limit)]
        timings.append(
            ("structural exemplars", run("structural exemplars", argv, args.dry_run))
        )

    if args.dry_run:
        print("\n(dry run: nothing was executed)")
        return

    print("\n=== done ===")
    for name, seconds in timings:
        print(f"  {name:<22} {seconds:>6.0f}s")
    if want_anchors:
        print(f"  {'-> ' + args.anchors_out.name:<22} {rows(args.anchors_out)}")
    if want_exemplars:
        print(f"  {'-> ' + args.exemplars_out.name:<22} {rows(args.exemplars_out)}")
    print("\npass these to the eval as --value-anchors and --sql-examples.")


if __name__ == "__main__":
    main()
