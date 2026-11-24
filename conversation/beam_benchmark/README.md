<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# BEAM

## Run it

Both commands need the env loaded first, and they must never run at the same
time: the eval reads the graph the build is wiping.

```
set -a && . ./.env && set +a
```

### 1. Build the graph

```
.venv/bin/python -m conversation.beam_benchmark.build_graph 2 --verify --subject-agent
```

Takes 16–30 minutes and wipes conversation 2's graph before it starts. The
transcripts themselves are untouched: the turns and their vectors are ingested
once and the graph is rebuilt over them, which is why the eval can run against
a graph built this way without re-ingesting anything.

Most of the time is the agentic subject pass. `--verify` and `--subject-agent`
are both optional and both change what lands, so a score is only comparable to
another score built with the same flags.

### 2. Score it

```
.venv/bin/python main.py --database-name beam --skip-migrate --skip-ingest \
  --scale 100K --conversations 2 --eval-workers 4 \
  --retrieval agent --run-dir conversation/beam_benchmark/results-agent
```

Two arms, and the difference is not a parameter of the same method:

| `--retrieval` | What answers |
| --- | --- |
| `vector` (default) | One similarity search over turn text, handed to a reader that answers in one call. Matches the published one-retrieval protocol. |
| `agent` | The ReAct agent with conversation search on. It writes its own queries, reads what comes back, and may search again, reaching the ontology through the same tool. |

Only `agent` uses the graph that step 1 builds, so a graph rebuild has no
effect on a `vector` score.

Output is named after the arm, so the command above writes
`results/100K/agent_r1.jsonl` and so on. To keep a run instead of letting
the next one of the same arm overwrite it, point `--run-dir` somewhere else --
the same flag and the same rule as the SQL arms, and the reason nothing is
named by hand: a score filed under a label that describes a configuration it
was not produced with is worse than no score.

`main.py` on the `beam` database has no graph stage — `--skip-ingest` plus
`--skip-eval` does nothing at all. `--skip-ingest` is not optional: without it
the transcripts are re-ingested with `reset=True`, which renumbers turns and
breaks every citation in the graph.

Keep `--eval-workers` at 4. At 8 the embeddings endpoint has returned 429 and
killed a round, and a round that answers fewer than 20 questions **invalidates
the whole run**: `_require_one_question_set` refuses to pool means taken over
different question sets, so four rounds minus one is not three rounds, it is an
error and no overall score. The per-round files survive on disk, so the fix is
to re-run only the short rounds.

**The score to beat is 0.79** — cognee's 0.794 on this conversation over four
runs.

### What the agent arm answers under

Nothing to export: these are library defaults, recorded onto every answer row
by `agent_limits` and reported in the summary, so a finished run says what it
was produced with rather than relying on the command that started it.

| Setting | Default | Env var |
| --- | --- | --- |
| Conversation search result size | 24,000 chars | `TEXT_TO_SQL_REACT_MAX_CONVERSATION_OBSERVATION_CHARS` |
| Any other tool result | 8,000 chars | `TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS` |
| Tool calls per question | 24 | `TEXT_TO_SQL_REACT_MAX_TOOL_CALLS` |

A transcript turn runs an order of magnitude longer than a catalog entry, which
is why conversation search has a ceiling of its own. Setting the general one
narrows conversations with it — a smaller context window is the usual reason
for setting it, and that reason applies to every source.

One setting belongs to the build rather than the score:
`CONVERSATION_EXTRACT_KNOWN_CAP` limits how many already-extracted nodes are
listed in each extraction window's prompt. Off by default, which lists them
all. Capping it saves prompt tokens at the cost of a window being unable to see
a node it may then introduce again under another name.

---

[Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs](https://arxiv.org/abs/2510.27246)
(ICLR 2026), [upstream implementation](https://github.com/mohammadtavakoli78/BEAM),
data on the Hub as [Mohammadta/BEAM](https://huggingface.co/datasets/Mohammadta/BEAM)
and [Mohammadta/BEAM-10M](https://huggingface.co/datasets/Mohammadta/BEAM-10M).

100 synthetic multi-session conversations, 2,000 probing questions, scored by an
LLM judge against hand-written rubrics. 20 conversations at 100K tokens, 35 at
500K, 35 at 1M, 10 at 10M; 20 questions per conversation, two for each of ten
memory abilities.

The paper calls the smallest scale 128K and the dataset files call it 100K. This
directory uses 100K throughout, because that is what has to be fetched.

## How a score is computed

Scored the way cognee scored their published **0.794**, so that a number of ours
can be put beside it: their judge prompt
([`beam_rubric.py`](https://github.com/topoteretes/cognee/blob/main/cognee/eval_framework/beam/eval/metrics/beam_rubric.py)),
their judge model (`gpt-5`), their reader prompts on the vector arm, and four
rounds. All of it is fixed in `conversation/benchmarks.py`, not by a flag.

Each question carries a rubric: a list of **nuggets**, atomic self-contained
claims the response has to satisfy. The judge sees the question, one nugget and
the response, and returns `0.0`, `0.5` or `1.0`, snapped to those three levels
with cuts at 0.25 and 0.75. The question's score is the mean over its nuggets,
for all ten abilities, event ordering included. At 100K that is 400 questions
and 1,051 nuggets, a median of 2 per question and up to 9.

There is no gold answer to diff against: BEAM ships a non-empty
`ideal_response` for its 40 abstention questions and leaves it blank for all
360 others. The rubric is the reference.

As in cognee's judge, an unparseable or failed judge reply **scores 0.0** with
no retry. The summary counts these as `judge_errors`, and it should be 0 before
a score is read. Connection-shaped endpoint failures are retried below the
judge, so an outage does not turn into zeros.

Per-conversation ability scores are averaged across conversations, then the ten
ability means are averaged into the headline number. With two questions per
ability per conversation that equals cognee's flat mean over questions; it is
kept as a two-level average so a dropped question does not silently reweight an
ability. [`compare_cognee.py`](../../compare_cognee.py) puts a run beside
cognee's figures, and refuses one that was not judged by cognee's prompt and
judge model or that is missing questions.

## What cognee's 0.794 is

Cognee reports **0.79** at 100K and **0.67** at 10M
([report](https://github.com/topoteretes/cognee/blob/main/cognee/eval_framework/beam/REPORT.md)).
Things to know before reading a gap to it:

- **One conversation, 20 questions.** Conversation 2 at 100K. Their prompts and
  retriever settings were tuned on two other conversations and checked on two
  more. The quoted standard deviation is across four repeated answer-and-judge
  rounds on a fixed question set: judge and decoding noise, not sampling error
  over questions.
- **`gpt-5` answers as well as judges**, so a frontier reader alone accounts for
  part of the distance from simpler baselines.
- **The ability label selects the answer prompt.** It ships with the benchmark,
  so this is legitimate as a benchmark run, and their report says outside a
  labelled benchmark the type "would need to be inferred from the question
  itself". Our vector arm does the same; the agent arm does not.

Their memory construction is a superset of ours: `add` + `cognify` builds a
knowledge graph of entities and relationships, plus summaries, plus chunks, and
hybrid retrieval draws from all three. The chunk grain matches ours, one
user/assistant turn per chunk.

## Layout

```
beam_benchmark/
  download_data.py   fetch a scale from the Hub into ../data/beam/<scale>/
  build_graph.py     rebuild one conversation's semantic layer
  verify_graph.py    check a stored graph's edges against their turns
  evaluator/         cognee's rubric judge, ported
  reference.py       cognee's answers, joined onto our rows
  prompts/           third-party prompts, byte-for-byte (see THIRD_PARTY_NOTICES.md)
    judge/           cognee's judge prompt
    answer/cognee/   cognee's per-ability answer prompts
  cognee_results/    cognee's published config, summary and answers
  results/           run output (git-ignored)
```

`cognee_results/` holds their config and cross-run summary copied
unmodified from
[`cognee/eval_framework/beam/report_artifacts/100k_fixed/`](https://github.com/topoteretes/cognee/tree/main/cognee/eval_framework/beam/report_artifacts/100k_fixed),
plus `cognee_answers_conv2.json` distilled from the per-question artifacts in
the same directory. So `compare_cognee.py` runs offline and the numbers it
prints for their column are theirs rather than ours transcribed.
