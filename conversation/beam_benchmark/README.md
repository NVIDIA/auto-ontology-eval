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
.venv/bin/python build_graph.py 2 --verify --subject-agent
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

## Published results

Table 1 of the paper, reproduced here so a run of ours has something to be
compared against without re-reading the PDF. Mean question score, 0–1, higher is
better. **Vanilla** is the whole conversation in the context window, **RAG** is
their retrieval baseline, **LIGHT** is the paper's own method.

| Scale | Reader model | Vanilla | RAG | LIGHT |
| --- | --- | --- | --- | --- |
| 100K | Qwen2.5-32B-AWQ | 0.280 | 0.269 | 0.311 |
| 100K | Llama-4-Maverick-fp8 | 0.240 | **0.323** | 0.358 |
| 100K | Gemini-2.0-flash | 0.242 | 0.280 | 0.294 |
| 100K | GPT-4.1-nano | 0.239 | 0.309 | 0.345 |
| 500K | Qwen2.5-32B-AWQ | 0.200 | 0.291 | 0.316 |
| 500K | Llama-4-Maverick-fp8 | 0.283 | **0.330** | 0.359 |
| 500K | Gemini-2.0-flash | 0.257 | 0.267 | 0.292 |
| 500K | GPT-4.1-nano | 0.194 | 0.314 | 0.335 |
| 1M | Qwen2.5-32B-AWQ | 0.193 | 0.285 | 0.309 |
| 1M | Llama-4-Maverick-fp8 | 0.259 | **0.307** | 0.336 |
| 1M | Gemini-2.0-flash | 0.199 | 0.271 | 0.284 |
| 1M | GPT-4.1-nano | 0.191 | 0.302 | 0.336 |
| 10M | Qwen2.5-32B-AWQ | 0.133 | 0.211 | 0.238 |
| 10M | Llama-4-Maverick-fp8 | 0.104 | **0.249** | 0.266 |
| 10M | Gemini-2.0-flash | 0.122 | 0.216 | 0.192 |
| 10M | GPT-4.1-nano | 0.109 | 0.218 | 0.226 |

The best baseline is Llama-4-Maverick with RAG at every scale (bolded). Two
things are worth noticing before treating any of these as a target:

- The baseline barely moves between 100K and 1M. The benchmark does not start
  punishing length until 10M.
- LIGHT beats the best baseline by 0.02–0.035. That is roughly a fifth of the
  spread you get from which 20 questions you happen to sample, so the ranking
  between nearby rows is not something a small run can resolve.

### Per-ability, 100K

Kept because the overall mean hides which abilities the score is made of —
abstention carries a lot of it and contradiction resolution contributes almost
nothing for anyone.

| Memory ability | Qwen RAG | Llama RAG | Gemini RAG | GPT RAG |
| --- | --- | --- | --- | --- |
| Abstention | 0.650 | 0.800 | 0.800 | 0.800 |
| Contradiction resolution | 0.025 | 0.031 | 0.050 | 0.018 |
| Event ordering | 0.201 | 0.162 | 0.191 | 0.169 |
| Information extraction | 0.338 | 0.392 | 0.341 | 0.362 |
| Instruction following | 0.375 | 0.375 | 0.287 | 0.350 |
| Knowledge update | 0.275 | 0.350 | 0.325 | 0.375 |
| Multi-hop reasoning | 0.203 | 0.225 | 0.148 | 0.263 |
| Preference following | 0.379 | 0.512 | 0.416 | 0.550 |
| Summarization | 0.074 | 0.111 | 0.093 | 0.083 |
| Temporal reasoning | 0.162 | 0.275 | 0.150 | 0.125 |
| **Average** | **0.269** | **0.323** | **0.280** | **0.309** |

## What the 0.323 actually is

The RAG baseline is deliberately plain, and its chunk grain is the same one our
`conversation` table uses: one question/answer pair per row, embedded. From §4.1:

> For RAG baselines, each user–assistant turn pair is treated as a document,
> embedded and stored in a vector database. At inference, the top five most
> similar documents are retrieved and passed to the LLM.

In `src/answer_probing_questions/long_term_memory_methods.py` that is
`retrieval_method="pair_chunk"` with `retriever="dense"`:

| Step | Setting |
| --- | --- |
| Chunk | one `USER: … ASSISTANT: …` pair, no overlap |
| Embedding | `BAAI/bge-small-en-v1.5`, normalized |
| Index | FAISS, in-memory, cosine similarity |
| Depth | top `k=5` |
| Context | hits concatenated to a ~29,000-token reader budget |
| Reader | "answer ONLY from the context, do not use internal knowledge" |
| Judge | `gpt-4.1-mini`, temperature 0 |

`dense` is just their label for the bge-small configuration; `e5` and the dense
half of `hybrid` are dense retrievers too.

One discrepancy between the prose and the code, which matters if you treat 0.323
as a clean reference. Their `pair_chunk` builder nests the pair construction
inside `for message in turn:`, so each pair is appended once per message in its
turn. On `chats/100K/7` that turns 130 unique pairs into 324 index entries, a
2.5x duplication. With `k=5` over a duplicated index the reader sees two or three
distinct pairs rather than five, so the published baseline is weaker than the
description suggests.

## The two judge prompts

`--judge-prompt` selects which. They differ in one input line and three
behaviours, and the choice is not a detail: the same answers score very
differently under the two.

| | `beam` (default) | `cognee` |
| --- | --- | --- |
| Source | `src/prompts.py`, verbatim | `beam_rubric.py`, verbatim |
| Judge sees the question | no | **yes** |
| System message | none | "You are an evaluation judge. Return only valid JSON." |
| Score snapping | none | 0/0.5/1, cuts at 0.25 and 0.75 |
| Unparseable or failed reply | raises, question excluded | **scores 0.0**, no retry |
| Comparable to | 0.323 | Cognee's 0.79 |

`beam` is the default because it is the instrument the published table was
measured on, and a number scored any other way does not belong beside it. It is
also the worse prompt. If the goal is to know whether a retrieval change helped,
`cognee` is the better measurement; if the goal is a figure next to 0.323, it
has to be `beam`.

Two warnings about `cognee`. It silently converts an endpoint failure into a
real zero rather than raising, so a reasoning judge that runs out of token
budget contributes a confident 0.0 -- the summary reports `judge_errors` for
exactly this, and it should be 0 before a score is read. And the snapping is
theirs, not a rounding convenience; it only bites when a judge invents 0.8, but
it is part of their arithmetic.

## How a score is computed

Each question carries a rubric: a list of **nuggets**, atomic self-contained
claims the response has to satisfy. The judge sees one nugget and the response —
not the question, not a gold answer — and returns `0.0`, `0.5` or `1.0`. The
question's score is the mean over its nuggets. At 100K that is 400 questions and
1,051 nuggets, a median of 2 per question and up to 9.

There is no gold answer to diff against, and not only because the judge is not
shown one: BEAM ships a non-empty `ideal_response` for its 40 abstention
questions and leaves it blank for all 360 others. The rubric is the reference.
Most nuggets are phrased as instructions about the answer rather than as the
answer — `LLM response should state: March 29`, `LLM response should contain:
avoids recommending large frameworks` — so anything that scores a run by
comparing text to `ideal_response` is scoring 40 questions and silently zeroing
the rest.

Event ordering is scored differently, and the paper and the code disagree about
how. §2 says Kendall tau-b, and `compute_metrics.evaluate_event_ordering`
computes `final_score = tau_norm * f1` **and** a nugget `llm_judge_score`. But
`report_results.py` aggregates neither:

```python
if key == "event_ordering":
    score += object['tau_norm']
else:
    score += object['llm_judge_score']
```

So the published event-ordering numbers are bare `tau_norm = (tau_b + 1) / 2`,
with the response split on newlines and aligned to the gold events by an LLM
equivalence detector. That is why every system scores 0.15–0.27 on it: below 0.5
means the rank vectors are anti-correlated, which is what you get when the
response and the rubric name mostly different events.

**Cognee does not do this**, and it took us a while to notice. Their
[`beam_rubric.py`](https://github.com/topoteretes/cognee/blob/main/cognee/eval_framework/beam/eval/metrics/beam_rubric.py)
contains no tau and no case for this ability — it is the nugget mean for all
ten.
[`kendall_tau.py`](https://github.com/topoteretes/cognee/blob/main/cognee/eval_framework/beam/eval/metrics/kendall_tau.py)
is a separate metric registered alongside it, and their published summary
carries both, with tau `null` for the nine abilities it does not apply to:

```json
"event_ordering": {
  "beam_rubric": { "mean": 0.7875, ... },
  "kendall_tau": { "mean": 0.514638501094228, ... }
},
"information_extraction": {
  "beam_rubric": { "mean": 0.9791666666666667, ... },
  "kendall_tau": null
}
```

The proof that their headline *includes* the nugget score for this ability and
not the tau is in the same file: `overall.kendall_tau.mean` is
`0.514638501094228`, equal to `event_ordering.kendall_tau.mean` to the last
digit, because
[`overall_from_by_type`](https://github.com/topoteretes/cognee/blob/main/cognee/eval_framework/beam/eval/aggregate_cross_run.py)
pools each metric separately and only one ability contributes a tau. Meanwhile
`overall.beam_rubric.mean` of `0.7943` is the flat mean of all twenty questions'
rubric scores, the ordering pair at 0.7875 among them.

So `--judge-prompt` picks the rule as well as the prompt:

| | event ordering score | tau |
| --- | --- | --- |
| `beam` | `tau_norm` | is the score |
| `cognee` | nugget mean | reported beside it |

Both numbers are computed and stored either way, so a finished run can be
restated against either table without re-judging — see
[`compare_cognee.py`](../../compare_cognee.py).

Per-conversation ability scores are averaged across conversations, then the ten
ability means are averaged into the headline number. With two questions per
ability per conversation that equals a flat mean over questions; it is kept as a
two-level average so a dropped question does not silently reweight an ability.

## Other reported numbers

Cognee reports **0.79** at 100K and **0.67** at 10M
([report](https://github.com/topoteretes/cognee/blob/main/cognee/eval_framework/beam/REPORT.md)).
Neither is on the same instrument as anything above, for five separate reasons,
and the report is candid about most of them.

**They changed what event ordering is scored by**, replacing rank correlation
with the nugget mean and reporting rank correlation separately. The report does
not mention it and `beam_rubric.py` does not flag it, but their published
summary shows both numbers for that ability and the aggregator pools them
separately; the [scoring section](#how-a-score-is-computed) works through the
proof. This is the one difference that moves a *comparison* rather than the
score itself, because it is invisible unless you look: both tables print a
column called event ordering and the two columns are different metrics. It is
reproduced as part of `--judge-prompt cognee`.

**They changed the judge, twice over.** Answer generation *and* the rubric judge
both run on `openai/gpt-5` (Appendix A), where the paper's judge is
`gpt-4.1-mini` at temperature 0. They also changed the *prompt*, which the
report does not mention but
[`beam_rubric.py`](https://github.com/topoteretes/cognee/blob/main/cognee/eval_framework/beam/eval/metrics/beam_rubric.py)
is explicit about:

> Fix applied: `<question>` is actually substituted — the paper's code never
> fills it in, leaving the literal string "`<question>`" for the judge.

They are right that it is a bug, and it is a consequential one. BEAM's prompt
instructs the judge "if the response does not address the QUESTION, score 0.0
and stop" while never supplying a question, so the judge has to guess at
responsiveness. `--judge-prompt beam` reproduces the defect faithfully.

So 0.79 is on a third instrument -- neither ours nor the paper's -- and cannot
be ranked against 0.323. `evaluator/` implements their variant as
`--judge-prompt cognee` so the comparison has a shared ruler.

**They changed the reader.** `openai/gpt-5` answers as well, so a frontier
reader alone can account for a large part of the distance from the published
baselines, independent of whether Cognee's memory helps.

**They used BEAM's answer key as an input.** Each question's ability label
selects an ability-specific answer prompt. The label ships with the benchmark,
so this is legitimate as a benchmark run, and §5 says plainly that outside a
labelled benchmark the question type "would need to be inferred from the question
itself" and that they did not evaluate a router. The paper's baselines send one
prompt for all ten abilities.

**One conversation, 20 questions, 1% of the benchmark**, per scale. At 100K
prompts and retriever settings were tuned on two conversations, checked on two
more, and a fifth was held out for the reported score. At 10M there is no
held-out set: the question-type routing was selected by sweeping the same 20
questions that produced the 0.67, which the report labels in-sample and
exploratory. The quoted standard deviation of 0.005 is variation across four
repeated QA-and-judging rounds on a fixed question set — it measures judge and
decoding noise, not sampling error over questions, and the report says repetition
"does not add new question samples".

Their memory construction is a genuine superset of ours and the interesting part
to compare against later: `add` + `cognify` builds a knowledge graph of entities
and relationships, plus summaries, plus chunks, and hybrid retrieval draws from
all three views. The chunk grain matches ours — one user/assistant turn per
chunk, turn order and session metadata preserved, no overlap chunking. They also
LLM-compressed roughly 0.2% of turns before ingestion to strip noise and leaked
conversation-plan text, which is a data-cleaning step neither we nor the paper
apply.

## Layout

```
beam_benchmark/
  download_data.py   fetch a scale from the Hub into ../data/beam/<scale>/
  evaluator/         the paper's judge, ported
  reference.py       cognee's answers, joined onto our rows
  prompts/           third-party prompts, byte-for-byte (see THIRD_PARTY_NOTICES.md)
    judge/           BEAM's judge and equivalence prompts, cognee's judge
    answer/          BEAM's answer prompt
      cognee/        cognee's per-ability answer prompts
  cognee_results/    cognee's published config, summary and answers
  results/           run output (git-ignored)
```

`cognee_results/` holds their config and cross-run summary copied
unmodified from
[`cognee/eval_framework/beam/report_artifacts/100k_fixed/`](https://github.com/topoteretes/cognee/tree/main/cognee/eval_framework/beam/report_artifacts/100k_fixed),
plus `cognee_answers_conv2.json` distilled from the per-question artifacts in
the same directory. So `compare_cognee.py` runs offline and the numbers it
prints for their column are theirs rather than ours transcribed.

Each benchmark judges with the model its published figure was judged with.
`beam-paper` runs on `azure/openai/gpt-4.1-mini` at temperature 0, which is the
paper's `gpt-4.1-mini` under the id `REASONING_ENDPOINT` routes it by. `beam`
runs on `azure/openai/gpt-5`, cognee's judge: it is the stricter of the two on
the same answers, so a cognee-prompt score judged by `gpt-4.1-mini` overstates
where we stand against 0.794. Both are set in `conversation/benchmarks.py`, not
by a flag.

Smoke-tested against that endpoint on one question per ability: an answer built
from the rubric's own nuggets scores 1.000 on seven of the ten and the
deliberately wrong answer scores 0.000 on all nine nugget-averaged abilities.
Event ordering returns 1.000 for the events in the right order and 0.146 for the
wrong answer — the below-chance floor described above, visible in practice.

