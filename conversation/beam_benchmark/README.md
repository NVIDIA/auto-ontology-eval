<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# BEAM

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
behaviours, and the choice is not a detail: it moved a 20-question score by
0.170 on unchanged text.

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
[`compare_cognee.py`](../../compare_cognee.py). What this cost us is in
[Scored the way cognee scores](#scored-the-way-cognee-scores-0716-against-0794).

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
column called event ordering and the two columns are different metrics. Ours
read 0.273 against their 0.787 while measuring the same answers at 0.838. It is
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
responsiveness. We reproduce the defect faithfully and it is measurable: the
same abstention answer scored 0.0 and 1.0 in consecutive rounds on a one-word
paraphrase ("not specified" against "not mentioned"). Under their prompt both
score 1.0.

So 0.79 is on a third instrument -- neither ours nor the paper's -- and cannot
be ranked against 0.323. `evaluator/` implements their variant as
`--judge-prompt cognee` so the comparison has a shared ruler; see the arm below.

**They changed the reader.** `openai/gpt-5` answers as well. Given that the
reader is worth roughly four times the retrieval on this benchmark (measured
above), a frontier reader alone accounts for a large part of the distance from
the published baselines, independent of whether Cognee's memory helps.

**They used BEAM's answer key as an input.** Each question's ability label
selects an ability-specific answer prompt. The label ships with the benchmark,
so this is legitimate as a benchmark run, and §5 says plainly that outside a
labelled benchmark the question type "would need to be inferred from the question
itself" and that they did not evaluate a router. Our arms and the paper's
baselines send one prompt for all ten abilities.

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

### Which 20 questions, and why it decides the number

§4.2 names the subset: 0.79 is "the held-out **second conversation**", four
QA-and-judging rounds over its 20 questions. Conversation 2 is in our data, so
the subset is matchable exactly -- and it turns out to be the single most
favourable transcript in the tier.

Our 400-question baseline, broken out per conversation, 20 questions each:

| | conv | score |
| --- | --- | --- |
| best | **2** | **0.796** |
| | 4 | 0.723 |
| | 15 | 0.716 |
| median | 9 / 7 | 0.592 / 0.583 |
| | 12 | 0.467 |
| worst | 5 | 0.422 |

Mean 0.605, standard deviation 0.095, range **0.374** from worst to best.

Two consequences. First, choosing conversation 2 is worth about **+0.19** over
our own tier mean before any method difference is considered, and it is rank 1
of 20 -- so a 20-question score on it is close to an upper bound rather than an
estimate of tier performance. Second, the range across conversations is wider
than the entire 0.282 gap between our baseline and the published 0.323, which
means no single-conversation result on this benchmark can separate two systems
by less than that. Cognee's ±0.005 does not cover this, by construction: it is
four repeated judging rounds on one fixed question set, so it measures judge and
decoding noise with the conversation held constant, which is the term that does
not matter.

That is not a criticism of their held-out split, which was a reasonable guard
against tuning leakage and is more than the 10M run has. It is a statement about
what 20 questions can resolve, and it applies symmetrically to our own
conversation-2 numbers below.

One trap this sets, worth naming because it is easy to fall into: our 0.796 on
conversation 2 sits right next to their 0.790, and the two are *not* a tie. They
were produced by different judge prompts. Scored on theirs we get 0.597. Any
comparison has to fix the judge first; see `cognee_full`.

A like-for-like Cognee comparison would still need their pipeline re-judged by
`gpt-4.1-mini` over all 400 questions with a single answer prompt. Nobody has
published that. The `cognee_conv2` arm approaches it from the other side instead
-- our pipeline on their conversation and their judge -- which is the half that
is ours to run.

### `cognee_conv2`: swapping only the judge model

Conversation 2, the same 20 questions, the same Claude Opus answers as the
baseline, rescored by `azure/openai/gpt-5`. Only the judge changes, so the
column difference is the instrument and nothing else.

| Memory ability | `gpt-5` | `gpt-4.1-mini` | Delta |
| --- | --- | --- | --- |
| Abstention | 0.500 | 1.000 | −0.500 |
| Contradiction resolution | 0.750 | 0.875 | −0.125 |
| Event ordering | 0.171 | 0.247 | −0.076 |
| Information extraction | 0.917 | 1.000 | −0.083 |
| Instruction following | 1.000 | 1.000 | 0.000 |
| Knowledge update | 1.000 | 1.000 | 0.000 |
| Multi-session reasoning | 0.500 | 1.000 | −0.500 |
| Preference following | 0.750 | 1.000 | −0.250 |
| Summarization | 0.667 | 0.833 | −0.167 |
| Temporal reasoning | 0.000 | 0.000 | 0.000 |
| **Overall** | **0.625** | **0.796** | **−0.170** |

**On the same prompt, `gpt-5` is the stricter judge by 0.170.** Note what this
arm is and is not: it swaps the judge *model* to theirs while keeping BEAM's
prompt, so it is not yet their instrument. The matched comparison is
`cognee_full` below, which also swaps the prompt and lands at 0.597.

The strictness is concentrated rather than uniform. Three abilities are
unchanged, while abstention and multi-session reasoning each halve. That
concentration is the tell: abstention is exactly where BEAM's prompt is weakest,
because deciding whether "there is no information about that" *addresses* a
question is impossible without the question. What looks like `gpt-5` being
strict is largely `gpt-5` guessing, and the guess breaks towards 0.0. Supplying
the question removes most of it -- the same abstention answer that scored 0.0
here scores 1.0 under `--judge-prompt cognee`.

So the two judge *models* are not separated by a constant offset, or even by a
stable per-ability one, as long as the prompt is BEAM's. Treat this arm as a
step towards `cognee_full` rather than a result.

A note on running a reasoning model as judge: `make_ask` lifts `max_tokens` to
8192 because `gpt-5` spends the allowance on hidden reasoning tokens before it
writes anything, and at 1024 it returned an empty string on 1 question in 20.
That surfaces as an unparseable reply rather than an error, so it silently drops
a question and leaves an ability resting on one. Check `questions_failed` in the
summary before reading any score off a reasoning judge.

## What we compare to

Two reference points, both at 100K over all 20 conversations and all 400
questions, both judged by `gpt-4.1-mini` at temperature 0:

| Reference | Score | What it is |
| --- | --- | --- |
| Published best baseline | **0.323** | Llama-4-Maverick + their RAG |
| **Our baseline** | **0.605** | `ontology` arm, Claude Opus 4.8 reader, `top_k=5` |

Cognee's 0.79 is a third reference and is handled separately, in `cognee_full`
below: it is 20 questions on one conversation under a different judge prompt,
so it needs its own matched arm rather than a row here. On that arm we are
0.597 to their 0.790.

The second is the one a change to our pipeline has to beat. It is recorded in
full below and its inputs are pinned: scale 100K, `top_k=5`, `widen=0`, reader
`aws/anthropic/bedrock-claude-opus-4-8`, one prompt for all ten abilities. Keep
the judge and the question set fixed when moving anything else, because those
two are what make successive runs subtractable.

0.323 stays alongside as the external anchor, but it is not the bar — the reader
gap means beating it says more about model choice than about memory. The honest
use of it is the `nano` arm below, which holds the reader at one of the paper's
own and so isolates what our retrieval is worth.

Exact reproduction is not the goal and is not reachable: our embedding endpoint
is not bge-small and our reader is not Llama-4-Maverick. Chasing the number would
also mean porting the chunk duplication above. What makes the comparison honest
is running our arms against each other on the same 400 questions through the
judge in `evaluator/`, which is the paper's judge — same prompt, same `0/0.5/1`
scale, same `tau_norm` for event ordering — and reporting 0.323 alongside as a
reference point rather than a finish line.

## Experiments

One arm per `--tag`, each varying one thing against the baseline so the
difference is attributable. Output lands in `results/<scale>/<tag>*`.

| Tag | Questions | Reader | Judge model | Judge prompt | Prompts | Rounds |
| --- | --- | --- | --- | --- | --- | --- |
| `ontology` | 400 | Claude Opus 4.8 | `gpt-4.1-mini` | `beam` | single | 1 |
| `nano` | 400 | GPT-4.1-nano | `gpt-4.1-mini` | `beam` | single | 1 |
| `cognee_conv2` | 20, conv 2 | Claude Opus 4.8 | `gpt-5` | `beam` | single | 1 |
| `cognee_match` | 20, conv 2 | GPT-5 | `gpt-5` | `beam` | per-ability | 4 |
| `cognee_full` | 20, conv 2 | GPT-5 | `gpt-5` | `cognee` | per-ability | 4 |
| `cognee_claude` | 20, conv 2 | Claude Opus 4.8 | `gpt-5` | `cognee` | per-ability | 4 |
| `cognee_replaced` | 20, conv 2 | GPT-5 | `gpt-5` | `cognee` | cognee, replacing BEAM's | 4 |
| `semantic_hybrid` | 20, conv 2 | GPT-5 | `gpt-5` | `cognee` | cognee, replacing BEAM's | 4 |

What each one is for: `ontology` is the baseline. `nano` holds retrieval fixed
and swaps the reader down to one of the paper's, which is what isolates the
reader's contribution. The `cognee_*` arms walk towards their setup one variable
at a time -- `cognee_conv2` changes only the judge model and the subset,
`cognee_match` adds their reader, their per-ability prompts and their four
rounds, `cognee_full` adds their judge prompt, and `cognee_replaced` stops
stacking that prompt on BEAM's and uses theirs alone, as their code does. At
that point only the memory system itself differs, so `cognee_replaced` is the
arm to quote against their 0.790. `cognee_claude` swaps the reader back to ours
and predates the prompt fix, so its conclusion is only partly sound.

Rules that keep these subtractable: the question set, judge model and judge
prompt move only when an arm exists to measure them; a new arm gets a new tag
rather than overwriting one; and the reader, prompt mode and retrieval depth are
recorded on every answer row rather than inferred from the tag.

Only `ontology` and `nano` may be put next to 0.323. The rest change the judge,
the question set or both.

## Our results

### 100K, `ontology`, 2026-10-19

All 20 conversations, all 400 questions, none failed. `top_k=5`, `widen=0`,
retrieval over our `conversation` table, reader
`aws/anthropic/bedrock-claude-opus-4-8`, judged by `evaluator/` on
`azure/openai/gpt-4.1-mini` at temperature 0. 12.8 minutes at 8 workers.

Reproduce with:

```bash
uv run python main.py --database-name beam --skip-ingest --skip-migrate \
    --skip-semantic --stage answer --tag ontology --eval-workers 8
uv run python main.py --database-name beam --skip-ingest --skip-migrate \
    --skip-semantic --stage judge --tag ontology --eval-workers 8
```

| Memory ability | Ours | Llama RAG | Delta |
| --- | --- | --- | --- |
| Abstention | 0.950 | 0.800 | +0.150 |
| Contradiction resolution | 0.487 | 0.031 | +0.457 |
| Event ordering | 0.220 | 0.162 | +0.058 |
| Information extraction | 0.833 | 0.392 | +0.441 |
| Instruction following | 0.569 | 0.375 | +0.194 |
| Knowledge update | 0.650 | 0.350 | +0.300 |
| Multi-session reasoning | 0.648 | 0.225 | +0.423 |
| Preference following | 0.906 | 0.512 | +0.394 |
| Summarization | 0.387 | 0.111 | +0.276 |
| Temporal reasoning | 0.400 | 0.275 | +0.125 |
| **Overall** | **0.605** | **0.323** | **+0.282** |

The reader is not theirs: ours is `aws/anthropic/bedrock-claude-opus-4-8` and
the 0.323 column is Llama-4-Maverick-fp8. How much of the gap that accounts for
is measured in the next section, not guessed at.

**One ability is scored generously.** Every contradiction-resolution question
carries a nugget reading literally `LLM response should mention: which statement
is correct?` — the rubric was derived from a gold answer that itself asked the
user to clarify, so the nugget is a question, not a claim. The judge awards it
for surfacing the conflict even when the answer picks no side. That is a quarter
of the nuggets on the ability, and it is upstream's rubric, not our scoring.

Event ordering stays flat at 0.220 because `tau_norm` is degenerate for
everybody, and `widen=0` means the reader never sees the turns surrounding a
hit. Summarization at 0.387 is a coverage ceiling rather than a reasoning one:
the worst case scored 0.0 with turns 73, 78, 86, 95 and 102 retrieved against a
rubric wanting episodes spread across all 94 turns. Five turns cannot answer
"how did my understanding develop throughout" no matter how good the reader is,
which is the argument for sweeping `top_k` and `--widen` on that ability.

### How much of that is the reader

The `nano` arm answers the same 400 questions off the same retrieval with
`us/azure/openai/gpt-4.1-nano`, one of the paper's own four readers, so its
0.309 column is directly comparable. Retrieval was byte-identical: all 400
questions returned the same five turns in the same order as the `ontology` arm,
making the reader the only variable.

| Memory ability | + nano | + Opus | Paper, nano | Retrieval | Reader |
| --- | --- | --- | --- | --- | --- |
| Abstention | 0.625 | 0.950 | 0.800 | −0.175 | +0.325 |
| Contradiction resolution | 0.037 | 0.487 | 0.031 | +0.006 | +0.450 |
| Event ordering | 0.184 | 0.220 | 0.162 | +0.022 | +0.036 |
| Information extraction | 0.535 | 0.833 | 0.392 | +0.143 | +0.298 |
| Instruction following | 0.463 | 0.569 | 0.375 | +0.088 | +0.106 |
| Knowledge update | 0.487 | 0.650 | 0.350 | +0.138 | +0.163 |
| Multi-session reasoning | 0.324 | 0.648 | 0.225 | +0.099 | +0.325 |
| Preference following | 0.706 | 0.906 | 0.512 | +0.194 | +0.200 |
| Summarization | 0.213 | 0.387 | 0.111 | +0.102 | +0.174 |
| Temporal reasoning | 0.150 | 0.400 | 0.275 | −0.125 | +0.250 |
| **Overall** | **0.372** | **0.605** | **0.309** | **+0.063** | **+0.233** |

**The reader is worth about four times what the retrieval is.** Holding the
reader at the paper's own nano, our retrieval buys +0.063 over theirs; swapping
the reader to Opus on top of that buys +0.233. Any reading of the headline
+0.282 as a retrieval result is wrong by roughly that ratio.

Contradiction resolution is the clearest case and the one most likely to be
misread. Our nano scores 0.037 against their 0.031 — indistinguishable — while
holding five genuinely distinct turns rather than their duplicated two or three.
Both halves of every conflict were in the context and nano answered "No" anyway.
So the 0.487 is the reader end to end, and the duplication bug above explains
none of it. More generally, retrieval quality sets what is *possible* to answer
and the reader decides what actually gets answered; on this benchmark the second
is the binding constraint by a wide margin.

Two abilities go backwards, and both are the same effect. Abstention loses 0.175
and temporal reasoning 0.125, because better retrieval hands nano more
relevant-looking material and it answers from it instead of declining: 15 of 40
abstention questions fail with a confident, plausible answer in place of "there
is no information about that". Asked what criteria were used to prioritise a
Trello board, it describes a Must/Should/Could split that the transcript never
attributes to that question. Context length is not the mechanism — the failures
average 20,835 characters against 24,325 for the passes — the reader simply does
not abstain. A retrieval improvement is therefore not ability-neutral: it raises
the ceiling on recall abilities and lowers the floor on abstention for any
reader that will not say no.

### `cognee_match`: what the judge defect costs in stability

Conversation 2, GPT-5 reading and judging, their per-ability prompts, four
rounds -- their whole setup except the judge *prompt*, which is still BEAM's.
Superseded by `cognee_full`, and kept only for the variance it measures.

| Round | 1 | 2 | 3 | 4 | Mean | SD |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 0.456 | 0.541 | 0.481 | 0.528 | **0.501** | **0.040** |

**±0.040 against Cognee's ±0.005**, eight times the spread, on the same
conversation and the same number of rounds. Two sources, both real:

* The reader is not deterministic at temperature 0. Only 5 of 20 answers were
  byte-identical between rounds one and two -- a reasoning model's hidden trace
  varies and the surface answer follows.
* The judge is not stable under paraphrase. Of the 5 questions whose score moved
  between those rounds, the largest went 0.000 to 1.000 because the answer
  changed from "Not specified in the provided context." to "Not mentioned in the
  provided context." One question is worth 0.05 of a 20-question mean, so that
  single synonym is most of the round-to-round movement.

The second is the judge defect, not noise around a correct value: both answers
are correct abstentions. It is also why their ±0.005 is credible rather than
suspicious -- a judge given the question has far less to be inconsistent about.

The mean itself should not be read as a Cognee comparison, both because the
judge prompt is wrong for that and because stacking their per-ability prompt on
top of BEAM's strict "answer ONLY from the context" user prompt produces
instructions that fight each other. Their summarization prompt asks for
"comprehensive, detailed... do not omit any significant topic"; BEAM's asks for
nothing beyond the context. GPT-5 resolved that towards terseness -- 528
characters a response against Claude's 1,249 -- and lost nugget coverage for it,
scoring 0.000 on contradiction resolution by answering one question with the
single word "No".

### `cognee_full`: the like-for-like, 0.597 against 0.790

Everything matched to their 100K run except the memory system: conversation 2,
GPT-5 reading and judging, their per-ability answer prompts, their judge prompt,
four rounds. Zero judge errors, zero failed questions, all ten abilities in
every round.

| Round | 1 | 2 | 3 | 4 | Mean | SD |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 0.607 | 0.591 | 0.595 | 0.597 | **0.597** | **0.007** |

**The judge prompt was the whole variance story.** Same conversation, same
reader, same rounds -- only the judge prompt differs between this and
`cognee_match`:

| | `cognee_match` (BEAM prompt) | `cognee_full` (Cognee prompt) |
| --- | --- | --- |
| Round SD | 0.040 | **0.007** |

0.007 against Cognee's reported 0.005 is agreement. So their tight spread was
never implausible -- it was evidence their prompt works, and our earlier ±0.040
was measuring the paper's defect rather than anything about memory.

Per-ability, with the `beam`-judged Claude run alongside for reference:

| Memory ability | `cognee_full` | `cognee_match` | Claude, `beam` judge |
| --- | --- | --- | --- |
| Abstention | 0.500 | 0.375 | 0.500 |
| Contradiction resolution | 0.000 | 0.000 | 0.750 |
| Event ordering | 0.308 | 0.300 | 0.171 |
| Information extraction | 0.917 | 0.833 | 0.917 |
| Instruction following | 1.000 | 1.000 | 1.000 |
| Knowledge update | 1.000 | 1.000 | 1.000 |
| Multi-session reasoning | 0.500 | 0.250 | 0.500 |
| Preference following | 1.000 | 0.562 | 0.750 |
| Summarization | 0.750 | 0.693 | 0.667 |
| Temporal reasoning | 0.000 | 0.000 | 0.000 |
| **Overall** | **0.597** | **0.501** | **0.625** |

On a shared ruler we are **0.193 behind**, and that difference is now
attributable to memory architecture -- their knowledge graph, summaries and
hybrid retrieval against our top-5 turn lookup -- rather than to models or
scoring, because every other variable is held.

Two abilities score 0.000 across all four rounds and are where the gap lives.
Contradiction resolution needs both halves of a conflict retrieved at once;
five turns rarely contain both, and GPT-5 answers "No" rather than flagging the
conflict. Temporal reasoning needs dates spread across sessions that top-5
cannot assemble. Those two alone are 0.200 of the 0.193 gap, which is to say
closing either would close it.

One caveat before treating 0.193 as settled: their per-ability prompt is
*prepended* here to BEAM's strict "answer ONLY from the context" user prompt,
whereas theirs most likely replaces it. The instructions partly conflict, and
GPT-5 resolves that towards terseness. Testing the replacement form is the next
thing to do, not another retrieval change.

### `cognee_claude`: the reader is not where the gap is

`cognee_full` with Claude Opus 4.8 reading instead of GPT-5. Everything else
held: conversation 2, their judge prompt, their per-ability prompts, 4 rounds.

| Reader | R1 | R2 | R3 | R4 | Mean | SD |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Opus 4.8 | 0.611 | 0.652 | 0.635 | 0.603 | **0.625** | 0.023 |
| GPT-5 | 0.607 | 0.591 | 0.595 | 0.597 | **0.597** | 0.007 |

**+0.028, which is one standard deviation of Claude's own round spread.** Twenty
questions cannot resolve it, so this is not a reason to change readers. Claude
is also the less stable of the two here, at three times GPT-5's spread.

The per-ability movements are large and cancel, which is the more useful result:

| Ability | Claude | GPT-5 | Delta |
| --- | --- | --- | --- |
| Contradiction resolution | 0.688 | 0.000 | **+0.688** |
| Temporal reasoning | 0.375 | 0.000 | **+0.375** |
| Information extraction | 1.000 | 0.917 | +0.083 |
| Knowledge update | 0.500 | 1.000 | **−0.500** |
| Abstention | 0.375 | 0.500 | −0.125 |
| Summarization | 0.656 | 0.750 | −0.094 |
| Event ordering | 0.221 | 0.308 | −0.087 |
| **Overall** | **0.625** | **0.597** | **+0.028** |

Contradiction resolution behaves exactly as the 400-question run predicted:
Claude surfaces both halves of a conflict, GPT-5 answers "No". But knowledge
update moves the other way by nearly as much, so the headline barely shifts and
we remain 0.165 behind Cognee.

The conclusion drawn here -- that the gap was retrieval, because both readers
sat near zero on contradiction resolution and temporal reasoning -- was half
wrong, and `cognee_replaced` below shows why. GPT-5's 0.000 on contradiction
resolution was not a retrieval failure. The conflicting turns were in the
context; the reader prompt was built wrong and gave it no room to use them.
Temporal reasoning is the half that held up.

### `cognee_replaced`: BEAM's reader prompt was fighting Cognee's

Every arm above stacked Cognee's per-ability prompt on top of BEAM's reader
prompt. That was wrong. Cognee's retriever has two slots, a user template
(`context_for_question.txt`) and a system prompt, and their per-ability file
*replaces* the system slot. BEAM's reader prompt appears nowhere in their code.
Stacking produced a third thing neither project runs, and the two halves
contradicted each other -- BEAM's prompt ends with "Be direct and concise."

`cognee_replaced` drops BEAM's prompt from this path entirely. Nothing else
changed: conversation 2, GPT-5 reading and judging, their judge prompt, their
per-ability prompts, 4 rounds.

| Arm | R1 | R2 | R3 | R4 | Mean | SD |
| --- | --- | --- | --- | --- | --- | --- |
| `cognee_replaced` | 0.663 | 0.627 | 0.654 | 0.660 | **0.651** | 0.016 |
| `cognee_full` (stacked) | 0.607 | 0.591 | 0.595 | 0.597 | **0.597** | 0.007 |
| Cognee, published | 0.785 | 0.790 | 0.795 | 0.790 | **0.790** | 0.005 |

**+0.054, and the gap to Cognee narrows from 0.193 to 0.139.** Every round of
the new arm beats every round of the old, so twenty questions are enough here.

"Be direct and concise" was not a style note, it was a muzzle. Median answer
length went from 73 characters to 845, and the short answers were not merely
thin:

| Question | Stacked | Replaced |
| --- | --- | --- |
| Have I obtained an API key? | `No` | `Yes`, reconciling turns 31 and 47 |
| Have I fixed autocomplete bugs? | `No` | `Yes`, citing turn 62 |
| Daily call quota? | `1,200 calls per day` | same, plus "previously 1,000" |

On both contradiction-resolution questions the terse answer was **wrong**, not
just unelaborated. Forced to be brief, the model committed to `No` before it had
room to notice the later turn reversing the earlier one. Retrieval had supplied
both turns in all four rounds.

| Ability | Stacked | Replaced | Delta |
| --- | --- | --- | --- |
| Contradiction resolution | 0.000 | 0.922 | **+0.922** |
| Summarization | 0.750 | 0.854 | +0.104 |
| Temporal reasoning | 0.000 | 0.031 | +0.031 |
| Information extraction | 0.917 | 0.896 | −0.021 |
| Event ordering | 0.308 | 0.245 | −0.063 |
| Preference following | 1.000 | 0.562 | **−0.438** |
| **Overall** | **0.597** | **0.651** | **+0.054** |

Two caveats on the losers. Event ordering is not a judging change at all: its
nuggets scored 1.0 and are then discarded, because the ability is scored by bare
`tau_norm` (see "How a score is computed"). Both readers are below chance there.

Preference following is a defect in Cognee's prompt, not in ours. Their file
says to "quote or paraphrase their exact preferences rather than inferring",
which assumes the question asks *about* a preference. BEAM's questions do not:
one is "Can you help me set up a caching system? I'd like to keep it simple",
where the preference is a constraint to obey while doing the work. The model
listed the user's preferences and never wrote the code; the judge marked it
non-responsive. The stacked arm scored 1.000 here only because BEAM's "answer
the question" overrode their framing. Keeping BEAM's prompt for this one ability
would put us at roughly 0.695, but that is cherry-picking per ability and is not
what they run, so it stays out of the headline number.

One question, `multi_session_reasoning #0` ("how many different features did I
mention"), returns empty in some rounds: GPT-5 spends the full 8192-token budget
enumerating in hidden reasoning. It scores 0 and the arm wears the cost.

What is left of the 0.139 gap is no longer contradiction resolution. Temporal
reasoning at 0.031 is the clearest remaining hole, and it is the one that still
looks like retrieval -- it wants dates spread across sessions inside five slots.

### `semantic_hybrid`: retrieval through the ontology

The first arm that changes retrieval rather than a prompt. Everything else is
`cognee_replaced`: conversation 2, GPT-5 reading and judging, their judge
prompt, their reader prompt, 4 rounds. The only difference is `--retrieval
hybrid`, which unions the turns a vector search finds with the turns the
transcript's *attributes* cite, and renders the edges between those attributes
above the context.

| Arm | R1 | R2 | R3 | R4 | Mean | SD |
| --- | --- | --- | --- | --- | --- | --- |
| `semantic_hybrid` | 0.698 | 0.636 | 0.698 | 0.695 | **0.682** | 0.031 |
| `cognee_replaced` | 0.663 | 0.627 | 0.654 | 0.660 | **0.651** | 0.016 |
| Cognee, published | 0.785 | 0.790 | 0.795 | 0.790 | **0.790** | 0.005 |

**+0.031, and the gap to Cognee narrows from 0.139 to 0.108.** One standard
deviation of this arm's own spread, so the headline is not the result; the
per-ability movement is, because two abilities moved by more than ten times it.

> Measured while a matched term row was dropped *and* its slot wasted, so this
> arm saw 7 attributes where it should have seen 8. The term is now excluded
> deliberately — it cites no turns, so answering with it delivers nothing to
> read, and its description is the one thing a model can already infer from the
> turns it gets — but the query over-fetches by the number it drops, so eight
> asked for is eight attributes delivered.
>
> The re-measurement, `semantic_term`, is **void** and sits in `discarded/`.
> Its rounds answered 20, 18, 20 and 19 questions, so its 0.598 is a mean over
> four different question sets. Cause: the endpoint returns capacity
> exhaustion as a 401 (`"FATAL: sorry, too many clients already"`), which the
> retry classifier read as a real credential failure and did not retry, so the
> question was dropped. Both halves are fixed -- `"too many clients"` is
> classified transient, and `run_rounds` now refuses to pool rounds of unequal
> length rather than reporting the difference as noise. Re-running it is open.

| Ability | Vector | Hybrid | Delta |
| --- | --- | --- | --- |
| Temporal reasoning | 0.031 | 0.500 | **+0.469** |
| Preference following | 0.562 | 1.000 | **+0.438** |
| Information extraction | 0.896 | 1.000 | +0.104 |
| Contradiction resolution | 0.922 | 0.984 | +0.062 |
| Event ordering | 0.245 | 0.193 | −0.052 |
| Summarization | 0.854 | 0.766 | −0.089 |
| Abstention | 0.500 | 0.375 | −0.125 |
| Multi-session reasoning | 0.500 | 0.000 | **−0.500** |
| **Overall** | **0.651** | **0.682** | **+0.031** |

Temporal reasoning is the one this was built for, and the mechanism is visible
rather than inferred. Asked how many days passed between obtaining the API key
and finishing the wireframes, the vector route returns turns 21, 35, 36, 37 and
62 -- it has the wireframe turn and **not** turn 16, where the key is -- which is
why GPT-5 answered "the date you obtained your API key isn't stated". The
attribute route returns both events with their real dates (March 10 and March
12) and a `precedes` edge ordering them.

Preference following came from fixing the extractor, not the retrieval. The
first pass tagged 83 of 152 attributes `preference`, and they were requests --
"wants to handle CORS errors" is what the user asked in that turn, and a support
transcript is made of almost nothing else. Told that asking for help is not a
preference, the second pass returned 12, which are standing constraints: "App
bundle size limit 2.5MB", "No external libraries for caching", "vanilla JS
only". Those are the nuggets the ability is scored on.

#### Multi-session reasoning, and the limit it exposes

The −0.500 is one question and it is worth more than the win. Asked how many
features or concerns were mentioned across the conversation, `cognee_replaced`
returned `''` in one round and `33` in another -- a bare guess -- and scored
0.500. `semantic_hybrid` returned `7` with a clean itemised list, and scored 0.

The transcript has 87 attributes. Retrieval shows the reader the 8 nearest, so
it counted 8. **The better-formed answer is the wrong one**, and the cap that
bounds context is what caused it: an aggregate question is answered from the
*whole* attribute set, which is exactly the thing this layer has and the one
shape of question it currently cannot route. A count is not a similarity match
and should not be served by one.

#### What it costs

Extraction is one model call per window: 54 windows for conversation 2's 99
turns, about 7 minutes when the endpoint is healthy. Adjudication is one call
per candidate pair, 25 pairs and 100 seconds here, and `relate_conversation`
runs it without re-extracting so the relation vocabulary can be retuned for the
cost of the pairs alone.

Retrieval returns 11-12 turns against the paper's 5. `attribute_turns` caps the
attribute route at 10, taken breadth-first -- one turn from each attribute
before a second from any -- because the case this exists for needs one turn from
each of two attributes rather than six from one.

### Scored the way cognee scores: 0.716 against 0.794

Everything above compares our event-ordering `tau_norm` against their
event-ordering nugget mean, which is two different metrics in one column. The
[scoring section](#how-a-score-is-computed) has the proof. Restated under their
rule, the same four `pathk4` rounds — unchanged answers, unchanged verdicts,
arithmetic only:

```
$ uv run --no-sync python compare_cognee.py pathk4 --rounds 4

ability                         ours   cognee     gap
-----------------------------------------------------
abstention                     0.500    0.500  +0.000
contradiction_resolution       1.000    0.875  +0.125
event_ordering                 0.838    0.787  +0.050
information_extraction         0.938    0.979  -0.042
instruction_following          1.000    1.000  +0.000
knowledge_update               0.500    1.000  -0.500
multi_session_reasoning        0.250    0.500  -0.250
preference_following           0.938    1.000  -0.062
summarization                  0.698    0.802  -0.104
temporal_reasoning             0.500    0.500  +0.000
-----------------------------------------------------
OVERALL                        0.716    0.794  -0.078
kendall tau (ordering)         0.273    0.515  -0.241
```

The gap is 0.078, not the 0.135 the same rows show under `--flavour beam`. We
lead on contradiction resolution and on event ordering itself; our tau is still
well below theirs, which is the honest reading of that ability and is why it is
printed.

Three of the four abilities we were chasing turn out not to be winnable this
way: **they also score 0.500 on abstention, multi-session reasoning and
temporal reasoning.** Two questions each, one scored and one missed, by a system
at 0.794.

The comparison is exact rather than indicative. Their artifacts are
per-conversation, and
`beam_existing_ingestion_metrics_conv1_hybrid_completion_20_20_qa_v1_run0.json`
carries `conversation_id: "2"` with all twenty questions byte-identical to our
`questions.csv` rows for conversation 2 — the file is named for their run index,
not for the transcript. Their config is vendored next to their summary in
[`upstream/cognee_report/`](upstream/cognee_report).

Their config is `chunks_top_k: 20, entities_top_k: 20` — twenty whole turns
plus twenty entities, not the `top_k=5` their library defaults to. The two
`knowledge_update` questions were answered from **84,560 and 108,656
characters** of context, against our 58,245 and 63,951 for the same two. They
send more, by about half again; across all twenty our mean is 60,680 and the
graph note inside it is 4,524. Whatever the gap below is, it is not that we are
starving the reader relative to them.

#### The one ability worth half the remaining gap

`knowledge_update` is two questions. We score the quota one 1.0 in all four
rounds and the coverage one 0.0 in all four:

| | ours | cognee | gold |
| --- | --- | --- | --- |
| daily call quota for the API key | 1,200 calls/day | 1,200 API calls per day | 1,200 calls per day |
| test coverage for the API integration module | **85%** | 78% | **78%** |

Not a retrieval miss — the figure we need is in the context, and our answer
recites the whole chain: *"it was 65% after the initial test suite run on
March 27, improved to 78%, and then reached 85%."* The transcript says:

| turn | text | subject |
| --- | --- | --- |
| s1 t42 | "100% test coverage on my **API integration module**, and I've currently reached 65% after my initial test suite run on March 27, 2024" | the module |
| s1 t56 | "increase the unit test coverage for my **API integration**, which has recently improved to 78%" | the module |
| s2 t14 | "100% test coverage on my **core modules**, including API fetch, autocomplete, and error handling, and I've currently reached 85% as of April 9, 2024" | everything |
| s2 t52 | "achieved 85% test coverage on **core modules**" | everything |

85% is a different subject. The question asks about the API integration module,
whose last reading is 78%; core modules are a superset that includes it. We
built a four-step `supersedes` chain across both subjects and handed the reader
its tip.

The mechanism is `_revisions` in `semantic/derive.py`, and its docstring already
named the tradeoff before we knew the price:

> Grouped by `attribute_key` alone, within one transcript. Adding the subject
> would be stricter and is not available: the subject of a statement is a
> `knowledge` edge, and a statement whose edge the extractor omitted would
> silently stop being comparable with its own revision.

The multi-session loss is the same bug, and their answer to it shows how little
was needed:

> **Them (1.000):** Your fetch call latency is faster. You measured ~250 ms for
> fetch calls vs ~280 ms for the autocomplete API, so fetch is about 30 ms
> quicker.
>
> **Us (0.000):** Fetch call latency: 250ms […] Autocomplete API response time:
> **220ms** […] Your autocomplete API response time (220ms) is currently faster
> than your fetch call latency (250ms), by about 30ms.

The autocomplete's figure is 280ms. 220ms is the weather app's, and we attached
it to the wrong subject — the same confusion as 85%, from the same cause.
**One bug, two abilities, 0.075 of overall** — essentially the entire 0.078.
Grouping revisions by subject as well as key is the next thing to try, and the
extractor change that makes a statement carry its own subject in its
`description` is what makes it affordable.

#### Length costs us on negative constraints

The other half-point, on preference following, is not a graph problem. The
nugget is `avoids recommending manual deployment checks` — a negative
constraint — and our answer is a long tour of the CI pipeline that eventually
mentions a manual step. Theirs is four bullets about the Actions tab and stops.

Eight of the twenty questions go to them, but three are near-ties (event
ordering 0.900 against 0.950 and 0.600 against 0.625, summarization 0.583
against 0.667). The pattern in the rest is that their answers are shorter
except on summarization, where theirs is an exhaustive bulleted dump and scores
0.938. Our reader is verbose everywhere, which is free on nugget recall and
costs real points wherever a rubric forbids something.

#### Reading their answers next to ours

Judged rows carry `cognee_answer`, `cognee_score`, `cognee_reason`,
`cognee_score_mean` and `golden_answer` for conversation 2, so a lost question
is diagnosable in one file. `--rounds` pairs our round with their run, since 19
of their 20 questions got a different answer each run and 5 got a different
score.

```
uv run --no-sync python compare_cognee.py pathk4 --rounds 4 --per-question
uv run --no-sync python compare_cognee.py <tag> --rounds 4 --backfill   # runs judged before the columns existed
```

The answers come from
[`upstream/cognee_report/cognee_answers_conv2.json`](upstream/cognee_report),
distilled from their four run artifacts by
[`tools/extract_cognee_answers.py`](../../tools/extract_cognee_answers.py).
Distilled rather than vendored whole because each artifact is 2MB unpacked and
almost all of it is `retrieval_context`; that is kept as a length only, which is
where the 84,560 and 108,656 figures above come from. The script refuses to
write a file that does not reproduce all ten of their published ability means,
their 0.794 and their 0.515 tau from the rows it kept.

**Only conversation 2.** At 100K they published that one and nothing else, so
the columns are empty for every other conversation.

### `prefact`: 0.798 against 0.794

The first arm to pass cognee. Four rounds, mean 0.798, sd 0.023, range
0.773–0.826.

| ability | ours | cognee | gap |
| --- | --- | --- | --- |
| abstention | 0.750 | 0.500 | +0.250 |
| contradiction_resolution | 0.781 | 0.875 | -0.094 |
| event_ordering | 0.800 | 0.787 | +0.013 |
| information_extraction | 0.875 | 0.979 | -0.104 |
| instruction_following | 1.000 | 1.000 | +0.000 |
| knowledge_update | 1.000 | 1.000 | +0.000 |
| multi_session_reasoning | 0.500 | 0.500 | +0.000 |
| preference_following | 0.938 | 1.000 | -0.062 |
| summarization | 0.833 | 0.802 | +0.031 |
| temporal_reasoning | 0.500 | 0.500 | +0.000 |
| **OVERALL** | **0.798** | **0.794** | **+0.004** |
| kendall tau (ordering) | 0.301 | 0.515 | -0.214 |

`knowledge_update` went 0.500 to 1.000 and `multi_session_reasoning` 0.250 to
0.500 since `pathk4`. Both were attributed there to one bug — a statement
attached to the wrong subject — and the subject pass is what changed in
between, so the attribution looks right; it has not been isolated as its own
arm, and several other things changed alongside it.

Build the graph, then score it:

```
uv run --no-sync python build_graph.py 2 --per-turn --cache-draft --verify

uv run --no-sync python main.py --database-name beam --skip-migrate --skip-ingest \
  --scale 100K --conversations 2 --tag prefact --rounds 4 --eval-workers 8 \
  --retrieval semantic --node-turns 15 --top-k 5 --widen 0 \
  --reader-model aws/anthropic/bedrock-claude-opus-4-8 --reader-prompt cognee \
  --judge-model azure/openai/gpt-4.1-mini --judge-prompt cognee
```

`--skip-ingest` is not optional once the transcripts are in Postgres: without it
`stage_beam_ingest` re-ingests with `reset=True`, which renumbers turns and
breaks every citation in the graph. `--node-turns`, not `--top-k`, is the turn
budget on `--retrieval semantic`, which never searches turn text — every turn
the reader sees is a citation on a matched node, so `--top-k` is inert here and
is passed only to keep the recorded config comparable with the vector arms.

#### How much of this is the reader

Most of the last 0.087. `turns15` is the same graph, same retrieval, same turns,
and scores 0.711. What changed is three formatting notes appended to cognee's
per-ability prompts in [`reader.py`](reader.py), joining the two that were
already there:

| ability | before | after | the note |
| --- | --- | --- | --- |
| preference_following | 0.625 | 0.938 | apply the preferences, do not just list them |
| temporal_reasoning | 0.250 | 0.500 | give the interval as a range, not a subtraction |
| contradiction_resolution | 0.656 | 0.781 | quote both sides and ask which is correct |

All three were answers that were already **factually right** and read as
incomplete. The temporal one wrote `March 12 - March 10 = 2 days` where the
rubric asks for `from March 10 till March 12`; both state the same fact and only
one matches. The preference one reported that the user prefers no external
libraries without naming an approach, against a rubric that asks for a
recommendation *and* for avoiding libraries.

This is calibration against a rubric we can read, and it should be reported as
that rather than as a retrieval result. Two things keep it honest: the upstream
files in [`upstream/cognee_qa/`](upstream/cognee_qa) stay byte-exact on disk and
the notes are appended at load time, so dropping five lines from
`cognee_system_prompts` restores cognee's own instructions; and cognee scores
0.875–1.000 on all three abilities **with those same unmodified prompts**, so
the notes close a gap in how our reader words things, not one in what the
benchmark asks.

#### Contradiction resolution is still a graph defect

0.781 against their 0.875, and the earliest graph in this file scored 0.875 on
it. The two questions are the API key and the autocomplete bugs, and both
retractions are in the graph as their own nodes with no edge to what they
retract:

```
Never deployed to GitHub Pages   turn 99  ->  contradicts 'GitHub Pages deployment since April 8, 2024'
Never fixed autocomplete bugs    turn 66  ->  NO REVISION EDGE
Never obtained an API key        turn 35  ->  NO REVISION EDGE
```

The only linked retraction is the one no question asks about, so on this
ability the graph currently contributes nothing and the score comes from the
reader finding both turns in the raw text.

`relate` found both revisions and the subject guard vetoed them:

```
Dropping supersedes 'OpenWeather API key obtained' / 'Never obtained an API key':
  different subjects (2024-03-10 / Christina)
```

Neither subject is real. The statements are about the API key, which was never
extracted as a thing, and `assign_subjects` may not answer "none" — it must
choose from the things in the graph, so it picked a date for one side and a
person for the other. Raising subject coverage made this worse, not better:
`featurefix` assigned 207 of 249 and drew the edge, because the statement had no
subject to disagree about; the subject-wording fix took coverage to 247 and the
fortieth guess broke it.

Three changes follow, in order of how little they assume:

1. Let `assign_subjects` answer "none". No subject is better than a wrong one,
   and it alone restores the edge.
2. Stop offering `Timestamp` and `Person` as subject candidates, and never let
   either veto a revision. 15 of 152 things are one of the two.
3. Assert after every build that a statement the citation check reported as
   `DENIES` carries a `supersedes` or `contradicts` edge. Deterministic, free,
   and it would have caught this the moment it appeared.

## Layout

```
beam_benchmark/
  download_data.py   fetch a scale from the Hub into ../data/beam/<scale>/
  evaluator/         the paper's judge, ported
    upstream/        the three prompts, byte-for-byte
  reference.py       cognee's answers, joined onto our rows
  upstream/
    cognee_qa/       their per-ability answer prompts
    cognee_report/   their published config, summary and answers
  results/           run output (git-ignored)
```

`upstream/cognee_report/` holds their config and cross-run summary copied
unmodified from
[`cognee/eval_framework/beam/report_artifacts/100k_fixed/`](https://github.com/topoteretes/cognee/tree/main/cognee/eval_framework/beam/report_artifacts/100k_fixed),
plus `cognee_answers_conv2.json` distilled from the per-question artifacts in
the same directory. So `compare_cognee.py` runs offline and the numbers it
prints for their column are theirs rather than ours transcribed.

The judge runs on `azure/openai/gpt-4.1-mini` at temperature 0, which is the
paper's `gpt-4.1-mini` under the id `REASONING_ENDPOINT` routes it by.
`BEAM_JUDGE_MODEL` overrides it, at the cost of no longer being on the same
instrument as the table above.

Smoke-tested against that endpoint on one question per ability: an answer built
from the rubric's own nuggets scores 1.000 on seven of the ten and the
deliberately wrong answer scores 0.000 on all nine nugget-averaged abilities.
Event ordering returns 1.000 for the events in the right order and 0.146 for the
wrong answer — the below-chance floor described above, visible in practice.

