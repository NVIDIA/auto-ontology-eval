# BEAM: where we stand against Cognee

**0.856 against Cognee's 0.794.** Four rounds, conversation 2 at 100K,
2026-10-28. The context figures below are from the 0.835 run of 2026-10-25 and
have not been remeasured.

## What BEAM is

BEAM tests whether a system can **remember a long conversation**. It is not a
retrieval benchmark over documents -- the input is a chat history.

The input we run is one user talking to an assistant across **3 sessions, 200
turns, 493,000 characters** (~123k tokens, hence "100K") while building a
weather app. Over those sessions the user changes their mind, contradicts
themselves, sets preferences, and schedules things by date.

Then 20 questions are asked about that history. **No single turn contains the
answer** -- each one requires joining facts stated far apart, or noticing that
two turns disagree. A worked example:

> **Question:** Have I obtained an API key for this project?
>
> The user says in one session that they have an API key, and in a later
> session that they never obtained one. The correct answer is not yes or no --
> it is to notice the conflict, quote both sides, and ask the user which holds.

So the system must ingest 200 turns, store them in some structure, and retrieve
the right handful on demand. That is the whole task.

The 20 questions cover **10 abilities, two questions each**: abstention,
contradiction resolution, event ordering, information extraction, instruction
following, knowledge update, multi-session reasoning, preference following,
summarization, temporal reasoning.

## How it is scored

Not by string match. Each question ships with a rubric of 2 to 5 **nuggets** --
the specific things a correct answer must say. The API-key question above
carries exactly four:

```
1. state: there is contradictory information
2. mention: you said you have an API key for the project
3. mention: you also mentioned that you have never obtained one
4. mention: which statement is correct?
```

A judge model reads our answer and marks each nugget full, partial or none. The
question's score is the mean over its nuggets -- naming the conflict but
forgetting to ask the user back scores 0.75, not 1.0. The headline figure is the
mean over all 20 questions.

We run Cognee's judge prompt and judge model (`gpt-4.1-mini`) unchanged, so
0.856 and 0.794 are produced by the same instrument and are directly
comparable.

## Where Cognee's numbers come from

Their own published results, not our re-run of their system. They commit the
raw per-question artifacts to their repo:

- [cross-run summary](https://github.com/topoteretes/cognee/blob/main/cognee/eval_framework/beam/report_artifacts/100k_fixed/hybrid_completion_20_20_qa_v1_cross_run_summary.json)
  -- their own headline: `beam_rubric.mean` **0.794375**, `run_std` 0.0046, and
  a per-ability breakdown.
- [per-question artifacts](https://github.com/topoteretes/cognee/tree/main/cognee/eval_framework/beam/report_artifacts/100k_fixed)
  -- `beam_existing_ingestion_metrics_conv1_hybrid_completion_20_20_qa_v1_run{0..3}.json.gz`,
  four runs of 20 questions, each row carrying their answer, the rubric, the
  judge's score and reason, and the full retrieval context they fed the reader.

We recomputed 0.7944 from those four files and it matches their summary. A
copy of the per-question data is vendored at
`upstream/cognee_report/cognee_answers_conv2.json`, checked against the
originals rather than trusted.

One asymmetry worth stating plainly: **their score is the one they published
for their own answers; ours is our answers scored by us.** We never re-judged
their output. The comparison holds because we run their judge prompt and judge
model unmodified, which is why that is not something we tune.

The context sizes are the only figures we derived rather than read off. Both
sides count **retrieved material only** -- their `retrieval_context` field and
our note-plus-turns block -- measured before either is substituted into a
prompt, so neither number includes the system prompt or reader template.
Counted over all 80 question-rounds on each side.

## Results

| round | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|
| ours | 0.846 | 0.841 | 0.830 | 0.907 | **0.856** +/- 0.035 |
| cognee | | | | | **0.794** |

| ability | ours | cognee | |
|---|---|---|---|
| abstention | **1.000** | 0.500 | we win |
| multi-session reasoning | **0.625** | 0.500 | we win |
| contradiction resolution | **0.969** | 0.875 | we win |
| instruction following | 1.000 | 1.000 | tie |
| knowledge update | 1.000 | 1.000 | tie |
| preference following | 1.000 | 1.000 | tie |
| information extraction | 0.979 | 0.979 | tie |
| event ordering | 0.787 | 0.787 | tie |
| summarization | 0.760 | **0.802** | we lose |
| temporal reasoning | 0.438 | **0.500** | we lose |

The spread is wider than the 0.835 run's and the mean is higher: one round of
20 questions moves 0.05 per question, so +/- 0.035 over four rounds puts the
true mean somewhere around 0.82 to 0.89. The defensible claim is that this sits
clearly above our previous band, not that it is 0.856 exactly.

### What this build changed

Statements now carry a `request` kind for a concern the user asked to have
handled -- a failure or an edge case, as against asking for help building
something -- and the subject pass files those on the person who asked rather
than on the part they asked about. 54 of them, all on the user.

Two things followed. `abstention` went to **1.000, eight correct out of
eight**, because the `[proposed]` tag is now both present and acted on: the
gloss tells the reader that if the only lines bearing on "what did I enforce"
are tagged that way, the answer is that the transcript does not say. Earlier
builds had the tag and the reader recited the rules anyway, then lost the tag
entirely and recited them with no caveat at all.

And `information_extraction` reached parity at 0.979 despite `proposed` rising
from 10 to 146 nodes, which was predicted to make the reader timid and did not.

### What is still open

Per-question means over the four rounds:

| | question |
|---|---|
| 0.00 | how many days between scheduling the meeting and the testing period |
| 0.25 | how many features or concerns did I mention wanting to handle |
| 0.71 | comprehensive summary of how the project progressed |
| 0.79 | order in which I brought up aspects of autocomplete |

The counting question scored 1.0 in round 4 and 0.0 in the other three, on
answers that do not differ structurally -- that 0.125 is judge variance, not a
fix. Its gold is the single word "Four" with no evidence pointer, and the graph
holds at least eight concerns she asked to have handled, each defensible. The
temporal question is the one both systems fail.

## Us against Cognee

| | ours | cognee |
|---|---|---|
| score | **0.856** | 0.794 |
| context per question, mean | **77,296** chars | 98,639 chars |
| smallest / largest question | 11,345 / 97,340 | 63,748 / 135,814 |
| context per round of 20 | **1.55 M** chars | 1.97 M chars |

We answer better on **22% less context**. Ours also sizes itself to the
question -- an eightfold spread from the cheapest to the dearest -- where
theirs is a fixed 20 passages plus 20 entities every time and never drops
below 64k, however simple the question.

Our figure is identical in all four rounds because retrieval is deterministic:
same question, same graph, same nodes, and only the reader's wording varies.

## The two graphs

**Ours** -- one typed graph over the transcript: 401 nodes, 778 edges.

```
about 247   precedes 234   built_with 124   refines 62
has_component 50   supersedes 18   contradicts 6
```

458 edges are extracted from the text, 217 derived (chronology from dates,
supersession from repeated attributes). Every node has a canonical name, a
type, and the turns that state it. **Zero duplicate names in 401 nodes.**
Retrieval returns on average 20 nodes and 67 edges per question.

**Theirs** -- chunks plus an entity layer. Exactly 20 text passages (each an
LLM summary, a fact list, and the full raw turn) and exactly 20 entities per
question. Names are uncanonicalised -- one node is `decision to standardize
code style with eslint + prettier (airbnb, prettier v2.8) and enable
format-on-save` -- and `airbnb javascript style guide` appears **twice in the
same context**. 28 duplicate entity names across 15 of the 20 questions.

They have no supersession or contradiction relations at all; "supersedes" and
"retracted" appear zero times in any of their contexts. They re-derive recency
from timestamps in the raw text on every question.

## Tokens

**Cognee publishes none.** The artifacts linked above hold the answer, the
score, the rubric and the retrieval context, and no token, cost or usage field
of any kind -- searching every row key for `token`, `cost` or `usage` returns
nothing.

Context characters are the only like-for-like measure available, and we send
22% fewer. At roughly 4 characters per token that is about **19,000 tokens per
question for us against 25,000 for them** -- but that is our arithmetic on
their text, not a number they report.

## Concrete errors -- ours

| what | status |
|---|---|
| One reversed edge in 778: `Loading spinner --has_component--> Autocomplete` sits alongside its correct inverse. Logically impossible, catchable with no model. | **open** |
| Two retractions carry no revision edge: `Never fixed autocomplete bugs` and `Never obtained an API key` are stored as free-standing facts, not corrections. | **open** |
| A multi-session question answers "three" features where the answer is "Four". A genuine undercount. | **open** |
| An answer that reasons out loud in the output ("Wait -- let me reconsider"). Poor output irrespective of score. | **open** |
| A summary question missed a component that was in the graph, because an edge was dropped when only one end was retrieved. | **fixed** -- 0 of 4 rounds to 4 of 4 |
| An ordering question missed content because 6 of 10 retrieval slots went to near-duplicate paths from 2 nodes. | **fixed** -- 13 rows now return 28 nodes instead of 7 |

11 of 600 nugget verdicts cite an incorrect claim, and all but two trace to the
single temporal question described below.

Contradiction resolution is the one ability that moved the wrong way: it scored
1.000 across all 16 question-rounds of the two preceding arms and fell to 0.781
here, after one reader-prompt sentence was reworded to the rubric's own
phrasing. That wording is already changed back; a confirming run is queued.

## Where the judge is wrong, and we are right

**1. A true answer scores below a shorter one.** Which technologies did the
project use:

```
ours    Vanilla JavaScript ES2021, HTML5, and CSS3 (with axios as the HTTP client library).
cognee  Vanilla JavaScript (ES2021), HTML5, and CSS3.
```

Theirs 1.0, ours 0.5, because ours "also mentions additional technologies:
HTML5, CSS3, and axios". **HTML5 and CSS3 are nuggets of the same rubric,
scored 1.0 each in the same pass.** The only content separating the two answers
is `axios`, which the transcript states. We are marked down for being more
complete; the only fix is to stop saying something true.

**2. Kendall tau measures phrasing, not order.** The scorer asks the judge
whether two snippets describe the same event:

```
First snippet:  Dropdown and error handling
Second snippet: Handling the 5-item dropdown limit and error handling for invalid API key
```

It answers **NO**. Four of our five items then match nothing and the fifth
matches the wrong label. With nothing matched tau is a fixed 0.143, which is
why that exact figure repeats every round. Our items were in correct
chronological order throughout -- turns 10, 37, 41, 70. Cognee writes short
label-like lines the classifier recognises, and scores 0.515. We report tau
beside the score as they do, and do not treat it as a target.

**3. The same facts scored 0.0 and 1.0.** On "how do I track each deployment
step", all four of our answers named GitHub Actions in the first sentence. The
three that opened on configuring it scored **0.0** -- "does not mention any
automated workflow monitoring tools" -- and the one that opened on what it does
by itself scored **1.0**.

**4. Two questions neither system can win.** A temporal question asks the days
between a meeting and a testing period. The rubric wants 21, reading "two weeks
of testing before the April 5 deadline" as testing starting April 5. Both
systems instead read the start date off the project schedule in the transcript:
we answer 7, Cognee answers 10, both score 0.00 every round. **The practical
ceiling on this transcript is about 0.90, not 1.00.**
