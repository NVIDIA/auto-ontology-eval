# Dataset file formats

Each dataset lives in `datasets/<name>/`. The folder name is the dataset name
that `main.py --database-name` selects.

```
datasets/<name>/
  evaluation.json        # questions and gold SQL (required)
  metadata.json          # table and column descriptions (optional)
  custom_analyses.json   # named example analyses (optional)
```

The examples below use `wideworldimporters`.

## `evaluation.json`

A JSON array of question objects. The benchmark reads these fields:

| Field         | Required | Purpose                                                              |
| ------------- | :------: | -------------------------------------------------------------------- |
| `question_id` |    no    | ID both arms are joined on. Defaults to a hash of the question.      |
| `db_id`       |   yes    | Database the question runs against.                                  |
| `question`    |   yes    | Natural-language question sent to both arms.                         |
| `evidence`    |    no    | Hint text appended to the question.                                  |
| `SQL`         |   yes    | Gold SQL.                                                            |
| `difficulty`  |    no    | Groups accuracy by difficulty in the report.                         |

```json
[
  {
    "question_id": 3,
    "db_id": "WIDEWORLDIMPORTERS",
    "question": "calculate the customer count by state province name",
    "evidence": "",
    "SQL": "select sp.stateprovincename as state_province_name, count(c.customerid) as customer_count from sales.customers c join application.cities ct on c.deliverycityid = ct.cityid join application.stateprovinces sp on ct.stateprovinceid = sp.stateprovinceid group by sp.stateprovincename order by state_province_name desc;",
    "difficulty": ""
  }
]
```

## `metadata.json`

An object keyed by table name. Ingestion uses it to add descriptions and sample
values to Auto-Ontology's catalog, and skips this step when the file is absent.

```json
{
  "sales.customers": {
    "description": "One row per customer account.",
    "columns": [
      {
        "name": "customername",
        "description": "Display name of the customer.",
        "value_examples": ["Tailspin Toys", "Wingtip Toys"]
      }
    ]
  }
}
```

## `custom_analyses.json`

An array of named example analyses. During ingestion, each one is parsed against
the ingested schema, added to the ontology as a custom analysis, and embedded
for retrieval.

```json
[
  {
    "name": "Americas Region Countries",
    "description": "Countries that belong to the Americas region.",
    "sql": "select region from application.countries where region like '%Americas%'"
  }
]
```

For BIRD, the repository's own descriptions and analyses live in
`inputs/annotations/` and take precedence over the downloaded dataset's copies;
see [the ingestion guide](../auto_ontology_eval/ingestion/README.md).

## Conversation corpora

Transcripts live under `datasets/` too, one directory per corpus and one
subdirectory per revision — a scale for BEAM, an export for Slack. Both are
git-ignored: BEAM is regenerable and large, and the Slack export is not
redistributable at all, so the seed script is the thing worth reviewing rather
than its output.

```
datasets/beam/<scale>/
  conversations/<id>.json   # role-tagged sessions, fetched by scripts/seed/seed_beam.py
  source.json               # what was fetched, and the hash that pins it
datasets/slack/<revision>/
  export/<channel>/<channel>.json   # pinned by scripts/seed/seed_slack.py
  source.json
  questions.csv             # the hand-checked question set
```

The subdirectory is never named by a caller. Each format's reader owns its own
layout, the way a SQL connector owns its dialect, so ingesting any corpus takes
a name and a revision and nothing else:

```bash
uv run python -m auto_ontology_eval.ingestion.conversation_ingest slack
uv run python -m auto_ontology_eval.ingestion.conversation_ingest beam --revision 100K
```

One conversation is one memory scope, named `<corpus>_<revision>_<id>`. The
scope is what a vector search filters on, so pooling a workspace would let a
question about one channel be answered from another. The revision is in the
name because turn indices only mean anything against the export that produced
them — which is also why `questions.csv` sits beside the export it cites
rather than with the code.

### BEAM

Synthetic and two-party: one user, one assistant, across sittings. Scored the
way cognee scored it, against their published **0.794** — their judge prompt,
judge model and four rounds — so none of those is a command-line flag. See
[`auto_ontology_eval/beam/cognee/config.py`](../auto_ontology_eval/beam/cognee/config.py).

### Slack

Eleven NVIDIA channels, 9,316 threads and 104,831 messages from May 2021 to
November 2026, in which a dozen people talk to each other. The questions worth
asking of it are the ones BEAM cannot pose: who reported a thing, who was asked
to fix it, who was in the room. There is no published judge for it, so its
numbers are internally comparable only and are not quotable the way BEAM's are.

**A thread is one turn.** The export arrives thread-shaped — its `text` is the
whole exchange with every message labelled by its author — and a thread is the
unit people reply within, so it is also the unit worth retrieving. The root
message becomes the turn's question and the replies its answer, which is a mild
fiction since a thread's repliers are not an assistant. It survives because the
author stays inside the text, so a model reads back `question: Chris Workman:
…` and `answer: Reply Dmitriy Marshak: …` rather than two anonymous blocks. It
is still a fiction, and the reason a turn should eventually hold attributed
messages instead of two sides.

**A channel is one conversation and a day is one session.** The day is the only
boundary the format offers, and it carries the meaning a session is needed for:
a value restated a week later is a revision, while the same words half an hour
later are a correction. It also puts the sittings on a timeline for
`days_since_start`.

Real names are kept. The point of this corpus is to ask what specific people
said, and a pseudonym cannot be checked against a memory of the person.
Credentials disclosed in a thread are redacted on parse; see
`auto_ontology.conversations.sources.slack`.
