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
see [the ingestion guide](../ontology_sql_eval/ingestion/README.md).
