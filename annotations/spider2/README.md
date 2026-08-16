# Spider2 durable annotations

These files outlive `datasets/spider2/` (gitignored / re-seeded). Point `.env` at them:

```bash
SAVED_CUSTOM_ANALYSES_DIR=annotations/spider2/custom_analyses
SAVED_METADATA_DIR=annotations/spider2/metadata
SAVED_DESCRIPTIONS_CSV=annotations/spider2/semantic_descriptions.csv
```

| Path | Applied when | Contents |
|------|----------------|----------|
| `custom_analyses/<db>.json` | ingest (`add_custom_analyses`) | Custom analyses |
| `metadata/<db>.json` | ingest (`apply_metadata`) | Table/column description overlays |
| `semantic_descriptions.csv` | semantic `--override-descriptions` | Column / ColumnAttribute text |

To refresh from the working copies under `datasets/spider2/`, re-copy the JSON
files (or re-run the export used when this tree was created).
