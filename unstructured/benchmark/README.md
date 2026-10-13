# ViDoRe v3 HR benchmark — the chunk-search baseline

A copy of the `nemo-retriever-rag` project, pointed at the ViDoRe v3 HR corpus
in `unstructured/data/vidore_v3_hr`, with the results of the published run.

This folder holds the baseline only: chunks in LanceDB, searched directly.

    query → chunk vectors (LanceDB) → chunks → answer

It is what the attribute pipeline in `unstructured/eval_attributes.py` is
measured against. Nothing here touches auto-ontology's Postgres. Extraction and
embedding go through NeMo Retriever; generation goes through the hosted NVIDIA
endpoints.

| File | Role |
| --- | --- |
| `retrieve_generate.py` | The baseline: LanceDB chunk search, then answer |
| `ingest.py` | PDFs to chunks in LanceDB (already done, `lancedb/` is shipped) |
| `main.py` | Ask one ad-hoc question against the ingested corpus |
| `remove.py` | List or drop ingested documents |
| `config.py` | LanceDB paths and helpers for the three scripts above |
| `setup.sh` | Activate the NeMo Retriever venv |

Everything the baseline shares with the attribute pipeline — generator, prompt,
token budget, reasoning cap, pacing, page-key arithmetic, graded qrels, scoring
tiers, report format — is in `unstructured/pipeline.py`, not here. That is what
makes retrieval the only difference between the two reports. The LLM judge is a
third stage, `unstructured/judge.py`, which scores both sides.

## Running

NeMo Retriever pins versions that clash with this repo's environment, so it
has its own venv:

```bash
source unstructured/benchmark/setup.sh

python -m unstructured.benchmark.retrieve_generate --limit 10     # pilot
python -m unstructured.benchmark.retrieve_generate                # all 318
python -m unstructured.benchmark.retrieve_generate --report-only  # no model calls

python -m unstructured.judge --only chunks                        # tier 3
```

It resumes: answers are appended to `results/answers.jsonl` and searches cached
in `results/retrieval.json`, so an interrupted run never pays for the same query
twice. Generation and judging are paced client-side against their own per-minute
budgets (`GEN_RPM`, `JUDGE_RPM` in `pipeline.py`), and every row records the
judge that scored it, so swapping judges re-judges existing answers instead of
regenerating them.

`NVIDIA_API_KEY` is read from the repo-root `.env`.

## Results

`--report-only` regenerates `report.json` from the committed answers, which
also checks the code and the corpus still line up.

| File | Written by | Contents |
| --- | --- | --- |
| `report.json` | `retrieve_generate.py` | Headline retrieval and answer-quality numbers |
| `answers.jsonl` | `retrieve_generate.py`, `judge.py` | One scored record per query, with its chunks |
| `retrieval.json` | `retrieve_generate.py` | Cached chunks and page ranking per query |
| `ingest.log` | `ingest.py` | Per-PDF ingest output of the shipped table |

The attribute pipeline writes the same three files to `unstructured/results/`.

Over 318 English queries with `nemotron-3-super-120b-a12b` generating:

| Metric | Value |
| --- | --- |
| nDCG@10 | 0.605 |
| recall@10 | 0.649 |
| answer-in-context rate | 96.5% |
| mean token F1 | 0.399 |
| mean judge score | pending — see note |

The judge score is pending a re-run. The 0.585 (301 judged) in earlier reports
came from `z-ai/glm-5.3`, whose quota is now spent; the judge has moved to
`nemotron-3.5-lightning-30b-a3b`, which re-scores every saved answer so the two
pipelines are never compared across different judges.

## Differences from the original project

- `config.DATA_DIR` is the repo's `unstructured/data/vidore_v3_hr/pdfs`, and
  the `.env` is the repo-root one.
- Queries come from `queries_en.csv` and `qrels/qrels_en.csv` instead of the
  queries and corpus parquet. The parquet files are here too, but the csv
  carries the same rows with the pages already resolved, so nothing needs
  pyarrow. Page keys and metrics are unchanged.
- `data/vidore_v3_hr/corpus/` is not copied: it is 422 MB of page images and
  the harness only ever used it to map a `corpus_id` to a page, which
  `qrels_en.csv` already does.
- `lancedb/` (21 MB, 1785 chunks over 14 PDFs) is present but gitignored;
  `python unstructured/benchmark/ingest.py` rebuilds it from the PDFs.
