# csv-sql-judge

csv-sql-judge is a lightweight, automated evaluation utility designed to re-score Text-to-SQL test results. By leveraging LLMs with structured outputs, it analyzes generated SQL queries against user questions and ground truth data. It evaluates queries for logic matching, semantic appropriateness, and structural similarity, appending detailed grading metrics and error descriptions directly back into your evaluation CSV.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+.

```bash
uv sync
cp .env.example .env   # then fill in NVIDIA_API_KEY
```

Configuration is read from `.env` (or the environment):

| Variable         | Description                    | Default                                 |
| ---------------- | ------------------------------ | --------------------------------------- |
| `MODEL_NAME`     | NVIDIA scoring model.          | `nvidia/nvidia/Nemotron-3-Nano-30B-A3B` |
| `BASE_URL`       | API base URL.                  | `https://inference-api.nvidia.com/v1`   |
| `NVIDIA_API_KEY` | API key for the scoring model. | _(empty)_                               |

## Usage

Drop one or more CSV files into the `input/` folder. Each file must contain
`question`, `expected_sql`, and `returned_sql` columns (and optionally
`returned_answer`). The tool scores every CSV in `input/` and writes the
results to `output/<name>_scores.csv` with the LLM score columns appended.

```bash
# Via the console script
uv run csv-sql-judge

# Or directly
uv run python main.py
```

Options:

- `--input-dir` — folder to scan for input CSVs (default: `input`)
- `--output-dir` — folder to write scored CSVs (default: `output`)
- `--workers` — number of concurrent scoring workers (default: `1`)

### Output columns

`llm_logic_match`, `llm_semantic_match`, `llm_final_weighted_score`,
`llm_sql_vs_ground_truth`, `llm_is_valid_sql`, `llm_is_sql_returns_data`,
and `llm_logic_issues`.
