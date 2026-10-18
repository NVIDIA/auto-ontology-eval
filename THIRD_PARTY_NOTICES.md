# Third-Party Notices

This project uses the third-party packages listed below. They are consumed as
standard Python package dependencies (dynamic linking via the Python
interpreter); no third-party Python *package* source is copied or vendored into
this repository.

Separately, this repository includes third-party content, documented in
[Included Third-Party Content](#included-third-party-content) below.

| Package | Version (minimum) | License |
|---|---|---|
| langchain-nvidia-ai-endpoints | latest | MIT |
| langchain-openai | ≥1.3.0, <1.3.5 | MIT |
| langchain-postgres | ≥0.0.17 | MIT |
| pandas | ≥2.0, <3 | BSD-3-Clause |
| pydantic | ≥2.0 | MIT |
| python-dotenv | latest | BSD-3-Clause |
| psycopg[binary] | ≥3.3.3 | LGPL-3.0-only |
| psycopg-pool | ≥3.3.1 | LGPL-3.0-only |
| snowflake-connector-python | ≥4.6.0 | Apache-2.0 |
| databricks-sql-connector | ≥4.3.0 | Apache-2.0 |
| duckdb | ≥1.5.2 | MIT |
| pyheavydb | ≥8.0.1.post1 | Apache-2.0 |
| hvac | ≥2.4.0 | Apache-2.0 |
| httpx | ≥0.28.1 | BSD-3-Clause |
| fastapi | ≥0.115.0 | MIT |
| uvicorn[standard] | ≥0.32.0 | BSD-3-Clause |
| importlib-metadata | ≥9.0.0 | Apache-2.0 |
| numpy | ≥2.0 | BSD-3-Clause |

**Development dependencies** (not distributed):

| Package | Version | License |
|---|---|---|
| ruff | 0.15.9 | MIT |

**Source dependencies:**

- `auto_ontology` — sourced from
  [NVIDIA/auto-ontology](https://github.com/NVIDIA/auto-ontology) as a pinned
  Git dependency (Apache-2.0)

---

## Included Third-Party Content

The following third-party material is included in this repository, in addition
to the package dependencies listed above.

### 1. WideWorldImporters sample database

| | |
|---|---|
| **Location in this repo** | `data/datasets/wideworldimporters/` (`ddl/*.sql`, `data/*.csv`) |
| **Upstream project** | SQL Server Samples — https://github.com/microsoft/sql-server-samples |
| **Copyright holder** | Copyright (c) Microsoft Corporation |
| **License** | MIT |

A Postgres port of Microsoft's WideWorldImporters sample OLTP database,
included as a public worked example. The upstream notice is reproduced in
`data/datasets/wideworldimporters/LICENSE`; the DDL files carry the Microsoft
copyright and MIT identifier in their headers.

### 2. BEAVER benchmark evaluation logic

| | |
|---|---|
| **Location in this repo** | `tests/beaver_reference.py` |
| **Upstream project** | BEAVER — https://github.com/beaverbench/beaver |
| **Copyright holder** | Copyright (c) 2024 MIT Data Systems Group |
| **License** | MIT |

`beaver_reference.py` ports the execution and comparison logic of BEAVER's
official execution-accuracy (`ex_acc`) evaluator, from `eval/utils/ex_acc.py`.
Only the test suite uses it, as the reference the benchmark's own
implementation is checked against. The file carries the upstream copyright and
license identifier in its header, followed by the NVIDIA copyright block covering
NVIDIA's modifications.

### Note on the BEAVER dataset

The BEAVER **dataset** is not included in this repository.
`scripts/seed/seed_beaverbench.py` downloads the questions and MySQL dumps at runtime
from the gated HuggingFace datasets
[`beaverbench/beaver-query`](https://huggingface.co/datasets/beaverbench/beaver-query)
and [`beaverbench/beaver-table`](https://huggingface.co/datasets/beaverbench/beaver-table),
and `data/datasets/beaverbench/` ships only a README and a `.gitkeep`. Its terms are
set by its authors on those dataset pages.

### Note on the BIRD dataset

The BIRD Mini-Dev **dataset** is not included in this repository.
`scripts/seed/seed_bird.py` downloads it at runtime from the official distribution,
and `data/datasets/bird/` ships only a README and a `.gitkeep`. The BIRD dataset is
distributed by its authors under CC BY-SA 4.0.

### Note on FDABench-Lite and its source databases

FDABench-Lite tasks and the SQLite databases they require are **not** included
in this repository. `scripts/seed/seed_fdabench.py` downloads them at runtime, and
`data/datasets/fdabench/` ships only a README and a `.gitkeep`.

| Runtime download | Upstream | License (as published by authors) |
|---|---|---|
| FDABench-Lite JSONLs | [FDAbench2026/Fdabench-Lite](https://huggingface.co/datasets/FDAbench2026/Fdabench-Lite) / [fdabench/FDAbench](https://github.com/fdabench/FDAbench) | MIT |
| BIRD train SQLite DBs | [BIRD](https://bird-bench.github.io/) `train.zip` | CC BY-SA 4.0 |
| Spider2-lite local SQLite pack | [xlang-ai/Spider2](https://github.com/xlang-ai/Spider2) Google Drive local pack | see upstream Spider2 |
| Spider1 databases | [Yale Spider](https://yale-lily.github.io/spider) dataset zip | CC BY-SA 4.0 |

Dabstep (`merchant_data.db`) tasks are intentionally skipped by the seed
script because that database is not redistributed with FDABench-Lite.

---

## License Texts

### MIT License

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Applies to: langchain-nvidia-ai-endpoints, langchain-openai, langchain-postgres,
pydantic, duckdb, fastapi, ruff.

Also applies to the included third-party content above, under their respective
copyrights:

- SQL Server Samples (`data/datasets/wideworldimporters/`) — Copyright (c) Microsoft Corporation
- BEAVER (`tests/beaver_reference.py`) — Copyright (c) 2024 MIT Data Systems Group

---

### BSD 3-Clause License

> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
> 3. Neither the name of the copyright holder nor the names of its contributors
>    may be used to endorse or promote products derived from this software
>    without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Applies to: pandas, python-dotenv, httpx, uvicorn, numpy.

---

### Apache License 2.0

Full text: https://www.apache.org/licenses/LICENSE-2.0

Applies to: snowflake-connector-python, databricks-sql-connector, pyheavydb,
hvac, importlib-metadata.

---

### GNU Lesser General Public License v3.0 (LGPL-3.0)

Full text: https://www.gnu.org/licenses/lgpl-3.0.html

Applies to: psycopg, psycopg-pool.

These packages are used via dynamic linking (standard Python import). No LGPL
source is copied or modified.
