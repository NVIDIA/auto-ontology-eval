# Third-Party Notices

This project uses the following third-party packages. All are consumed as
standard Python package dependencies (dynamic linking via the Python
interpreter); no third-party source is copied or vendored into this repository.

| Package | Version (minimum) | License |
|---|---|---|
| langchain-nvidia-ai-endpoints | latest | MIT |
| langchain-openai | ≥1.3.2 | MIT |
| langchain-postgres | ≥0.0.17 | MIT |
| pandas | ≥2.0, <3 | BSD-3-Clause |
| pydantic | ≥2.0 | MIT |
| python-dotenv | latest | BSD-3-Clause |
| neo4j | ≥6.1.0 | Apache-2.0 AND Python-2.0 |
| psycopg[binary] | ≥3.3.3 | LGPL-3.0-only |
| psycopg-pool | ≥3.3.1 | LGPL-3.0-only |
| snowflake-connector-python | ≥4.6.0 | Apache-2.0 |
| databricks-sql-connector | ≥4.3.0 | Apache-2.0 |
| kumoai | ≥2.22.0 | MIT |
| duckdb | ≥1.5.2 | MIT |
| pyheavydb | ≥8.0.1.post1 | Apache-2.0 |
| hvac | ≥2.4.0 | Apache-2.0 |
| httpx | ≥0.28.1 | BSD-3-Clause |
| fastapi | ≥0.115.0 | MIT |
| uvicorn[standard] | ≥0.32.0 | BSD-3-Clause |
| importlib-metadata | ≥9.0.0 | Apache-2.0 |
| func-timeout | ≥4.3.5 | LGPLv2 |
| numpy | ≥2.0 | BSD-3-Clause |

**Development dependencies** (not distributed):

| Package | Version | License |
|---|---|---|
| ruff | 0.15.9 | MIT |

**NVIDIA internal dependencies** (not third-party):

- `nemo-retriever` — sourced from [NVIDIA/NeMo-Retriever](https://github.com/NVIDIA/NeMo-Retriever)
- `gsf` / `dev_tools` — sourced from [NVIDIA/GSF](https://github.com/NVIDIA/GSF) via `PYTHONPATH`

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
pydantic, kumoai, duckdb, fastapi, ruff.

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

Applies to: neo4j, snowflake-connector-python, databricks-sql-connector,
pyheavydb, hvac, importlib-metadata.

---

### GNU Lesser General Public License v3.0 (LGPL-3.0)

Full text: https://www.gnu.org/licenses/lgpl-3.0.html

Applies to: psycopg, psycopg-pool.

These packages are used via dynamic linking (standard Python import). No LGPL
source is copied or modified.

---

### GNU Lesser General Public License v2 (LGPLv2)

Full text: https://www.gnu.org/licenses/old-licenses/lgpl-2.0.html

Applies to: func-timeout.

This package is used via dynamic linking (standard Python import). No LGPLv2
source is copied or modified.

---

### Python Software Foundation License 2.0

Full text: https://docs.python.org/3/license.html

Applies to: neo4j (combined with Apache-2.0 above).
