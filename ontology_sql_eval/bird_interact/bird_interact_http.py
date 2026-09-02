"""HTTP clients for Bird ADK :6001 (user simulator) and :6002 (DB environment)."""

import httpx

USER_SIM_URL = "http://localhost:6001"
DB_ENV_URL = "http://localhost:6002"
_TIMEOUT = 120.0


async def ask_user(task_id: str, question: str) -> str:
    """POST :6001/ask → returns answer string."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.post(
            f"{USER_SIM_URL}/ask",
            json={"task_id": task_id, "question": question},
        )
        r.raise_for_status()
        data = r.json()
        if "answer" not in data:
            raise ValueError(f"Missing 'answer' key in response: {data}")
        return data["answer"]


async def submit_sql(task_id: str, sql: str) -> dict:
    """POST :6002/submit → returns full response dict."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.post(
            f"{DB_ENV_URL}/submit",
            json={"task_id": task_id, "sql": sql},
        )
        r.raise_for_status()
        return r.json()


async def phase_transition(task_id: str) -> None:
    """POST :6001/phase_transition."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{USER_SIM_URL}/phase_transition",
            json={"task_id": task_id},
        )
        r.raise_for_status()
