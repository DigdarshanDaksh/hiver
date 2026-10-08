"""Thin OpenAI wrapper: JSON-schema structured output, retries, parallel map."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from tqdm import tqdm

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY
    return _client


def chat_json(model: str, system: str, user: str, schema: dict, name: str, retries: int = 4) -> dict:
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": name, "schema": schema, "strict": True}},
    )
    # reasoning models reject temperature
    if not model.startswith(("o1", "o3", "o4", "gpt-5")):
        kwargs["temperature"] = 0
    for attempt in range(retries):
        try:
            resp = client().chat.completions.create(**kwargs)
            return json.loads(resp.choices[0].message.content)
        except Exception:  # rate limits / transient errors
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt + 1)


def pmap(fn, items, workers: int = 8, desc: str = ""):
    with ThreadPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(fn, items), total=len(items), desc=desc))
