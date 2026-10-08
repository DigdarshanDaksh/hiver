"""Run every system over the golden set; cache predictions to outputs/predictions/<system>.jsonl.

  python run_systems.py                 # run anything not cached yet
  python run_systems.py --force agent   # re-run one system
"""
import argparse
import json

import pandas as pd

import baselines
from agent import run_agent
from config import GOLDEN_CANDIDATES, PRED_DIR
from llm import pmap

SYSTEMS = {
    "trivial": (baselines.trivial, False),
    "simple": (baselines.simple, False),
    "agent_no_retrieval": (lambda t: run_agent(t, use_retrieval=False), True),
    "agent": (run_agent, True),
}


def run(name: str, gold: pd.DataFrame, force: bool):
    path = PRED_DIR / f"{name}.jsonl"
    if path.exists() and not force:
        print(f"[cached] {path.name}")
        return
    fn, uses_llm = SYSTEMS[name]
    rows = list(gold[["tweet_id", "customer_text"]].itertuples(index=False))

    def one(r):
        try:
            return {"tweet_id": int(r.tweet_id), **fn(r.customer_text)}
        except Exception as e:  # keep going; a failed call is scored as a failure
            return {"tweet_id": int(r.tweet_id), "error": repr(e)}

    preds = pmap(one, rows, workers=8 if uses_llm else 1, desc=name)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{name}: {sum('error' in p for p in preds)} errors -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="*", default=list(SYSTEMS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    gold = pd.read_csv(GOLDEN_CANDIDATES)
    for s in a.systems:
        run(s, gold, a.force)
