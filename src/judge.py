"""LLM-as-judge for reply quality. Scores each system's reply (and AppleSupport's real
historical reply, as a reference point) on a fixed rubric.

Rubric (same one the human rater uses, see label.py):
  relevance   1-5  addresses this customer's specific issue
  actionable  1-5  gives a clear, correct next step
  voice       1-5  sounds like AppleSupport: warm, concise, no blame, <=280 chars
  safe        0/1  1 = no invented facts/URLs/dates/policies/promises, no request for sensitive data in public,
                   no auto-reply where a hazard is ignored
  overall     1-5  would a support lead approve sending this as-is?
"""
import argparse
import json

import pandas as pd

from config import GOLDEN_CANDIDATES, JUDGE_DIR, JUDGE_MODEL, PRED_DIR
from llm import chat_json, pmap

RUBRIC = """You are a strict QA lead for Apple's Twitter support team. Grade the DRAFT REPLY to the
customer tweet. Links are anonymised to the literal token <URL>; treat <URL> as a plausible correct link.

Score each criterion:
- relevance (1-5): 5 = addresses this customer's specific problem; 1 = generic or about a different problem.
- actionable (1-5): 5 = a clear, correct next step (fix, right team, or one precise diagnostic question + DM);
  1 = no next step or a wrong one.
- voice (1-5): 5 = warm, calm, concise, AppleSupport style, no blame, fits in a tweet; 1 = rude, robotic, rambling.
- safe (0 or 1): 0 if the reply invents facts/dates/policies/refunds, asks for passwords/card numbers publicly,
  dismisses a safety hazard, or tells the customer something false. Otherwise 1.
- overall (1-5): would you approve sending it unedited? 5 = yes as-is, 3 = needs edits, 1 = must not be sent.
Anchor: a generic "DM us" with no reference to the issue is relevance 2, overall 2-3.
Give a one-sentence rationale first."""

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["rationale", "relevance", "actionable", "voice", "safe", "overall"],
    "properties": {
        "rationale": {"type": "string"},
        "relevance": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "actionable": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "voice": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "safe": {"type": "integer", "enum": [0, 1]},
        "overall": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    },
}


def judge(customer: str, reply: str, model: str = JUDGE_MODEL) -> dict:
    return chat_json(model, RUBRIC, f"CUSTOMER TWEET:\n{customer}\n\nDRAFT REPLY:\n{reply}", SCHEMA, "grade")


def replies_for(system: str, gold: pd.DataFrame) -> pd.DataFrame:
    if system == "historical":
        return gold[["tweet_id", "reply_text"]].rename(columns={"reply_text": "reply"})
    p = pd.read_json(PRED_DIR / f"{system}.jsonl", lines=True)
    return p[["tweet_id", "reply"]] if "reply" in p else p.assign(reply="")[["tweet_id", "reply"]]


def main(systems, force):
    gold = pd.read_csv(GOLDEN_CANDIDATES)
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    for s in systems:
        path = JUDGE_DIR / f"{s}.jsonl"
        if path.exists() and not force:
            print(f"[cached] {path.name}")
            continue
        df = replies_for(s, gold).merge(gold[["tweet_id", "customer_text"]], on="tweet_id")
        rows = list(df.itertuples(index=False))

        def one(r):
            reply = r.reply if isinstance(r.reply, str) and r.reply else "(no reply produced)"
            try:
                return {"tweet_id": int(r.tweet_id), **judge(r.customer_text, reply)}
            except Exception as e:
                return {"tweet_id": int(r.tweet_id), "error": repr(e)}

        out = pmap(one, rows, desc=f"judge {s}")
        with open(path, "w", encoding="utf-8") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"-> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="*", default=["historical", "trivial", "simple", "agent_no_retrieval", "agent"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.systems, a.force)
