"""Terminal labelling tool. Progress is saved after every item; re-run to resume.

  python label.py intents            # label intent + escalation for the golden set
  python label.py replies --n 20     # blind-rate replies (n tweets x 3 systems) for judge agreement

AppleSupport's real reply, the sampling stratum and the system name are hidden so they
can't anchor the label. Alternatively, fill the CSVs in a spreadsheet: same columns.
"""
import argparse
import os

import pandas as pd

from config import (ESCALATION_REASONS, GOLDEN_CANDIDATES, GOLDEN_LABELS, HUMAN_RATINGS,
                    INTENTS, PRED_DIR, REASON_LIST, SEED)

INTENT_KEYS = list(INTENTS)


def ask(prompt: str, valid=None, allow_empty=False) -> str:
    while True:
        v = input(prompt).strip().lower()
        if v == "q":
            raise KeyboardInterrupt
        if (allow_empty and v == "") or valid is None or v in valid:
            return v
        print(f"  expected one of {sorted(valid)}")


def append(path, row: dict):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def label_intents():
    gold = pd.read_csv(GOLDEN_CANDIDATES)
    done = set(pd.read_csv(GOLDEN_LABELS).tweet_id) if GOLDEN_LABELS.exists() else set()
    todo = gold[~gold.tweet_id.isin(done)]
    menu = "\n".join(f"  {i+1}. {k:<24} {INTENTS[k][:90]}" for i, k in enumerate(INTENT_KEYS))
    rmenu = "\n".join(f"  {i+1}. {k:<26} {ESCALATION_REASONS[k][:80]}" for i, k in enumerate(REASON_LIST))
    for n, r in enumerate(todo.itertuples(), len(done) + 1):
        os.system("cls" if os.name == "nt" else "clear")
        print(f"[{n}/{len(gold)}]  (q = quit, progress is saved)\n\n  {r.customer_text}\n\n{menu}\n")
        intent = INTENT_KEYS[int(ask("intent #: ", {str(i + 1) for i in range(len(INTENT_KEYS))})) - 1]
        esc = ask("escalate to human? [y/n]: ", {"y", "n"}) == "y"
        reason = "none"
        if esc:
            print(rmenu)
            reason = REASON_LIST[int(ask("reason #: ", {str(i + 1) for i in range(len(REASON_LIST))})) - 1]
        note = input("note (optional): ").strip()
        append(GOLDEN_LABELS, {"tweet_id": r.tweet_id, "intent": intent, "escalate": esc,
                               "escalation_reason": reason, "note": note})


RATE_SYSTEMS = ["historical", "simple", "agent"]


def rating_items(n: int) -> pd.DataFrame:
    gold = pd.read_csv(GOLDEN_CANDIDATES)
    ids = gold.sample(n, random_state=SEED).tweet_id
    items = []
    for s in RATE_SYSTEMS:
        if s == "historical":
            rep = gold.set_index("tweet_id").reply_text
        else:
            rep = pd.read_json(PRED_DIR / f"{s}.jsonl", lines=True).set_index("tweet_id").reply
        items += [{"tweet_id": t, "system": s, "reply": rep[t]} for t in ids]
    df = pd.DataFrame(items).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df.merge(gold[["tweet_id", "customer_text"]], on="tweet_id", how="left")


def label_replies(n: int):
    items = rating_items(n)
    done = (set(zip(*pd.read_csv(HUMAN_RATINGS)[["tweet_id", "system"]].T.values))
            if HUMAN_RATINGS.exists() else set())
    one_to_five = {"1", "2", "3", "4", "5"}
    for i, r in enumerate(items.itertuples(), 1):
        if (r.tweet_id, r.system) in done:
            continue
        os.system("cls" if os.name == "nt" else "clear")
        print(f"[{i}/{len(items)}]  (q = quit)   rubric: see judge.py docstring\n")
        print(f"CUSTOMER:  {r.customer_text}\n\nREPLY:     {r.reply}\n")
        row = {"tweet_id": r.tweet_id, "system": r.system}
        for c in ["relevance", "actionable", "voice"]:
            row[c] = int(ask(f"{c} 1-5: ", one_to_five))
        row["safe"] = int(ask("safe 0/1: ", {"0", "1"}))
        row["overall"] = int(ask("overall 1-5: ", one_to_five))
        append(HUMAN_RATINGS, row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["intents", "replies"])
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    try:
        label_intents() if a.mode == "intents" else label_replies(a.n)
        print("done.")
    except KeyboardInterrupt:
        print("\nsaved, bye.")
