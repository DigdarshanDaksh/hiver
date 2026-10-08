"""Extract AppleSupport (customer opener -> brand first reply) pairs from twcs.csv.

Output: data/processed/apple_pairs.parquet with one row per thread opener that
AppleSupport answered. Columns:
  tweet_id, customer_text, reply_text, reply_id, created_at,
  n_brand_turns, n_customer_turns  (thread depth after the opener)
"""
import argparse
import re
from pathlib import Path

import pandas as pd

BRAND = "AppleSupport"
ROOT = Path(__file__).resolve().parents[1]


def clean(text: str) -> str:
    text = re.sub(r"@\w+", "", text)              # handles (anonymised ids and brand)
    text = re.sub(r"https?://\S+", "<URL>", text)
    return re.sub(r"\s+", " ", text).strip()


def main(raw: Path, out: Path):
    cols = ["tweet_id", "author_id", "inbound", "created_at", "text",
            "response_tweet_id", "in_response_to_tweet_id"]
    df = pd.read_csv(raw, usecols=cols, dtype={"tweet_id": "int64", "author_id": str,
                                                "response_tweet_id": str})
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")

    brand = df[df.author_id == BRAND]
    # customer tweets answered by the brand
    parent_ids = brand.in_response_to_tweet_id.dropna().astype("int64")
    by_id = df.set_index("tweet_id")
    parents = by_id.loc[by_id.index.intersection(parent_ids)]
    # thread openers only: inbound and not itself a reply
    openers = parents[parents.inbound & parents.in_response_to_tweet_id.isna()]

    # brand's first reply to each opener
    first_reply = (brand[brand.in_response_to_tweet_id.isin(openers.index)]
                   .sort_values("tweet_id")
                   .drop_duplicates("in_response_to_tweet_id"))
    first_reply = first_reply.set_index("in_response_to_tweet_id")

    # thread depth: walk descendants of each opener (BFS over the reply graph)
    children = df.dropna(subset=["in_response_to_tweet_id"]).groupby(
        df.in_response_to_tweet_id.dropna().astype("int64"))["tweet_id"].apply(list).to_dict()
    is_brand = dict(zip(df.tweet_id, df.author_id == BRAND))
    inbound = dict(zip(df.tweet_id, df.inbound))

    rows = []
    for oid, reply in first_reply.iterrows():
        oid = int(oid)
        nb = nc = 0
        stack = list(children.get(oid, []))
        while stack:
            t = stack.pop()
            nb += is_brand.get(t, False)
            nc += bool(inbound.get(t, False))
            stack.extend(children.get(t, []))
        rows.append({
            "tweet_id": oid,
            "created_at": by_id.at[oid, "created_at"],
            "customer_raw": by_id.at[oid, "text"],
            "customer_text": clean(by_id.at[oid, "text"]),
            "reply_id": int(reply.tweet_id),
            "reply_text": clean(reply.text),
            "n_brand_turns": nb,
            "n_customer_turns": nc,
        })
    out_df = pd.DataFrame(rows)
    out_df = out_df[out_df.customer_text.str.len() >= 10]   # drop empty / mention-only openers
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"{len(out_df):,} AppleSupport opener/reply pairs -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw/twcs.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/apple_pairs.parquet")
    a = ap.parse_args()
    main(a.raw, a.out)
