"""TF-IDF nearest-neighbour search over historical AppleSupport threads (train window only)."""
from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from config import PAIRS, SPLIT_DATE


class Retriever:
    def __init__(self, max_docs: int | None = None):
        d = pd.read_parquet(PAIRS)
        ts = pd.to_datetime(d.created_at, format="%a %b %d %H:%M:%S %z %Y")
        d = d[ts < pd.Timestamp(SPLIT_DATE, tz="UTC")]          # no future leakage
        d = d.drop_duplicates("customer_text").reset_index(drop=True)
        if max_docs:
            d = d.sample(min(max_docs, len(d)), random_state=0).reset_index(drop=True)
        self.df = d
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
        self.X = self.vec.fit_transform(d.customer_text)

    def search(self, text: str, k: int = 5, diverse: bool = True) -> list[dict]:
        sims = linear_kernel(self.vec.transform([text]), self.X).ravel()
        order = sims.argsort()[::-1]
        out, seen_replies = [], set()
        for i in order:
            row = self.df.iloc[i]
            key = row.reply_text[:60]
            if diverse and key in seen_replies:     # avoid k copies of the same template
                continue
            seen_replies.add(key)
            out.append({"customer_text": row.customer_text, "reply_text": row.reply_text,
                        "score": float(sims[i])})
            if len(out) == k:
                break
        return out


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
