"""Sample the golden evaluation set (200 messages) from the held-out time window.

120 uniform-random messages (natural traffic distribution) + 80 targeted at rare,
high-risk situations that a uniform sample of 200 would barely contain. Every row
keeps its `stratum` so metrics can be reported on the uniform slice alone.
"""
import sys

import pandas as pd

from config import GOLDEN_CANDIDATES, GOLDEN_LABELS, PAIRS, SEED, SPLIT_DATE

TARGETED = {  # stratum -> (regex over customer text, n)
    "safety": (r"smok|fire|burn|melt|swell|explod|spark|electr?ocut|shock", 15),
    "money_account": (r"charged|refund|hack|stolen|locked out|apple ?id|password|subscri|billing|payment|unauthori", 20),
    "data_loss": (r"lost (?:all )?(?:of )?(?:my )?(?:photos|pictures|pics|contacts|data|messages|texts|music|notes|everything)|deleted|wiped|erased", 15),
    "frustrated_repeat": (r"already tried|tried everything|still (?:not|doesn|isn|hav)|\b(?:\d|three|four|five)(?:rd|th)? times\b|for (?:days|weeks)|genius bar|lawsuit|class action|\bsue\b|lawyer", 20),
}
N_UNIFORM = 120
N_NON_ENGLISH = 10   # targeted via the brand's own English-only redirect reply


def load_test_pool() -> pd.DataFrame:
    d = pd.read_parquet(PAIRS)
    d["ts"] = pd.to_datetime(d.created_at, format="%a %b %d %H:%M:%S %z %Y")
    d = d[d.ts >= pd.Timestamp(SPLIT_DATE, tz="UTC")]
    # one example per near-identical text (the 'i' bug produces thousands of clones)
    d["norm"] = d.customer_text.str.lower().str.replace(r"[^a-z ]", "", regex=True).str.strip()
    return d.drop_duplicates("norm")


def main():
    if GOLDEN_LABELS.exists() and "--force" not in sys.argv:
        sys.exit(f"{GOLDEN_LABELS} exists; resampling would orphan the labels. Use --force to override.")
    pool = load_test_pool()
    print(f"held-out pool after dedup: {len(pool):,} messages from {SPLIT_DATE} on")
    picked = [pool.sample(N_UNIFORM, random_state=SEED).assign(stratum="uniform")]
    taken = set(picked[0].tweet_id)

    lower = pool.customer_text.str.lower()
    for name, (pat, n) in TARGETED.items():
        cand = pool[lower.str.contains(pat, regex=True) & ~pool.tweet_id.isin(taken)]
        s = cand.sample(min(n, len(cand)), random_state=SEED).assign(stratum=name)
        taken |= set(s.tweet_id)
        picked.append(s)
    ne = pool[pool.reply_text.str.contains("in English", regex=False) & ~pool.tweet_id.isin(taken)]
    picked.append(ne.sample(N_NON_ENGLISH, random_state=SEED).assign(stratum="non_english"))

    gold = pd.concat(picked).sample(frac=1, random_state=SEED)  # shuffle so labeller can't infer stratum
    cols = ["tweet_id", "stratum", "created_at", "customer_text", "reply_text", "n_brand_turns", "n_customer_turns"]
    GOLDEN_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    gold[cols].to_csv(GOLDEN_CANDIDATES, index=False)
    print(gold.stratum.value_counts().to_string())
    print(f"-> {GOLDEN_CANDIDATES}")


if __name__ == "__main__":
    main()
