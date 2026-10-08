"""Compute all metrics from cached predictions, judge scores and human labels.
Writes outputs/results.md and outputs/agent_errors.csv (input to failure analysis)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from config import (GOLDEN_CANDIDATES, GOLDEN_LABELS, HUMAN_RATINGS, INTENT_LIST, JUDGE_DIR,
                    OUT, PRED_DIR, SEED)

SYSTEMS = ["trivial", "simple", "agent_no_retrieval", "agent"]
rng = np.random.default_rng(SEED)


def boot_ci(df, fn, n=1000):
    """95% bootstrap CI of metric fn over rows of df."""
    vals = [fn(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(n)]
    return np.nanpercentile(vals, [2.5, 97.5])


def fmt(x, ci=None):
    return f"{x:.2f}" if ci is None else f"{x:.2f} [{ci[0]:.2f}-{ci[1]:.2f}]"


def load():
    gold = pd.read_csv(GOLDEN_CANDIDATES).merge(pd.read_csv(GOLDEN_LABELS), on="tweet_id")
    gold["escalate"] = gold.escalate.astype(str).str.lower().isin(["true", "1", "y", "yes"])
    sims = {}
    for s in SYSTEMS:
        p = PRED_DIR / f"{s}.jsonl"
        if not p.exists():
            continue
        pr = pd.read_json(p, lines=True)
        # a failed LLM call counts as: wrong intent, escalated, no reply
        for c, default in [("intent", "ERROR"), ("escalate", True), ("reply", ""), ("escalation_reason", "none")]:
            pr[c] = pr[c].fillna(default) if c in pr else default
        pr = pr.add_prefix("p_").rename(columns={"p_tweet_id": "tweet_id"})
        j = JUDGE_DIR / f"{s}.jsonl"
        if j.exists():
            jd = pd.read_json(j, lines=True).add_prefix("j_").rename(columns={"j_tweet_id": "tweet_id"})
            pr = pr.merge(jd, on="tweet_id", how="left")
        sims[s] = gold.merge(pr, on="tweet_id")
    return gold, sims


def metrics(d: pd.DataFrame, sim_vec) -> dict:
    m = {"n": len(d)}
    m["intent_acc"] = (d.intent == d.p_intent).mean()
    mf1 = lambda x: f1_score(x.intent, x.p_intent, labels=INTENT_LIST, average="macro", zero_division=0)
    m["intent_macroF1"] = (mf1(d), boot_ci(d, mf1))
    y, p = d.escalate, d.p_escalate.astype(bool)
    rec = lambda x: (x.p_escalate.astype(bool) & x.escalate).sum() / max(x.escalate.sum(), 1)
    m["esc_recall"] = (rec(d), boot_ci(d, rec))
    m["esc_precision"] = (p & y).sum() / max(p.sum(), 1)
    m["auto_rate"] = (~p).mean()
    m["missed_escalations"] = int((~p & y).sum())
    r = d.p_reply.fillna("").astype(str)
    m["reply_len_ok"] = r.str.len().between(1, 280).mean()
    m["invented_url"] = r.str.contains(r"https?://|www\.", regex=True).mean()
    a, b = sim_vec.transform(r), sim_vec.transform(d.reply_text)
    m["sim_to_real_reply"] = float(np.asarray(a.multiply(b).sum(1)).mean())
    if "j_overall" in d:
        m["judge_overall"] = (d.j_overall.mean(), boot_ci(d, lambda x: x.j_overall.mean()))
        m["judge_safe"] = d.j_safe.mean()
        good = (d.j_overall >= 4) & (d.j_safe == 1)
        # the number that matters: share of traffic we auto-send AND that is fine to send
        m["auto_and_good"] = (~p & good).mean()
        m["auto_but_bad"] = (~p & ~good).mean()
    return m


def table(rows: dict) -> str:
    keys = list(dict.fromkeys(k for m in rows.values() for k in m))
    out = ["| metric | " + " | ".join(rows) + " |", "|---" * (len(rows) + 1) + "|"]
    for k in keys:
        cells = []
        for m in rows.values():
            v = m.get(k)
            if v is None:
                cells.append("")
            elif isinstance(v, tuple):
                cells.append(fmt(*v))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(v))
            else:
                cells.append(fmt(v))
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def judge_agreement() -> str:
    if not HUMAN_RATINGS.exists():
        return "_No human ratings yet: run `python label.py replies`._"
    h = pd.read_csv(HUMAN_RATINGS)
    js = [pd.read_json(JUDGE_DIR / f"{s}.jsonl", lines=True).assign(system=s)
          for s in h.system.unique() if (JUDGE_DIR / f"{s}.jsonl").exists()]
    m = h.merge(pd.concat(js), on=["tweet_id", "system"], suffixes=("_h", "_j"))
    lines = [f"n = {len(m)} (tweet, reply) pairs rated blind by a human and by the judge\n",
             "| criterion | kappa (quadratic; plain for safe) | spearman | exact | within 1 | human mean | judge mean |",
             "|---|---|---|---|---|---|---|"]
    for c in ["relevance", "actionable", "voice", "safe", "overall"]:
        hh, jj = m[f"{c}_h"], m[f"{c}_j"]
        k = cohen_kappa_score(hh, jj, weights=None if c == "safe" else "quadratic")
        lines.append(f"| {c} | {k:.2f} | {spearmanr(hh, jj).statistic:.2f} | {(hh == jj).mean():.2f} | "
                     f"{((hh - jj).abs() <= 1).mean():.2f} | {hh.mean():.2f} | {jj.mean():.2f} |")
    by = m.groupby("system")[["overall_h", "overall_j"]].mean().round(2)
    lines += ["", "Per-system mean overall (does the judge rank systems the way the human does?):", "", by.to_markdown()]
    return "\n".join(lines)


def main():
    gold, sims = load()
    allreplies = pd.concat([gold.reply_text] + [d.p_reply.fillna("").astype(str) for d in sims.values()])
    vec = TfidfVectorizer(sublinear_tf=True).fit(allreplies)
    uni = gold[gold.stratum == "uniform"]
    md = [f"# Results ({len(gold)} labelled golden examples)\n",
          f"Label distribution:\n\n{gold.intent.value_counts().to_markdown()}\n",
          f"Gold escalation rate: all {gold.escalate.mean():.2f} | uniform slice {uni.escalate.mean():.2f}\n"]
    for name, mask in [("All examples (stratified: oversamples risky cases)", None),
                       ("Uniform slice only (natural traffic)", "uniform")]:
        rows = {s: metrics(d if mask is None else d[d.stratum == mask], vec) for s, d in sims.items()}
        md += [f"## {name}\n", table(rows), ""]
    hist = JUDGE_DIR / "historical.jsonl"
    if hist.exists():
        h = pd.read_json(hist, lines=True)
        md.append(f"Judge on AppleSupport's REAL historical replies: overall {h.overall.mean():.2f}, "
                  f"safe {h.safe.mean():.2f}, overall>=4 {(h.overall >= 4).mean():.2f}\n")
    if "agent" in sims:
        d = sims["agent"]
        f1s = f1_score(d.intent, d.p_intent, labels=INTENT_LIST, average=None, zero_division=0)
        md += ["## Agent: per-intent F1\n",
               pd.DataFrame({"intent": INTENT_LIST, "support": [(d.intent == i).sum() for i in INTENT_LIST],
                             "f1": f1s.round(2)}).to_markdown(index=False)]
        cm = pd.DataFrame(confusion_matrix(d.intent, d.p_intent, labels=INTENT_LIST),
                          index=[f"T:{i[:14]}" for i in INTENT_LIST], columns=[i[:6] for i in INTENT_LIST])
        md += ["\n## Agent: confusion matrix (rows = human label)\n", cm.to_markdown(), ""]
        pe = d.p_escalate.astype(bool)
        by = d.assign(pe=pe).groupby("stratum").agg(
            n=("tweet_id", "size"), gold_escalate=("escalate", "sum"),
            caught=("escalate", lambda s: int((s & pe[s.index]).sum())),
            false_alarm=("escalate", lambda s: int((~s & pe[s.index]).sum())))
        md += ["## Agent: escalation by sampling stratum\n", by.to_markdown(), ""]
        bad = (d.intent != d.p_intent) | (d.escalate != pe)
        if "j_overall" in d:
            bad |= d.j_overall <= 2
        cols = [c for c in ["tweet_id", "stratum", "customer_text", "intent", "p_intent", "escalate", "p_escalate",
                            "escalation_reason", "p_escalation_reason", "p_reason", "p_reply", "j_overall",
                            "j_rationale", "reply_text", "note"] if c in d]
        d[bad][cols].to_csv(OUT / "agent_errors.csv", index=False)
        md.append(f"{int(bad.sum())} agent error rows written to outputs/agent_errors.csv\n")
    md += ["## LLM judge vs human agreement\n", judge_agreement()]
    (OUT / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
