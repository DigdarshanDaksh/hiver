# AppleSupport AI agent: classify, draft, escalate, and prove it

An AI first-response agent for **@AppleSupport** built from the Kaggle *Customer Support on Twitter* dataset.
For each incoming customer tweet, it:

1. **Classifies** the tweet into one of 9 intents defined from the data (`src/config.py`).
2. **Drafts a reply** grounded in how AppleSupport answered the most similar past tweets (TF-IDF retrieval → few-shot).
3. **Decides auto-handle vs escalate**, with a stated reason. An LLM decision is combined with a deterministic keyword
   guard that can only *add* escalations.

The report is in [REPORT.md](REPORT.md) and the decision log in [DECISIONS.md](DECISIONS.md). Generated numbers are in `outputs/results.md`.

## Reproduce the results

> **Current status:** the golden set is labelled and the baselines are evaluated. The LLM agent and judge have not
> been run yet (no API key was available), so their results are not cached and `run_all.py` needs a key.
> See the status note at the top of [REPORT.md](REPORT.md).

```bash
python -m venv .venv && .venv/Scripts/activate       # (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python src/evaluate.py                               # baseline metrics from committed files, no API key needed
```

To run the full pipeline including the LLM agent and judge (about 1,400 calls, ~10 min with 8 threads):

```bash
cp .env.example .env    # add OPENAI_API_KEY
python src/run_all.py --fresh
```

The raw 493 MB `twcs.csv` is **not** needed. `data/processed/apple_pairs.parquet` (15 MB, committed) contains
the 74k AppleSupport opener→reply pairs. To rebuild it, put `twcs.csv` in `data/raw/` and run `python src/prepare_data.py` (~1 min).

## Pipeline

| step | file | what it does |
|---|---|---|
| 1 | `src/prepare_data.py` | twcs.csv → AppleSupport thread openers paired with the brand's first reply, plus thread depth |
| 2 | `src/sample_golden.py` | 200-message golden set from the held-out window (≥ 2017-11-22): 120 uniform + 80 targeted risky cases |
| 3 | `src/label.py intents` | terminal tool for hand-labelling intent + escalation (Apple's reply and stratum hidden) |
| 4 | `src/run_systems.py` | runs trivial, simple, agent-without-retrieval and agent → `outputs/predictions/` |
| 5 | `src/judge.py` | LLM-as-judge rubric for every system **and** AppleSupport's real replies → `outputs/judge/` |
| 6 | `src/label.py replies` | blind human rating of 60 replies on the same rubric (to check the judge) |
| 7 | `src/evaluate.py` | metrics, bootstrap CIs, uniform vs stratified slices, judge–human agreement, error dump |

Try the agent on one message: `python src/agent.py "my iphone gets really hot and dies by noon"`

## Systems compared

| system | intent | escalation | reply |
|---|---|---|---|
| `trivial` | most common intent | never escalate | AppleSupport's most frequent generic "DM us" reply |
| `simple` | keyword rules | keyword rules | nearest-neighbour historical reply, copied verbatim |
| `agent_no_retrieval` | LLM | LLM + keyword guard | LLM, zero-shot (ablation) |
| `agent` | LLM | LLM + keyword guard | LLM, grounded in 5 retrieved past AppleSupport replies |

Models: `gpt-4.1-mini` (agent) and `gpt-4.1` (judge), overridable via `AGENT_MODEL` / `JUDGE_MODEL`.
The judge is a different, stronger model than the agent so it doesn't grade its own outputs.

## Golden set: sampling and labelling

* **Pool.** Thread-opening customer tweets that AppleSupport answered, posted on or after 2017-11-22. The
  retrieval index and baselines only see earlier tweets, so this is a temporal hold-out. Near-duplicate texts
  were removed (the iOS 11.1 "i" bug alone produced thousands of clones).
* **120 uniform-random** messages represent natural traffic. **80 targeted** messages were found with keyword regexes for rare,
  high-cost situations: safety 15, money/account 20, data loss 15, frustrated/repeat contact 20,
  non-English 10 (picked using AppleSupport's own English-only redirect reply). Each row keeps its `stratum`,
  and every metric is reported for all 200 *and* for the uniform 120 only.
* **Labelling.** Done by Claude (Anthropic's model) rather than by hand, under the same blind conditions as
  `src/label.py`, which shows only the customer tweet. The order is shuffled, and Apple's
  reply and the stratum are hidden. Each tweet gets one intent, auto/escalate, and an escalation reason. Definitions are in
  `src/config.py`; edge-case conventions are written up in REPORT.md §2.
* **Judge check.** 20 tweets × 3 replies (AppleSupport's real reply, simple baseline, agent), shuffled and blind,
  rated on the same rubric the judge uses (`src/judge.py`).

## Attribution

* Dataset: Thought Vector, *Customer Support on Twitter* (Kaggle), CC BY-NC-SA 4.0.
* Libraries: pandas, scikit-learn (TF-IDF, metrics), scipy, openai, tabulate.
* Built with help from an AI coding assistant (Claude Code), which also produced the golden-set labels.
  All design choices are listed in DECISIONS.md.
