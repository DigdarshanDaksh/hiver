# Report: AppleSupport first-response agent

> Sections marked **TODO** depend on human labels and LLM runs. Fill them from `outputs/results.md`
> and `outputs/agent_errors.csv`.

## 1. Problem framing

**What the brand's support actually looks like.** In Oct–Dec 2017 AppleSupport answered 74k thread-opening tweets.
The first reply is almost never the resolution:

* 52% move the customer to DM, usually after asking for the device and iOS version (`Settings > General > About`).
* About 8% are a single canned reply: the iOS 11.1 "i → A [?]" autocorrect workaround (5,091 identical replies).
* About 3% redirect non-English tweets ("We offer support via Twitter in English…").
* Billing and Apple ID issues are sent to the iTunes Store or Account Security teams via a link.

**What "good" means here.** A good first reply (a) recognises the specific problem, (b) gives the *right next step*
(known fix, right team, or one precise diagnostic question + DM), (c) sounds like AppleSupport (warm, short, never
blames the user), and (d) never invents facts, links, dates or refunds. A good *system* sends as many of those
replies as possible unreviewed while **never** auto-sending on a safety hazard, a compromised account or money issue,
lost data, a legal threat, or a customer who has already tried the standard fixes.

So the metric I optimise is **auto_and_good**: the share of traffic that is auto-sent *and* judged sendable. It is
subject to a hard constraint on **missed escalations**.

**What I chose not to build:**
* Multi-turn / DM handling. The DM conversations aren't in the dataset, so there's nothing to ground them on.
* Actually resolving issues (KB lookup, account actions, real URLs). Links stay as the `<URL>` placeholder.
* A fine-tuned classifier. 200 labels are an evaluation set, not a training set.
* Banking77. Its intents don't transfer to consumer electronics.
* Sentiment-based escalation (see DECISIONS.md #4).

## 2. Data, intents, golden set

* Pairs: `src/prepare_data.py`. Golden sampling and labelling: README § Golden set.
* Intent definitions and escalation reasons: `src/config.py`.
* Labelling conventions for edge cases (**TODO: write yours as you label**). For example: a tweet with two issues →
  label the one that decides the handling; "battery dies after update" → battery_charging; Mac software problems
  → software_update_issue; keyword hit but not a real hazard ("X smokes iPhone in a speed test") → not escalated.

## 3. Results vs baselines

**TODO**: paste both tables from `outputs/results.md` (all 200 and uniform-only) and comment on:
* intent macro-F1 with CI: agent vs simple vs trivial.
* escalation recall, missed_escalations, auto_rate.
* judge overall / safe, and auto_and_good vs auto_but_bad.
* the retrieval ablation (agent vs agent_no_retrieval). Does grounding help style (sim_to_real_reply, voice) more
  than correctness?
* the reference point: the judge's score for AppleSupport's own replies.

## 4. Judge reliability

**TODO**: paste the agreement table. Report weighted kappa per criterion. If kappa on `overall` < 0.4, say the judge
is only usable for system ranking, not per-message gating. Check whether the judge ranks the three systems the same way you do.

## 5. Failure analysis: top 5 failure modes

**TODO**: from `outputs/agent_errors.csv`, group the errors, count them, and give each one a real example plus a
hypothesis plus a fix. Candidates to check for (based on the data, not yet confirmed):
1. Intent overlap: battery drain *after an update* (battery vs software_update).
2. Keyword-guard false alarms ("smokes", "killing my battery", "fire") push auto_rate down.
3. Escalations hidden inside sarcasm or a rant that the LLM reads as venting.
4. Replies that confidently promise a fix "in a future update" for issues Apple hasn't acknowledged.
5. Retrieval pulling the "i"-bug canned workaround for unrelated keyboard complaints (temporal drift).

## 6. What is misleading about my headline number?

Written ahead of the results; **TODO**: confirm each point with numbers.

* **n = 200, one labeller.** The bootstrap CIs are wide. I have no inter-annotator agreement on the gold labels, so
  "accuracy" means "agrees with me".
* **The stratified set over-represents risky cases.** Escalation recall on 200 is driven by 80 hand-picked
  examples. On natural traffic (uniform slice) there are only a handful of true escalations, so that recall is
  close to anecdotal.
* **The targeted strata were found with keyword regexes**, and the simple baseline and the agent's guard use similar
  keywords. Risky messages phrased *without* those keywords are under-sampled, so recall on them is overestimated.
* **"Good reply" is an LLM judge's opinion**, validated on only 60 human ratings. It can't check factual correctness
  of Apple-specific advice, and `<URL>` hides whether the right article would be linked.
* **One time window and one brand.** Late-2017 iOS 11 bug waves dominate, and new bugs aren't in the retrieval index.
* **auto_and_good assumes a judge-approved reply is safe to send.** A 4/5 reply posted publicly can still be wrong.

## 7. With one more week

* A second labeller on the golden set for inter-annotator kappa; adjudicate disagreements and refine intent definitions.
* A larger, *non-keyword* sample of risky cases (e.g. LLM-flagged from 5k random tweets, then human-verified) to measure
  escalation recall without the regex bias.
* Calibrate the escalate threshold: sweep a confidence score and plot auto_rate vs missed escalations, then pick an
  operating point with the brand.
* A KB of Apple's actual canned responses (cluster the 74k replies) so drafts pick a known template + slot fills instead
  of free text. This makes them more grounded and easier to audit.
* Shadow mode on fresh traffic: compare drafts to what human agents actually send, and track edit distance.
