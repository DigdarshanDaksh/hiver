# Report: AppleSupport first-response agent

> **Status.** The golden set is labelled and the two non-LLM baselines (`trivial`, `simple`) are fully evaluated
> below. The LLM agent, its no-retrieval ablation and the LLM judge have **not been run yet**: the pipeline needs an
> `OPENAI_API_KEY` that was not available when this was submitted. Sections 3–5 mark exactly which numbers are
> pending. Run `python src/run_all.py --fresh` with a key to fill them in; nothing else needs changing.
>
> **Who labelled the data.** The 200 golden-set labels (intent, escalate, reason) were produced by Claude
> (Anthropic's model, used through Claude Code), not by me. Claude labelled from the customer tweet only, with
> AppleSupport's reply and the sampling stratum hidden, following the definitions in `src/config.py` and the rules
> in §2. Treat every "accuracy" number here as agreement with that annotator.

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
* Labels: `data/golden/golden_labels.csv` (annotated by Claude, see the note at the top; tricky cases carry a `note`).
* Labelling conventions used for edge cases:
  * A tweet with two issues → the one that decides the handling. "Battery dies after update" → battery_charging.
    Mac software problems → software_update_issue.
  * Anger, swearing or "switching to Android" is never an escalation reason on its own.
  * Hyperbole is not a hazard: "X smokes iPhone in a speed test", a laptop "burning my legs" under heavy load → not
    escalated. Smoke, melting, fire, swelling, electric shock, or a phone too hot to hold while charging → safety.
  * failed_troubleshooting only when the customer reports several attempts, a store/AppleCare visit, an open case,
    or an issue unresolved for days. A single restart does not count.
  * The "it → I.T." autocorrect bug is labelled keyboard_i_bug (same iOS 11.1 autocorrect family).
  * Image-only or content-free tweets ("WHAT IS THIS?! <URL>") → escalate as unclear: the agent cannot see images.
  * Non-English → non_english, escalated only if an escalation reason applies regardless of language
    (e.g. contacts deleted → data_loss).
  * Refund/replacement demands and disputed charges → account_security_or_money. A phishing pop-up with no account
    compromise → not escalated. Public security holes (macOS root-without-password) → legal_or_media.
* Resulting labels: 65 / 200 escalated (33%); 22 / 120 (18%) on the uniform slice. Reasons: failed_troubleshooting 22,
  data_loss 16, account_security_or_money 10, safety 8, unclear 7, legal_or_media 2.

## 3. Results vs baselines

From `outputs/results.md`. Baselines only; agent columns are pending (see Status).

All 200 (stratified, oversamples risky cases):

| metric | trivial | simple | agent |
|---|---|---|---|
| intent_acc | 0.34 | 0.56 | pending |
| intent_macroF1 | 0.06 [0.05-0.06] | 0.57 [0.50-0.63] | pending |
| esc_recall | 0.00 | 0.46 [0.34-0.59] | pending |
| esc_precision | 0.00 | 0.65 | pending |
| auto_rate | 1.00 | 0.77 | pending |
| missed_escalations | 65 | 35 | pending |
| sim_to_real_reply | 0.15 | 0.18 | pending |

Uniform slice (120, natural traffic):

| metric | trivial | simple | agent |
|---|---|---|---|
| intent_acc | 0.40 | 0.52 | pending |
| intent_macroF1 | 0.06 [0.05-0.07] | 0.50 [0.37-0.57] | pending |
| esc_recall | 0.00 | 0.14 [0.00-0.29] | pending |
| esc_precision | 0.00 | 0.50 | pending |
| auto_rate | 1.00 | 0.95 | pending |
| missed_escalations | 22 | 19 | pending |

What the baselines already show:
* Keyword rules get intent macro-F1 to ~0.5–0.57, far above the majority-class floor (0.06). That is the bar the
  agent has to clear.
* The keyword escalation rule looks passable on the stratified set (recall 0.46) but collapses on natural traffic
  (0.14, 3 of 22 caught). The targeted strata were found with similar keywords, so the stratified number flatters
  any keyword-based method. This is the main reason the agent's LLM escalation decision matters.
* Never-escalate (trivial) auto-sends everything and misses all 65 escalations, so auto_rate alone is meaningless
  without the missed-escalation constraint.

Still to report once the LLM runs exist:
* intent macro-F1 with CI: agent vs simple vs trivial.
* escalation recall, missed_escalations, auto_rate.
* judge overall / safe, and auto_and_good vs auto_but_bad.
* the retrieval ablation (agent vs agent_no_retrieval). Does grounding help style (sim_to_real_reply, voice) more
  than correctness?
* the reference point: the judge's score for AppleSupport's own replies.

## 4. Judge reliability

**Pending**: needs the judge scores and the 60 blind reply ratings, which in turn need the agent's replies. The
ratings will be done by the same annotator as the golden set (Claude), so this will measure judge-vs-second-model
agreement, not judge-vs-human. Plan: report weighted kappa per criterion. If kappa on `overall` < 0.4, say the judge
is only usable for system ranking, not per-message gating. Check whether the judge ranks the three systems the same way you do.

## 5. Failure analysis: top 5 failure modes

**Pending** the agent run (`outputs/agent_errors.csv`). Hypotheses written before seeing agent output; the golden
labels already contain examples of each (row notes in `golden_labels.csv`):
1. Intent overlap: battery drain *after an update* (battery vs software_update).
2. Keyword-guard false alarms ("smokes", "killing my battery", "fire") push auto_rate down.
3. Escalations hidden inside sarcasm or a rant that the LLM reads as venting.
4. Replies that confidently promise a fix "in a future update" for issues Apple hasn't acknowledged.
5. Retrieval pulling the "i"-bug canned workaround for unrelated keyboard complaints (temporal drift).

## 6. What is misleading about my headline number?

Written ahead of the agent results; numbers below are from the labels and baselines.

* **n = 200, one labeller, and that labeller is a model.** The bootstrap CIs are wide (simple's macro-F1 on the
  uniform slice: 0.37–0.57). There is no inter-annotator agreement, and the labels came from Claude, so "accuracy"
  means "agrees with Claude", and an LLM agent may agree with an LLM labeller more than a human would.
* **The stratified set over-represents risky cases.** Escalation recall on 200 is driven by 80 hand-picked
  examples. On natural traffic (uniform slice) there are only 22 true escalations, so that recall is
  close to anecdotal (simple baseline: 0.14 with CI 0.00–0.29).
* **The targeted strata were found with keyword regexes**, and the simple baseline and the agent's guard use similar
  keywords. Risky messages phrased *without* those keywords are under-sampled, so recall on them is overestimated. Already
  visible: the keyword baseline's recall drops from 0.46 (all) to 0.14 (uniform).
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
