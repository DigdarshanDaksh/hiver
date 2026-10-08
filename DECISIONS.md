# Decision log

1. **AppleSupport as the brand.** It has the most volume (74k answered thread openers), a consistent voice, and a clear
   mix of templated cases (known-bug workaround, English-only redirect) and cases that need a human.
2. **Only thread openers paired with the brand's first reply.** The agent's job is first response. Later turns
   happen mostly in DM and aren't in the data, so there's nothing to ground them on.
3. **"Auto-handle" means "the first public reply can be posted without a human reading it".** It does *not* mean "issue
   resolved". 52% of AppleSupport's own first replies send people to DM, so "moves to DM" can't be used as an
   escalation label. That would mark half of all traffic as escalated and teach nothing.
4. **Escalation is defined by risk, not by difficulty.** It covers safety, money/account security, data loss, legal/media, failed
   troubleshooting, and low confidence. Anger alone is *not* a trigger: AppleSupport answers angry tweets routinely, and
   escalating on sentiment would flood humans.
5. **9 intents, picked from reading threads rather than clustering.** Clusters on tweets mostly group by device
   name. Intents should map to *different handling* (workaround link, billing team, DM diagnostics, redirect).
   The iOS 11.1 "i" bug has its own intent because it has its own canned answer.
6. **`non_english` is an intent.** AppleSupport sends a fixed English-only redirect. It's cheap to get right, and a
   generic LLM would otherwise happily reply in French.
7. **Temporal split (index < 2017-11-22 ≤ golden).** A random split would put near-identical tweets from the same
   bug wave on both sides and inflate retrieval quality. The temporal split also shows real drift: the "i" bug fades and
   iOS 11.2 issues appear.
8. **Stratified golden set (120 uniform + 80 targeted), with stratum stored.** 200 uniform samples would have about
   2–5 safety cases, too few to measure the most important failure. Uniform-only metrics are reported next to
   the overall ones so oversampling doesn't hide the natural-traffic picture.
9. **Labeller can't see Apple's reply or the stratum.** Apple's reply would anchor the intent label, and the stratum would
   anchor escalation. The tradeoff is that some ambiguous tweets are harder to label.
10. **Keyword guard can only add escalations, never remove them.** A missed hazard costs far more than a false alarm. Rules
    are auditable and catch the LLM ignoring "my cable is smoking" inside a long rant. This trades precision for recall
    on purpose.
11. **Retrieval is TF-IDF, not embeddings.** It's fast (2.5 s to index 63k tweets), runs offline, is deterministic, and makes
    reproducibility easy. Tweets are short and keyword-heavy (device names, iOS versions), which is where TF-IDF is
    competitive. Retrieved examples are de-duplicated by reply template so 5 copies of one canned reply don't dominate.
12. **Links are the literal `<URL>` placeholder.** The dataset anonymises links, and the agent can't know Apple's KB
    URLs. Any real `http` URL in a draft is counted as invented.
13. **Judge model ≠ agent model, and the judge also scores AppleSupport's real replies.** This avoids
    self-preference, and Apple's own replies calibrate the scale: if Apple scores 3/5, a 3/5 agent is "human-level",
    not "mediocre".
14. **Headline metric is `auto_and_good`, not accuracy.** That's the share of traffic the agent would send unreviewed *and* that the
    judge rates ≥ 4 and safe. It's paired with `missed_escalations` (absolute count) because one missed safety case
    matters more than five percentage points of intent accuracy.
15. **Failed LLM calls are scored as escalate + wrong intent + empty reply,** not dropped, so API flakiness can't
    quietly improve the numbers.
