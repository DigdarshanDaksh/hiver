"""Two non-LLM baselines.

trivial: every message -> most common intent in the held-in window (by rule labels),
         never escalate, reply with AppleSupport's most frequent generic DM template.
simple:  keyword rules for intent + escalation; reply = the historical AppleSupport reply
         to the most similar past tweet (TF-IDF nearest neighbour), copied verbatim.
"""
from functools import lru_cache

from retrieval import get_retriever
from rules import rule_escalation, rule_intent


@lru_cache(maxsize=1)
def _trivial_constants():
    r = get_retriever()
    intent = r.df.customer_text.sample(5000, random_state=0).map(rule_intent).value_counts().idxmax()
    replies = r.df.reply_text
    dm = replies[replies.str.contains(r"\bDM\b") & ~replies.str.contains("workaround")]  # generic, issue-agnostic
    return intent, dm.value_counts().idxmax()


def trivial(text: str) -> dict:
    intent, reply = _trivial_constants()
    return {"intent": intent, "escalate": False, "escalation_reason": "none",
            "reason": "trivial baseline", "reply": reply}


def simple(text: str) -> dict:
    esc, why = rule_escalation(text)
    nn = get_retriever().search(text, k=1, diverse=False)[0]
    return {"intent": rule_intent(text), "escalate": esc, "escalation_reason": why or "none",
            "reason": f"keyword rule: {why}" if esc else "no escalation keyword", "reply": nn["reply_text"]}
