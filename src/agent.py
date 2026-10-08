"""The LLM support agent: classify intent, draft a grounded reply, decide auto vs escalate."""
from config import AGENT_MODEL, ESCALATION_REASONS, INTENT_LIST, INTENTS, REASON_LIST
from llm import chat_json
from retrieval import get_retriever
from rules import rule_escalation

SYSTEM = f"""You are the first-response agent for @AppleSupport on Twitter.
For each incoming customer tweet you must (1) classify its intent, (2) draft the public
reply, and (3) decide whether the reply can be posted automatically or a human must handle it.

INTENTS:
{chr(10).join(f"- {k}: {v}" for k, v in INTENTS.items())}

ESCALATE to a human (escalate=true) if ANY of these apply, otherwise escalation_reason="none":
{chr(10).join(f"- {k}: {v}" for k, v in ESCALATION_REASONS.items())}
Anger or profanity alone is NOT a reason to escalate; AppleSupport answers angry tweets routinely.

HOW APPLESUPPORT REPLIES (follow the style of the past replies you are shown):
- Max 280 characters. Warm, calm, first-person plural ("We're here to help"). Never blame the customer, never argue.
- Known issue with a published fix/workaround (e.g. the iOS 11.1 'i' autocorrect bug): point to it, "<URL>".
- Anything needing device, account or personal details: ask ONE diagnostic question (e.g. device + iOS version
  from Settings > General > About) and invite them to DM: "Let's continue in DM. <URL>".
- Billing/account: point them to the right team (iTunes Store support / Apple ID account security) via "<URL>".
- Non-English: reply in English that Twitter support is English-only and point to local support: <URL>.
- Never invent URLs (always use the literal placeholder <URL>), release dates, policies, prices, refunds or
  promises. Never ask for passwords, card numbers or serial numbers in public.
- If the tweet is pure venting/joke with nothing actionable, a short friendly offer to help is fine.
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "confidence", "escalate", "escalation_reason", "reason", "reply"],
    "properties": {
        "intent": {"type": "string", "enum": INTENT_LIST},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "escalate": {"type": "boolean"},
        "escalation_reason": {"type": "string", "enum": REASON_LIST + ["none"]},
        "reason": {"type": "string", "description": "One sentence justifying the escalate/auto decision."},
        "reply": {"type": "string"},
    },
}


def build_prompt(text: str, examples: list[dict]) -> str:
    parts = []
    if examples:
        parts.append("PAST APPLESUPPORT THREADS SIMILAR TO THIS ONE (customer -> AppleSupport first reply):")
        for i, ex in enumerate(examples, 1):
            parts.append(f"{i}. Customer: {ex['customer_text']}\n   AppleSupport: {ex['reply_text']}")
        parts.append("")
    parts.append(f"NEW CUSTOMER TWEET:\n{text}")
    return "\n".join(parts)


def run_agent(text: str, use_retrieval: bool = True, k: int = 5, model: str = AGENT_MODEL) -> dict:
    examples = get_retriever().search(text, k) if use_retrieval else []
    out = chat_json(model, SYSTEM, build_prompt(text, examples), SCHEMA, "support_decision")
    out["llm_escalate"] = out["escalate"]

    # deterministic guard: rules can only ADD escalations, never remove them
    hit, rule_reason = rule_escalation(text)
    if hit and not out["escalate"]:
        out["escalate"], out["escalation_reason"] = True, rule_reason
        out["reason"] = f"[rule guard: {rule_reason}] " + out["reason"]
    if out["confidence"] == "low" and not out["escalate"]:
        out["escalate"], out["escalation_reason"] = True, "unclear"
        out["reason"] = "[low confidence] " + out["reason"]
    out["retrieved"] = [e["customer_text"][:80] for e in examples]
    return out


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(run_agent(" ".join(sys.argv[1:]) or "my iphone gets really hot and the battery dies by noon"), indent=2))
