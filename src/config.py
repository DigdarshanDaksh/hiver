"""Shared paths, intent taxonomy and escalation policy."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

PAIRS = ROOT / "data/processed/apple_pairs.parquet"
GOLDEN_DIR = ROOT / "data/golden"
GOLDEN_CANDIDATES = GOLDEN_DIR / "golden_set.csv"          # sampled messages (no labels)
GOLDEN_LABELS = GOLDEN_DIR / "golden_labels.csv"            # human intent/escalation labels
HUMAN_RATINGS = GOLDEN_DIR / "human_reply_ratings.csv"      # human reply-quality ratings
OUT = ROOT / "outputs"
PRED_DIR = OUT / "predictions"
JUDGE_DIR = OUT / "judge"

# Temporal split: retrieval index / baselines only see tweets before this date;
# the golden set is sampled from on/after it (simulates deploying on future traffic).
SPLIT_DATE = "2017-11-22"

AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4.1-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1")
SEED = 13

# Intent taxonomy, derived from reading ~300 AppleSupport threads (see REPORT.md).
INTENTS = {
    "keyboard_i_bug": "The iOS 11.1 autocorrect bug where typing 'i' becomes 'A [?]' / boxed question mark.",
    "software_update_issue": "Device slow, buggy, freezing, crashing or glitching (often after an iOS update). Not battery, not the 'i' bug.",
    "battery_charging": "Battery drain, overheating, charging problems, unexpected shutdowns.",
    "hardware_damage_safety": "Physical damage or malfunction: cracked/dead screen, broken buttons, liquid, smoking/swollen/burning devices or cables.",
    "account_billing": "Apple ID, iCloud account access, passwords, hacked accounts, charges, refunds, subscriptions, payment methods.",
    "app_service_issue": "A specific Apple app or service misbehaving: Music, iCloud sync/storage, App Store, Siri, Messages, FaceTime, Photos, Podcasts, Apple Pay.",
    "how_to_product_question": "How-to, feature or compatibility questions, purchase/availability/store questions, no fault reported.",
    "non_english": "Message is not in English (AppleSupport's Twitter is English-only and redirects these).",
    "feedback_other": "Venting, jokes, praise, opinions or spam with no specific actionable issue.",
}
INTENT_LIST = list(INTENTS)

# Escalation policy. "Auto-handle" = the agent's public first reply is posted without a
# human reading it first. "Escalate" = route to a human before anything is posted.
ESCALATION_REASONS = {
    "safety": "Possible physical harm: smoke, fire, burning, swelling, shock, injury.",
    "account_security_or_money": "Account compromise, lockout, unauthorized/duplicate charges, refunds — needs identity verification.",
    "data_loss": "Customer says personal data (photos, contacts, messages, backups) is lost.",
    "legal_or_media": "Legal threats, regulators, press, or a public-relations risk.",
    "failed_troubleshooting": "Customer says standard fixes were already tried, or this is a repeat contact / long unresolved issue.",
    "unclear": "Agent cannot tell what the customer needs with reasonable confidence.",
}
REASON_LIST = list(ESCALATION_REASONS)
