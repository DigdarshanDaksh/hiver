"""Keyword rules. Used (a) as the 'simple' baseline classifier and (b) as a
deterministic safety guard layered on top of the LLM agent's escalation decision."""
import re

INTENT_RULES = [  # first match wins; order matters
    ("keyboard_i_bug", r"\bi\b.{0,20}(a ?\[\?\]|question mark|box)|\[\?\]|letter [\"“']?i\b|\bi ?bug|autocorrect.{0,30}\bi\b"),
    ("hardware_damage_safety", r"smok|fire|burn|melt|swell|explod|spark|shock|crack|shatter|broke|water damage|liquid|screen (is )?(black|dead|broken)"),
    ("account_billing", r"charged|refund|hack|locked|apple ?id|password|subscri|billing|payment|credit|itunes account|icloud account|verification|2fa|two.factor"),
    ("battery_charging", r"batter|charg|drain|overheat|\bhot\b|dies|dying|shut(s|ting)? ?(down|off)|power"),
    ("app_service_issue", r"music|icloud|app ?store|siri|imessage|messages|facetime|photos|podcast|apple pay|itunes|airdrop|maps|mail\b|notes|health|watch"),
    ("software_update_issue", r"update|ios ?1\d|high ?sierra|slow|lag|freez|crash|glitch|bug|restart|reboot|frozen|not working|won'?t|doesn'?t work|stopped"),
    ("how_to_product_question", r"\bhow (do|can|to)\b|\bcan i\b|is there a way|where can|when will|does .* support|\?$"),
]

ESCALATION_RULES = [
    ("safety", r"smok|fire|flames?\b|burn(ed|t|ing)? (my|me|hand|skin)|melt|swell|swollen|explod|spark|electr?ocut|shocked|injur"),
    ("account_security_or_money", r"hack|stolen|unauthori|charged (me )?(twice|double)|fraud|locked out|refund|someone (used|accessed)"),
    ("data_loss", r"lost (all )?(of )?(my )?(photos|pictures|pics|contacts|data|messages|texts|notes|everything)|(deleted|erased|wiped) (all|everything|my)"),
    ("legal_or_media", r"lawsuit|class action|\bsue\b|lawyer|attorney|\bbbb\b|consumer (rights|protection)|journalist"),
    ("failed_troubleshooting", r"already tried|tried everything|nothing works|\b(\d|three|four|five)(rd|th)? (time|day)s?\b|for (days|weeks|months)|genius bar|still (not|doesn)"),
]

NON_ASCII_LETTERS = re.compile(r"[à-ÿ¿¡ñ]|[Ѐ-ӿ؀-ۿ぀-ヿ一-鿿]")
EN_STOP = {"the", "my", "is", "to", "and", "it", "i", "you", "a", "of", "this", "why", "on", "in", "for", "with", "can", "me", "your", "not"}


def looks_non_english(text: str) -> bool:
    toks = re.findall(r"[a-zA-Zà-ÿ]+", text.lower())
    if len(toks) < 3:
        return False
    en = sum(t in EN_STOP for t in toks) / len(toks)
    return en < 0.06 and (bool(NON_ASCII_LETTERS.search(text)) or en == 0)


def rule_intent(text: str) -> str:
    if looks_non_english(text):
        return "non_english"
    t = text.lower()
    for intent, pat in INTENT_RULES:
        if re.search(pat, t):
            return intent
    return "feedback_other"


def rule_escalation(text: str) -> tuple[bool, str | None]:
    t = text.lower()
    for reason, pat in ESCALATION_RULES:
        if re.search(pat, t):
            return True, reason
    return False, None
