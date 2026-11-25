"""Checks on generated text. A text that fails is regenerated once, then replaced by the template.

Customer letter:
- no sensitive or internal topic that is not in its facts (age, family,
  other customers, models, probabilities, scores);
- every number must come from the facts (no invented amounts, dates, rates);
- the outcome must match the decision (no "approved" in a rejection, etc.);
- no template placeholders such as "[Your Name]".

Officer summary: numbers from the facts, the recommendation stated, no placeholders.

The checks are deliberately simple pattern rules: they are explainable in the
report, and a false alarm only costs a fallback to the template text.
"""

from __future__ import annotations

import json
import re

from Source.decision.narrative.facts import ADVICE_ENOUGH, ADVICE_NOT_ENOUGH

# A topic is allowed only if it already appears in the facts (e.g. the legal-age rule may mention age).
CUSTOMER_FORBIDDEN_TOPICS = {
    "age": r"\b(age|aged|years? old|young|younger|older|elderly|born|retire\w*)\b",
    "family situation": r"\b(dependents?|children|child|kids|family|married|spouse|partner)\b",
    "other customers": r"\b(other|similar|past|previous)\s+(customers?|applicants?|clients?|cases?|profiles?|borrowers?)\b",
    "internal model details": r"\b(models?|algorithms?|xgboost|logistic|decision tree|machine learning|AI|score|scoring)\b",
    "probabilities": r"\b(probabilit\w*|likelihood|percent chance)\b",
}

OUTCOME_PATTERNS = {
    "approve": {"required": r"\bapprov", "forbidden": r"\b(declin\w*|reject\w*|unfortunately|not (been )?approved|"
                                                      r"cannot|unable|under review)\b"},
    "review": {"required": r"\b(review\w*|advis[oe]r)\b", "forbidden": r"\b(congratulat\w*|has been approved|is approved|"
                                                                     r"has been (declined|rejected))\b"},
    "reject": {"required": r"\b(declin\w*|unable|not able|cannot|can ?not|not (been )?approved|reject\w*|"
                           r"not in a position)\b",
               "forbidden": r"\b(congratulat\w*|has been approved|is approved|pleased to (approve|confirm))\b"},
}

# Advice-like wording, allowed in the letter only when the facts contain advice.
ADVICE_WORDING = (r"\b(consider(ing)?|we (recommend|suggest|advise|encourage)|tips?|to improve|you (should|could|may "
                  r"want to|might)|keep(ing)? (your )?\w+ low|make (timely|regular) payments|monitor\w*)\b")

# Never allowed in a letter: the bank cannot promise a future decision.
PROMISE = r"\b(guarantee\w*|will (be|get) approved|(ensure|secure)s? (your |an )?approval|certain(ly)? (be )?approved)\b"
# Allowed only when some change is enough on its own ("would not be enough" does not match).
ENOUGH_CLAIM = (r"\b(would|will|could|should|may|might) (be enough|be sufficient|suffice|qualify|"
                r"bring (a|your) (future )?application within)\b")

# Officer sections that must not appear when the facts have nothing for them.
OFFICER_SECTION_TOPICS = {
    "key factors": r"key[ _]factors?",
    "advice": r"\badvi[cs]e\b|what[ _]if",
    "unusual-case warning": r"unusual",
    "similar past applicants": r"similar",
}

PLACEHOLDER = r"\[[^\]]{2,40}\]|<[^>]{2,40}>|\{[^}]{2,40}\}"
NUMBER = r"\d[\d,]*(?:\.\d+)?"


def numbers_in(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(NUMBER, text)]


def _invented_numbers(text: str, facts: dict) -> list[str]:
    allowed = numbers_in(json.dumps(facts, ensure_ascii=False))
    invented = []
    for raw in re.findall(NUMBER, text):
        value = float(raw.replace(",", ""))
        # Rounding is fine (18.4% -> 18%); anything else is invented.
        if not any(abs(value - a) <= max(0.5, 0.01 * abs(a)) for a in allowed):
            invented.append(raw)
    return invented


def check_customer_letter(text: str, facts: dict, outcome: str) -> list[str]:
    issues = []
    facts_text = json.dumps(facts, ensure_ascii=False)
    for topic, pattern in CUSTOMER_FORBIDDEN_TOPICS.items():
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found and not re.search(pattern, facts_text, flags=re.IGNORECASE):
            issues.append(f"mentions {topic} ('{found.group(0)}'), which the letter must not do")
    rules = OUTCOME_PATTERNS[outcome]
    if not re.search(rules["required"], text, flags=re.IGNORECASE):
        issues.append(f"does not state the outcome clearly ({outcome})")
    found = re.search(rules["forbidden"], text, flags=re.IGNORECASE)
    if found:
        issues.append(f"contradicts the outcome {outcome} ('{found.group(0)}')")
    if not (facts.get(ADVICE_ENOUGH) or facts.get(ADVICE_NOT_ENOUGH)):
        found = re.search(ADVICE_WORDING, text, flags=re.IGNORECASE)
        if found:
            issues.append(f"gives advice that is not in the facts ('{found.group(0)}'); give no advice at all")
    found = re.search(PROMISE, text, flags=re.IGNORECASE)
    if found:
        issues.append(f"promises an approval ('{found.group(0)}'); never promise or guarantee approval")
    if not facts.get(ADVICE_ENOUGH):
        found = re.search(ENOUGH_CLAIM, text, flags=re.IGNORECASE)
        if found:
            issues.append(f"presents a change as enough ('{found.group(0)}'), but no change is enough on its own")
    issues += _common_issues(text, facts, min_len=150, max_len=2500)
    return issues


def check_officer_summary(text: str, facts: dict) -> list[str]:
    issues = []
    if facts["recommendation"].lower() not in text.lower():
        issues.append(f"does not state the recommendation {facts['recommendation']}")
    facts_text = json.dumps(facts, ensure_ascii=False)
    for section, pattern in OFFICER_SECTION_TOPICS.items():
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found and not re.search(pattern, facts_text, flags=re.IGNORECASE):
            issues.append(f"adds a '{section}' part although the facts have none; remove it")
    issues += _common_issues(text, facts, min_len=150, max_len=3000)
    return issues


def _common_issues(text: str, facts: dict, min_len: int, max_len: int) -> list[str]:
    issues = []
    invented = _invented_numbers(text, facts)
    if invented:
        issues.append("uses numbers that are not in the facts: " + ", ".join(invented[:5]))
    found = re.search(PLACEHOLDER, text)
    if found:
        issues.append(f"contains a placeholder ('{found.group(0)}')")
    if not min_len <= len(text) <= max_len:
        issues.append(f"length {len(text)} characters outside {min_len}-{max_len}")
    return issues
