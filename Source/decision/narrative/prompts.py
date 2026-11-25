"""System prompts. The facts are passed as JSON in the user message.

Small local models tend to fill every section of a template, even without
data, and to add generic advice. So the officer prompt lists only the sections
that have facts, and both prompts forbid advice that is not in the facts.
The guard (guard.py) rejects texts that still do it.
"""

from __future__ import annotations

import json

CUSTOMER_SYSTEM = """You write a short decision letter from a bank to a loan applicant.
Use ONLY the facts in the JSON you receive. Rules:
1. Start with "Dear applicant," and end with "Kind regards," followed by "The Credit Team" on the next line.
2. State the outcome clearly in the first paragraph, exactly as given:
   - "approved": the loan application is approved.
   - "under personal review by an advisor": an advisor will review the application personally; no decision yet. Do not say it is approved or declined.
   - "not approved": the bank is unable to approve the application at this time.
3. Explain the outcome with the given reasons and factors only, in plain words. Do not add any other statement about the applicant, their habits or their finances.
4. Advice: give advice only from these two lists, and only if they are present:
   - "changes_that_could_be_enough": changes that could bring a future application within the bank's criteria. Never promise or guarantee approval.
   - "changes_that_help_but_are_not_enough_alone": changes that would lower the risk but would NOT be enough on their own. Say so explicitly.
   Copy the changes as given. If neither list is present, give NO advice, tips or recommendations at all.
5. Never mention age, family situation, dependents, other customers or applicants, statistics, models, algorithms, scores or probabilities, unless the facts themselves mention them.
6. Do not invent anything: no amounts, numbers, dates, deadlines, products, phone numbers or names. Only use numbers that appear in the facts.
7. No placeholders such as [Name] or <date>.
8. Polite, respectful, plain English. 80 to 220 words. Output only the letter."""

OFFICER_SECTIONS = [
    # (facts key, section instruction); a section is requested only if its key is in the facts
    ("reasons", '"Why:" one bullet point per reason, restating that reason. No other bullet points: do not '
                'add your own judgement about the applicant.'),
    ("key_factors", '"Key factors:" each given factor with its value and effect. Write "(sensitive, never use '
                    'for advice)" after factors with sensitive_do_not_advise=true.'),
    ("similar_past_applicants", '"Similar past applicants:" one sentence with the given default rates.'),
    ("unusual_case_warning", '"Unusual-case warning:" the given warning.'),
    ("what_if_advice", '"Advice for the customer:" the given what-if changes and their new probabilities.'),
]


def officer_system(facts: dict) -> str:
    sections = [text for key, text in OFFICER_SECTIONS if facts.get(key)]
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(sections, start=2))
    return f"""You write a concise case summary for a bank loan officer.
Use ONLY the facts in the JSON you receive. Write exactly these parts, in this order, and nothing else:
1. First line: "Recommendation: {facts['recommendation']}" with the probability of default and the cut-off.
{numbered}
{len(sections) + 2}. Last line: "The officer takes the final decision."
Rules: do not add any other section, heading, advice or comment. Do not invent numbers or facts; only use numbers from the JSON. No placeholders. At most 200 words. Output only the summary."""


def user_message(facts: dict, feedback: list[str] | None = None) -> str:
    message = "Facts:\n" + json.dumps(facts, indent=2, ensure_ascii=False)
    if feedback:
        message += ("\n\nYour previous draft broke these rules, fix them:\n- " + "\n- ".join(feedback))
    return message
