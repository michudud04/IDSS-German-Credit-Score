"""Decision and language (Person D): final recommendation, risk slider, bank rules, texts.

Flow for one applicant:

    inputs --RiskPipeline--> probability of default per model + combined
           --UnusualCaseDetector (C)--> unusual-case check
           --rules.check_rules--> bank-policy hits
           --engine.decide(policy)--> Decision: approve / review / reject + reasons
           --narrative.writer--> officer summary + customer letter (LLM, guard, template fallback)

The ``RiskPolicy`` holds the slider setting (how costly a bad loan is compared
with a lost good customer) and turns it into the probability cut-off that the
advice module (C) and the app (E) use.

The app calls ``Source.decision.evaluate_decision`` (see service.py).
"""


def __getattr__(name):
    # Lazy import so that `import Source.decision.policy` stays light.
    if name == "evaluate_decision":
        from Source.decision.service import evaluate_decision
        return evaluate_decision
    raise AttributeError(name)
