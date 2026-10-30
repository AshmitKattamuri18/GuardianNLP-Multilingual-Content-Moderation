"""Optional, conservative rule-assisted negation candidate.

This does not alter IndicGuard or its API. Supported, unambiguous personal-insult
scopes produce an explicitly rule-assisted proposal; all other inputs abstain to
the unchanged ensemble label. This limited lexicon is not a general toxicity model.
"""
from __future__ import annotations

import re
from typing import Any

# Deliberately small English/Romanized-Hindish coverage; unsupported wording abstains.
_INSULTS = ("stupid", "idiot", "useless", "foolish", "fool", "bewakoof", "badtameez", "nalayak")
_INSULT_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, _INSULTS)) + r")\b", re.IGNORECASE)
_NEGATORS = {"not", "never", "nahi", "nahin", "nhi"}
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
# Commas separate local scope; contrast markers start a new clause so their
# asserted second conjunct is not accidentally governed by the first negator.
_CLAUSE_SPLIT_RE = re.compile(r"[,;.!?]+|\b(?:but|however|whereas|balke|lekin)\b", re.IGNORECASE)
_REPORTING_RE = re.compile(r"\b(?:said|says|called|wrote|quoted|bola|boli|kaha)\b", re.IGNORECASE)


def _quoted_report(text: str, start: int, end: int) -> bool:
    """Return true for an insult quoted after a recognizable reporting verb."""
    left = text.rfind('"', 0, start)
    right = text.find('"', end)
    if left < 0 or right < 0:
        left = text.rfind("'", 0, start)
        right = text.find("'", end)
    return left >= 0 and right >= 0 and bool(_REPORTING_RE.search(text[max(0, left - 60):left]))


def _clause_scope(clause: str) -> list[tuple[str, bool]]:
    """Return (insult, is_denied) pairs using bounded local negator scope."""
    found: list[tuple[str, bool]] = []
    for match in _INSULT_RE.finditer(clause):
        before = clause[:match.start()]
        after = clause[match.end():]
        before_words = [m.group(0).casefold() for m in _WORD_RE.finditer(before)][-4:]
        after_words = [m.group(0).casefold() for m in _WORD_RE.finditer(after)][:3]
        before_negators = sum(word in _NEGATORS for word in before_words)
        # Romanized Hindi commonly places nahi after the predicate: "bewakoof nahi ho".
        after_negators = sum(word in _NEGATORS for word in after_words)
        is_denied = (before_negators + after_negators) % 2 == 1
        found.append((match.group(0).casefold(), is_denied))
    return found


def analyze_negation_scope(text: str) -> dict[str, Any]:
    """Analyze only supported personal-insult cues; otherwise explicitly abstain."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    matches = list(_INSULT_RE.finditer(text))
    if not matches:
        return {"status": "no_supported_insult", "rule_label": None, "evidence": []}
    if any(_quoted_report(text, m.start(), m.end()) for m in matches):
        return {"status": "uncertain_quoted_insult", "rule_label": None,
                "evidence": [{"insult": m.group(0), "scope": "quoted_report"} for m in matches]}

    evidence: list[dict[str, str]] = []
    for clause in _CLAUSE_SPLIT_RE.split(text):
        for insult, denied in _clause_scope(clause):
            evidence.append({"insult": insult, "scope": "denied" if denied else "asserted"})
    if not evidence:
        return {"status": "no_supported_insult", "rule_label": None, "evidence": []}
    if any(item["scope"] == "asserted" for item in evidence):
        return {"status": "asserted_insult", "rule_label": "OFFENSIVE", "evidence": evidence}
    return {"status": "denied_insult", "rule_label": "NORMAL", "evidence": evidence}


def apply_scope_candidate(text: str, ensemble_result: dict[str, Any]) -> dict[str, Any]:
    """Keep raw ensemble outputs intact and append a separately named candidate."""
    analysis = analyze_negation_scope(text)
    original = ensemble_result["ensemble_prediction"]
    assisted = analysis["rule_label"] is not None
    return {
        "original_ensemble_prediction": original,
        "candidate_prediction": analysis["rule_label"] if assisted else original,
        "candidate_method": "scope_rule_assisted" if assisted else "original_ensemble_fallback",
        "rule_assistance_applied": assisted,
        "scope_analysis": analysis,
    }


def predict_side_by_side(model: Any, text: str) -> dict[str, Any]:
    """Run genuine model inference, then append the isolated candidate result."""
    raw = model.predict(text)
    return {"text": text, "original": raw, "candidate": apply_scope_candidate(text, raw)}
