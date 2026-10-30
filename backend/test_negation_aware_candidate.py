"""Focused tests for the optional, rule-assisted negation candidate."""
import pytest

from negation_aware_candidate import analyze_negation_scope


@pytest.mark.parametrize("text", [
    "Mira is foolish.",
    "That driver is a fool.",
    "Woh nalayak hai.",
])
def test_asserted_insult_is_offensive(text):
    result = analyze_negation_scope(text)
    assert result["status"] == "asserted_insult"
    assert result["rule_label"] == "OFFENSIVE"


@pytest.mark.parametrize("text", [
    "Mira is not foolish.",
    "Woh nalayak nahi hai.",
])
def test_insult_denial_is_normal(text):
    result = analyze_negation_scope(text)
    assert result["status"] == "denied_insult"
    assert result["rule_label"] == "NORMAL"


def test_two_negators_do_not_cancel_to_normal():
    result = analyze_negation_scope("The coach is not not a fool.")
    assert result["status"] == "asserted_insult"
    assert result["rule_label"] == "OFFENSIVE"


def test_reported_quoted_insult_abstains():
    result = analyze_negation_scope("He said, 'Mira is a fool.'")
    assert result["status"] == "uncertain_quoted_insult"
    assert result["rule_label"] is None


def test_contrastive_english_second_insult_remains_asserted():
    result = analyze_negation_scope("Mira is not foolish but a fool.")
    assert result["status"] == "asserted_insult"
    assert result["rule_label"] == "OFFENSIVE"


def test_hindish_sirf_contrast_second_insult_remains_asserted():
    result = analyze_negation_scope("Tum sirf nalayak nahi, badtameez bhi ho.")
    assert result["status"] == "asserted_insult"
    assert result["rule_label"] == "OFFENSIVE"


def test_negation_unrelated_to_insult_does_not_cancel_insult():
    result = analyze_negation_scope("I do not like traffic, but Mira is a fool.")
    assert result["status"] == "asserted_insult"
    assert result["rule_label"] == "OFFENSIVE"


def test_no_supported_insult_returns_no_rule_label():
    result = analyze_negation_scope("I do not like traffic.")
    assert result["status"] == "no_supported_insult"
    assert result["rule_label"] is None
