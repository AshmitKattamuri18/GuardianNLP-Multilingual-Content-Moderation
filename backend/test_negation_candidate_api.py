"""HTTP integration coverage for the opt-in scope candidate."""
from types import ModuleType
import sys

from fastapi.testclient import TestClient
import main


class CandidateFixtureModel:
    version = "api-candidate-test"
    device = "cpu"

    def predict(self, text):
        # Intentionally makes the base ensemble disagree on asserted and denied cues.
        if "not foolish" in text.casefold():
            probs = [0.15, 0.70, 0.15]
        elif "foolish" in text.casefold():
            probs = [0.80, 0.10, 0.10]
        else:
            probs = [0.20, 0.20, 0.60]
        return {
            "predicted_class": max(range(3), key=probs.__getitem__),
            "predicted_label": ("NORMAL", "OFFENSIVE", "HATE_SPEECH")[max(range(3), key=probs.__getitem__)],
            "lite_prediction": "NORMAL",
            "muril_prediction": "NORMAL",
            "ensemble_prediction": ("NORMAL", "OFFENSIVE", "HATE_SPEECH")[max(range(3), key=probs.__getitem__)],
            "lite_probabilities": [0.8, 0.1, 0.1],
            "muril_probabilities": [0.8, 0.1, 0.1],
            "ensemble_probabilities": probs,
            "confidence": max(probs),
            "model_version": self.version,
            "final_score": probs[1] + probs[2],
            "muril_score": 0.2,
        }


def install_fake_runtime(monkeypatch):
    runtime = ModuleType("indicguard_runtime")
    runtime.IndicGuard = lambda _bundle: CandidateFixtureModel()
    monkeypatch.setitem(sys.modules, "indicguard_runtime", runtime)


def test_candidate_disabled_preserves_ensemble_and_response_schema(monkeypatch):
    install_fake_runtime(monkeypatch)
    monkeypatch.setattr(main, "ENABLE_NEGATION_CANDIDATE", False, raising=False)
    with TestClient(main.app) as client:
        response = client.post("/predict", json={"text": "Mira is foolish."})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == "NORMAL"
    assert body["ensemble_prediction"] == "NORMAL"
    assert body["ensemble_probabilities"] == [0.8, 0.1, 0.1]
    assert "candidate" not in body and "scope_analysis" not in body
    assert set(body) == {
        "predicted_class", "predicted_label", "confidence", "probabilities",
        "model_predictions", "model_probabilities", "ensemble_probabilities",
        "lite_probabilities", "muril_probabilities", "lite_prediction",
        "muril_prediction", "ensemble_prediction", "final_score", "muril_score",
        "model_version", "class_order",
    }


def test_candidate_enabled_changes_supported_scope_without_exposing_heuristic(monkeypatch):
    install_fake_runtime(monkeypatch)
    monkeypatch.setattr(main, "ENABLE_NEGATION_CANDIDATE", True, raising=False)
    with TestClient(main.app) as client:
        response = client.post("/predict", json={"text": "Mira is foolish."})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == "OFFENSIVE"
    assert body["predicted_class"] == 1
    assert body["ensemble_prediction"] == "NORMAL"  # Original model prediction.
    assert body["ensemble_probabilities"] == [0.8, 0.1, 0.1]
    assert body["confidence"] == 0.8  # Original model confidence, not a fabricated rule score.
    assert body["label_was_rule_assisted"] is True
    assert body["final_score"] == 0.2
    assert "scope_analysis" not in body


def test_candidate_enabled_uses_denial_and_ambiguous_quote_falls_back(monkeypatch):
    install_fake_runtime(monkeypatch)
    monkeypatch.setattr(main, "ENABLE_NEGATION_CANDIDATE", True, raising=False)
    with TestClient(main.app) as client:
        denied = client.post("/predict", json={"text": "Mira is not foolish."})
        quoted = client.post("/predict", json={"text": "He said, 'Mira is a fool.'"})
    assert denied.status_code == quoted.status_code == 200
    assert denied.json()["predicted_label"] == "NORMAL"
    assert denied.json()["ensemble_probabilities"] == [0.15, 0.70, 0.15]
    assert quoted.json()["predicted_label"] == "HATE_SPEECH"
    assert quoted.json()["ensemble_probabilities"] == [0.20, 0.20, 0.60]
    assert "candidate" not in quoted.json() and "scope_analysis" not in quoted.json()


def test_candidate_corrected_normal_forces_allow_on_moderate(monkeypatch):
    install_fake_runtime(monkeypatch)
    monkeypatch.setattr(main, "ENABLE_NEGATION_CANDIDATE", True, raising=False)
    with TestClient(main.app) as client:
        response = client.post("/moderate", json={"text": "Mira is not foolish."})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == "NORMAL"
    assert body["ensemble_prediction"] == "OFFENSIVE"
    assert body["action"] == "ALLOW"
    assert body["label_was_rule_assisted"] is True
    assert body["ensemble_probabilities"] == [0.15, 0.70, 0.15]
    assert body["confidence"] == 0.70
    assert body["final_score"] == 0.85


def test_candidate_corrected_offensive_cannot_be_allowed(monkeypatch):
    install_fake_runtime(monkeypatch)
    monkeypatch.setattr(main, "ENABLE_NEGATION_CANDIDATE", True, raising=False)
    with TestClient(main.app) as client:
        response = client.post("/moderate", json={"text": "Mira is foolish."})
    assert response.status_code == 200
    assert response.json()["predicted_label"] == "OFFENSIVE"
    assert response.json()["action"] == "WARNING"


def test_moderate_disabled_keeps_original_score_threshold_behavior(monkeypatch):
    install_fake_runtime(monkeypatch)
    monkeypatch.setattr(main, "ENABLE_NEGATION_CANDIDATE", False, raising=False)
    with TestClient(main.app) as client:
        response = client.post("/moderate", json={"text": "Mira is not foolish."})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == "OFFENSIVE"
    assert body["action"] == "BLOCK"
    assert "label_was_rule_assisted" not in body
    assert set(body) == {
        "action", "regex_score", "matched_terms", "message",
        "predicted_class", "predicted_label", "confidence", "probabilities",
        "model_predictions", "model_probabilities", "ensemble_probabilities",
        "lite_probabilities", "muril_probabilities", "lite_prediction",
        "muril_prediction", "ensemble_prediction", "final_score", "muril_score",
        "model_version", "class_order",
    }

