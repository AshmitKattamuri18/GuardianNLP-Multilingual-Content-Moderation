from types import ModuleType
from fastapi.testclient import TestClient
import main

class FakeModel:
    version = "contract-test-only"
    device = "cpu"
    def predict(self, text):
        assert text == "test input"
        return {"predicted_class": 0, "predicted_label": "NORMAL", "lite_prediction": "NORMAL", "muril_prediction": "NORMAL", "ensemble_prediction": "NORMAL", "lite_probabilities": [0.8, 0.1, 0.1], "muril_probabilities": [0.7, 0.2, 0.1], "ensemble_probabilities": [0.735, 0.165, 0.1], "confidence": 0.735, "model_version": self.version, "final_score": 0.265, "muril_score": 0.3}

def install_fake_runtime(monkeypatch, factory):
    runtime = ModuleType("indicguard_runtime")
    runtime.IndicGuard = factory
    monkeypatch.setitem(__import__("sys").modules, "indicguard_runtime", runtime)

def test_not_ready_returns_503(monkeypatch):
    def unavailable(_bundle): raise FileNotFoundError("checkpoint missing")
    install_fake_runtime(monkeypatch, unavailable)
    with TestClient(main.app) as client:
        assert client.get("/health").json()["ready"] is False
        response = client.post("/predict", json={"text": "test input"})
        assert response.status_code == 503
        assert "model bundle is not ready" in response.json()["detail"].lower()
        assert "checkpoint missing" not in response.json()["detail"]

def test_predict_and_legacy_moderate(monkeypatch):
    install_fake_runtime(monkeypatch, lambda _bundle: FakeModel())
    with TestClient(main.app) as client:
        prediction = client.post("/predict", json={"text": "test input"})
        assert prediction.status_code == 200
        assert prediction.json()["ensemble_probabilities"] == [0.735, 0.165, 0.1]
        legacy = client.post("/moderate", json={"text": "test input"})
        assert legacy.status_code == 200
        assert legacy.json()["action"] == "ALLOW"
        assert legacy.json()["regex_score"] == 0.0

def test_predict_returns_named_probabilities(monkeypatch):
    install_fake_runtime(monkeypatch, lambda _bundle: FakeModel())
    with TestClient(main.app) as client:
        response = client.post("/predict", json={"text": "test input"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["probabilities"] == {"NORMAL": 0.735, "OFFENSIVE": 0.165, "HATE_SPEECH": 0.1}
        assert payload["model_predictions"] == {"lite": "NORMAL", "muril": "NORMAL"}

def test_prediction_failure_does_not_expose_exception(monkeypatch):
    class BrokenModel:
        def predict(self, _text): raise RuntimeError("C:/private/server/path")
    install_fake_runtime(monkeypatch, lambda _bundle: BrokenModel())
    with TestClient(main.app) as client:
        response = client.post("/predict", json={"text": "test input"})
        assert response.status_code == 500
        assert "C:/private/server/path" not in response.text
