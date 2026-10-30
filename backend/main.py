"""FastAPI service for the trained GuardianNLP bundle."""
from __future__ import annotations
import os
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")
BUNDLE_DIR = Path(os.getenv("GUARDIANNLP_BUNDLE_DIR", os.getenv("INDICGUARD_BUNDLE_DIR", str(BASE_DIR.parent / "guardian_nlp_dual_export"))))
ENABLE_NEGATION_CANDIDATE = os.getenv("ENABLE_NEGATION_CANDIDATE", "false").strip().casefold() in {"1", "true", "yes", "on"}
LABELS = ("NORMAL", "OFFENSIVE", "HATE_SPEECH")
LOGGER = logging.getLogger("guardiannlp")
_model: Any = None
_load_error: str | None = None

class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)

class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _load_error
    try:
        from indicguard_runtime import IndicGuard
        _model = IndicGuard(BUNDLE_DIR)
    except Exception as exc:
        _model = None
        _load_error = str(exc)
        app.state.model_load_error = _load_error
        LOGGER.exception("Could not load trained IndicGuard bundle from %s", BUNDLE_DIR)
    else:
        _load_error = None
        app.state.model_load_error = None
    yield
    _model = None

app = FastAPI(title="GuardianNLP API", version="1.0.0", lifespan=lifespan)
origins = [x.strip() for x in os.getenv("GUARDIANNLP_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

def health_payload():
    return {"status": "ok", "running": True, "ready": _model is not None, "model_version": getattr(_model, "version", None), "device": str(getattr(_model, "device", "unknown")) if _model else None, "detail": None if _model else "Trained model bundle is not ready; check backend logs."}

@app.get("/health")
def health():
    return health_payload()

@app.get("/health/ready")
def readiness():
    if _model is None:
        raise HTTPException(status_code=503, detail=health_payload()["detail"])
    return health_payload()

def require_model():
    if _model is None:
        raise HTTPException(status_code=503, detail="GuardianNLP model bundle is not ready.")
    return _model

def _candidate_label(text: str, raw: Any) -> str | None:
    """Return an internal opt-in label override; abstentions retain the ensemble."""
    if not ENABLE_NEGATION_CANDIDATE:
        return None
    from negation_aware_candidate import apply_scope_candidate
    candidate = apply_scope_candidate(text, raw)
    if not candidate["rule_assistance_applied"]:
        return None
    return candidate["candidate_prediction"]

def _validated_probs(raw: Any, key: str) -> list[float]:
    values = raw.get(key)
    if not isinstance(values, (list, tuple)) or len(values) != len(LABELS):
        raise ValueError(f"Model output {key} must contain exactly three probabilities")
    values = [float(v) for v in values]
    if any(not math.isfinite(v) or v < 0.0 or v > 1.0 for v in values) or abs(sum(values) - 1.0) > 1e-4:
        raise ValueError(f"Model output {key} is not a normalized probability vector")
    return values

def _format_prediction(raw: Any, effective_label: str | None = None,
                       label_was_rule_assisted: bool = False) -> dict[str, Any]:
    ensemble = _validated_probs(raw, "ensemble_probabilities")
    lite = _validated_probs(raw, "lite_probabilities")
    muril = _validated_probs(raw, "muril_probabilities")
    if effective_label is not None and effective_label not in LABELS:
        raise ValueError("Candidate prediction label is invalid")
    original_class = max(range(3), key=ensemble.__getitem__)
    predicted_class = (LABELS.index(effective_label) if effective_label is not None
                       else original_class)
    result = {
        "predicted_class": predicted_class,
        "predicted_label": LABELS[predicted_class],
        "confidence": ensemble[original_class],
        "probabilities": dict(zip(LABELS, ensemble)),
        "model_predictions": {
            "lite": LABELS[max(range(3), key=lite.__getitem__)],
            "muril": LABELS[max(range(3), key=muril.__getitem__)],
        },
        "model_probabilities": {
            "lite": dict(zip(LABELS, lite)),
            "muril": dict(zip(LABELS, muril)),
        },
        "ensemble_probabilities": ensemble,
        "lite_probabilities": lite,
        "muril_probabilities": muril,
        "lite_prediction": LABELS[max(range(3), key=lite.__getitem__)],
        "muril_prediction": LABELS[max(range(3), key=muril.__getitem__)],
        "ensemble_prediction": LABELS[original_class],
        "final_score": ensemble[1] + ensemble[2],
        "muril_score": muril[1] + muril[2],
        "model_version": str(raw.get("model_version", "unknown")),
        "class_order": list(LABELS),
    }
    if label_was_rule_assisted:
        result["label_was_rule_assisted"] = True
    return result

def _predict_response(text: str, model: Any) -> dict[str, Any]:
    raw = model.predict(text)
    original_label = LABELS[max(range(3), key=_validated_probs(raw, "ensemble_probabilities").__getitem__)]
    candidate_label = _candidate_label(text, raw)
    rule_assisted = candidate_label is not None
    return _format_prediction(
        raw,
        effective_label=candidate_label if candidate_label is not None else original_label,
        label_was_rule_assisted=rule_assisted,
    )

@app.post("/predict")
def predict(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be blank")
    try:
        model = require_model()
        result = _predict_response(text, model)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error.") from exc
    return result

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    model = require_model()
    results = []
    for text in request.texts:
        if not text.strip():
            raise HTTPException(status_code=422, detail="texts must not contain blank entries")
        if len(text) > 5000:
            raise HTTPException(status_code=422, detail="each text must be at most 5000 characters")
        try:
            cleaned = text.strip()
            results.append(_predict_response(cleaned, model))
        except Exception as exc:
            LOGGER.exception("Batch prediction failed")
            raise HTTPException(status_code=500, detail="Internal batch prediction error.") from exc
    return {"results": results}

@app.post("/moderate")
def moderate(request: TextRequest):
    """Legacy frontend contract; scores derive only from trained model probabilities."""
    result = predict(request)
    score = result["final_score"]
    action = "ALLOW" if score < .40 else "WARNING" if score < .70 else "CENSOR" if score < .85 else "BLOCK"
    if result.get("label_was_rule_assisted"):
        if result["predicted_label"] == "NORMAL":
            action = "ALLOW"
        elif action == "ALLOW":
            action = "WARNING"
    return {**result, "action": action, "regex_score": 0.0, "matched_terms": [], "message": "Text analyzed by GuardianNLP trained models."}
