"""Portable IndicGuard inference runtime for the exported dual-model bundle."""
from __future__ import annotations
import json, os, re, time
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

LABELS = ["NORMAL", "OFFENSIVE", "HATE_SPEECH"]
MAX_LENGTH = 128
ENSEMBLE_WEIGHTS = {"muril": 0.65, "lite": 0.35}

def clean_social_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()

class MuRILMeanPoolClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_config(config)
        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Linear(config.hidden_size, 3)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids, **kwargs)
        mask = attention_mask.unsqueeze(-1).to(enc.last_hidden_state.dtype)
        pooled = (enc.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return SequenceClassifierOutput(logits=self.classifier(self.dropout(pooled)))

class IndicGuard:
    def __init__(self, bundle_dir: str | os.PathLike, device=None):
        self.root = Path(bundle_dir)
        cfg = json.loads((self.root / "ensemble_config.json").read_text(encoding="utf-8"))
        self.version = cfg.get("model_version", "unknown")
        self.weights = cfg.get("ensemble_weights", ENSEMBLE_WEIGHTS)
        labels_file = self.root / "class_labels.json"
        if not labels_file.is_file() or json.loads(labels_file.read_text(encoding="utf-8")) != {"0":"NORMAL","1":"OFFENSIVE","2":"HATE_SPEECH"}:
            raise ValueError("Bundle class-label mapping must be 0=NORMAL, 1=OFFENSIVE, 2=HATE_SPEECH")
        if self.weights != ENSEMBLE_WEIGHTS:
            raise ValueError(f"Unexpected primary ensemble weights: {self.weights}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        lite_dir = self.root / "model_scratch"
        muril_dir = self.root / "model_finetuned"
        required = [lite_dir / "config.json", muril_dir / "config.json", muril_dir / "model_state.pt"]
        missing = [str(x) for x in required if not x.is_file()]
        if missing:
            raise FileNotFoundError("Trained IndicGuard checkpoint files missing: " + ", ".join(missing))
        self.tokenizer_lite = AutoTokenizer.from_pretrained(lite_dir, local_files_only=True)
        self.tokenizer_muril = AutoTokenizer.from_pretrained(muril_dir, local_files_only=True)
        self.model_lite = AutoModelForSequenceClassification.from_pretrained(lite_dir, local_files_only=True)
        if len(self.tokenizer_lite) != self.model_lite.get_input_embeddings().num_embeddings:
            raise ValueError(f"Lite tokenizer/model vocabulary mismatch: tokenizer={len(self.tokenizer_lite)}, embeddings={self.model_lite.get_input_embeddings().num_embeddings}")
        _id2label = {int(k): v for k, v in self.model_lite.config.id2label.items()}
        if _id2label != {0:"NORMAL",1:"OFFENSIVE",2:"HATE_SPEECH"}:
            raise ValueError(f"Lite checkpoint class order mismatch: {_id2label}")
        if int(self.model_lite.config.num_labels) != 3:
            raise ValueError("Lite checkpoint must have three logits")
        muril_cfg = AutoConfig.from_pretrained(muril_dir, local_files_only=True)
        _muril_labels = {int(k): v for k, v in muril_cfg.id2label.items()}
        if _muril_labels != {0:"NORMAL",1:"OFFENSIVE",2:"HATE_SPEECH"}:
            raise ValueError(f"MuRIL checkpoint class order mismatch: {_muril_labels}")
        if (muril_cfg.num_hidden_layers, muril_cfg.num_attention_heads, muril_cfg.hidden_size) != (12,12,768):
            raise ValueError("MuRIL checkpoint architecture does not match 12x768 base specification")
        self.model_muril = MuRILMeanPoolClassifier(muril_cfg)
        try:
            state = torch.load(muril_dir / "model_state.pt", map_location="cpu", weights_only=True)
        except TypeError:  # compatibility for PyTorch before weights_only was added
            state = torch.load(muril_dir / "model_state.pt", map_location="cpu")
        missing_keys, unexpected_keys = self.model_muril.load_state_dict(state, strict=True)
        if missing_keys or unexpected_keys:
            raise RuntimeError(f"MuRIL state mismatch; missing={missing_keys}, unexpected={unexpected_keys}")
        self.model_lite.to(self.device).eval()
        self.model_muril.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        clean = clean_social_text(text)
        outputs = {}
        for name, tokenizer, model in (("lite", self.tokenizer_lite, self.model_lite),
                                       ("muril", self.tokenizer_muril, self.model_muril)):
            batch = tokenizer(clean, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
                              padding="max_length")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            probs = torch.softmax(model(**batch).logits.float(), dim=-1)[0].cpu().tolist()
            if len(probs) != 3 or not all(torch.isfinite(torch.tensor(probs))):
                raise RuntimeError(f"{name} returned invalid three-class probabilities")
            if abs(sum(probs) - 1.0) > 1e-5:
                raise RuntimeError(f"{name} probabilities do not sum to one")
            outputs[name] = probs
        p_lite, p_muril = outputs["lite"], outputs["muril"]
        p_ens = [self.weights["muril"]*m + self.weights["lite"]*l for m,l in zip(p_muril,p_lite)]
        pred = max(range(3), key=p_ens.__getitem__)
        return {"predicted_class": pred, "predicted_label": LABELS[pred],
                "lite_prediction": LABELS[max(range(3), key=p_lite.__getitem__)],
                "muril_prediction": LABELS[max(range(3), key=p_muril.__getitem__)],
                "ensemble_prediction": LABELS[pred], "lite_probabilities": p_lite,
                "muril_probabilities": p_muril, "ensemble_probabilities": p_ens,
                "confidence": float(p_ens[pred]), "model_version": self.version,
                "final_score": float(p_ens[1] + p_ens[2]),
                "muril_score": float(p_muril[1] + p_muril[2])}
