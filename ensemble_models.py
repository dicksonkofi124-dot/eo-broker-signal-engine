"""
Dual-model ensemble for EO Broker Signal Engine
================================================
- LightGBM (gradient-boosted trees)
- Small Neural Network (sklearn MLPClassifier)

Both models train on the accumulated learning_samples.
Predictions are combined for more robust signals.

Designed to run on modest hardware (including older Macs).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("EO_Scraping_Engine")

# Optional heavy deps – imported lazily
_lgb = None
_sklearn_mlp = None
_StandardScaler = None


def _ensure_deps():
    global _lgb, _sklearn_mlp, _StandardScaler
    if _lgb is None:
        import lightgbm as lgb
        _lgb = lgb
    if _sklearn_mlp is None:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        _sklearn_mlp = MLPClassifier
        _StandardScaler = StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FEATURES = 9

MIN_SAMPLES_TO_TRAIN = 400
RETRAIN_EVERY_SAMPLES = 250
RETRAIN_EVERY_SECONDS = 20 * 60  # 20 minutes

MIN_AGREEMENT_EDGE = 0.04
ENSEMBLE_WEIGHT_LGB = 0.60
ENSEMBLE_WEIGHT_NN = 0.40


@dataclass
class EnsemblePrediction:
    proba: float
    edge: float
    direction: int
    lgb_proba: Optional[float] = None
    nn_proba: Optional[float] = None
    agreement: bool = False
    confidence: str = "Low"
    detail: str = ""


class DualModelEnsemble:
    """
    Maintains a LightGBM model and a small MLP.
    Call `maybe_retrain(samples)` periodically.
    Call `predict(features)` for a combined signal.
    """

    def __init__(self) -> None:
        self.lgb_model = None
        self.nn_model = None
        self.scaler = None
        self.last_train_size = 0
        self.last_train_time = 0.0
        self.is_ready = False
        self.train_count = 0

    def maybe_retrain(self, samples: list[dict]) -> bool:
        n = len(samples)
        if n < MIN_SAMPLES_TO_TRAIN:
            return False

        enough_new = (n - self.last_train_size) >= RETRAIN_EVERY_SAMPLES
        enough_time = (time.time() - self.last_train_time) >= RETRAIN_EVERY_SECONDS

        if not (enough_new or enough_time) and self.is_ready:
            return False

        try:
            self._train(samples)
            self.last_train_size = n
            self.last_train_time = time.time()
            self.train_count += 1
            self.is_ready = True
            logger.info(
                "Ensemble retrained (#%s) on %s samples | LGB=%s NN=%s",
                self.train_count, n,
                "ok" if self.lgb_model else "fail",
                "ok" if self.nn_model else "fail",
            )
            return True
        except Exception as exc:
            logger.error("Ensemble retrain failed: %s", exc)
            return False

    def _train(self, samples: list[dict]) -> None:
        _ensure_deps()

        X = np.array([s["features"] for s in samples], dtype=np.float64)
        y = np.array([int(s["label"]) for s in samples], dtype=np.int32)

        mask = np.all(np.isfinite(X), axis=1)
        X, y = X[mask], y[mask]
        if len(y) < MIN_SAMPLES_TO_TRAIN:
            raise ValueError("Not enough clean samples")

        self.scaler = _StandardScaler()
        Xs = self.scaler.fit_transform(X)

        # LightGBM
        try:
            self.lgb_model = _lgb.LGBMClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                min_child_samples=40,
                random_state=42,
                verbosity=-1,
                n_jobs=1,
            )
            self.lgb_model.fit(X, y)
        except Exception as exc:
            logger.warning("LightGBM train failed: %s", exc)
            self.lgb_model = None

        # Small Neural Net
        try:
            self.nn_model = _sklearn_mlp(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=0.01,
                batch_size=64,
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=42,
            )
            self.nn_model.fit(Xs, y)
        except Exception as exc:
            logger.warning("MLP train failed: %s", exc)
            self.nn_model = None

        if self.lgb_model is None and self.nn_model is None:
            raise RuntimeError("Both models failed to train")

    def predict(self, features: list[float]) -> EnsemblePrediction:
        if not self.is_ready:
            return EnsemblePrediction(proba=0.5, edge=0.0, direction=0, detail="not ready")

        x = np.array(features, dtype=np.float64).reshape(1, -1)
        if not np.all(np.isfinite(x)):
            return EnsemblePrediction(proba=0.5, edge=0.0, direction=0, detail="bad features")

        lgb_p = None
        nn_p = None

        if self.lgb_model is not None:
            try:
                lgb_p = float(self.lgb_model.predict_proba(x)[0, 1])
            except Exception:
                lgb_p = None

        if self.nn_model is not None and self.scaler is not None:
            try:
                xs = self.scaler.transform(x)
                nn_p = float(self.nn_model.predict_proba(xs)[0, 1])
            except Exception:
                nn_p = None

        probs = []
        weights = []
        if lgb_p is not None:
            probs.append(lgb_p)
            weights.append(ENSEMBLE_WEIGHT_LGB)
        if nn_p is not None:
            probs.append(nn_p)
            weights.append(ENSEMBLE_WEIGHT_NN)

        if not probs:
            return EnsemblePrediction(proba=0.5, edge=0.0, direction=0, detail="both failed")

        wsum = sum(weights)
        proba = sum(p * w for p, w in zip(probs, weights)) / wsum
        edge = proba - 0.5

        agreement = False
        if lgb_p is not None and nn_p is not None:
            same_side = (lgb_p - 0.5) * (nn_p - 0.5) > 0
            both_strong = abs(lgb_p - 0.5) >= MIN_AGREEMENT_EDGE and abs(nn_p - 0.5) >= MIN_AGREEMENT_EDGE
            agreement = same_side and both_strong

        direction = 0
        confidence = "Low"
        abs_edge = abs(edge)

        if abs_edge >= 0.12 and agreement:
            direction = 1 if edge > 0 else -1
            confidence = "High"
        elif abs_edge >= 0.07 and (agreement or abs_edge >= 0.10):
            direction = 1 if edge > 0 else -1
            confidence = "Medium"
        elif abs_edge >= 0.05:
            direction = 1 if edge > 0 else -1
            confidence = "Low"

        lgb_detail = f"{lgb_p:.3f}" if lgb_p is not None else "n/a"
        nn_detail = f"{nn_p:.3f}" if nn_p is not None else "n/a"
        detail = f"lgb={lgb_detail} nn={nn_detail} agree={agreement}"

        return EnsemblePrediction(
            proba=proba,
            edge=edge,
            direction=direction,
            lgb_proba=lgb_p,
            nn_proba=nn_p,
            agreement=agreement,
            confidence=confidence,
            detail=detail,
        )

    def status(self) -> str:
        return (
            f"Ensemble(ready={self.is_ready}, trains={self.train_count}, "
            f"samples={self.last_train_size}, "
            f"LGB={'yes' if self.lgb_model else 'no'}, "
            f"NN={'yes' if self.nn_model else 'no'})"
        )
