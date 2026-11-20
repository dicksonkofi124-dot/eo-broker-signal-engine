from __future__ import annotations
import logging, time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("EO_Scraping_Engine")
_lgb = _sklearn_mlp = _StandardScaler = _CalibratedClassifierCV = _TimeSeriesSplit = None

def _ensure_deps():
    global _lgb, _sklearn_mlp, _StandardScaler, _CalibratedClassifierCV, _TimeSeriesSplit
    if _lgb is None:
        import lightgbm as lgb
        _lgb = lgb
    if _sklearn_mlp is None:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        _sklearn_mlp, _StandardScaler = MLPClassifier, StandardScaler
        _CalibratedClassifierCV, _TimeSeriesSplit = CalibratedClassifierCV, TimeSeriesSplit

N_FEATURES = 9
MIN_SAMPLES_TO_TRAIN = 400
RETRAIN_EVERY_SAMPLES = 250
RETRAIN_EVERY_SECONDS = 20 * 60
MIN_AGREEMENT_EDGE = 0.015
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
            logger.info("Calibrated Ensemble retrained (#%s) on %s samples | LGB=%s NN=%s",
                        self.train_count, n, "ok" if self.lgb_model else "fail", "ok" if self.nn_model else "fail")
            return True
        except Exception as exc:
            logger.error("Ensemble retrain failed: %s", exc)
            return False

    def _train(self, samples: list[dict]) -> None:
        import numpy as np
        _ensure_deps()
        X = np.array([s["features"] for s in samples], dtype=np.float64)
        y = np.array([int(s["label"]) for s in samples], dtype=np.int32)
        mask = np.all(np.isfinite(X), axis=1)
        X, y = X[mask], y[mask]
        if len(y) < MIN_SAMPLES_TO_TRAIN:
            raise ValueError("Not enough clean samples")
        sample_weight = np.linspace(0.5, 1.5, len(y))
        self.scaler = _StandardScaler()
        Xs = self.scaler.fit_transform(X)
        tscv = _TimeSeriesSplit(n_splits=3, gap=10)
        try:
            base_lgb = _lgb.LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.04,
                                           subsample=0.80, colsample_bytree=0.80, reg_alpha=0.5, reg_lambda=2.0,
                                           min_child_samples=50, random_state=42, verbosity=-1, n_jobs=1)
            cal_lgb = _CalibratedClassifierCV(base_lgb, method="sigmoid", cv=tscv)
            cal_lgb.fit(X, y, sample_weight=sample_weight)
            self.lgb_model = cal_lgb
        except Exception as exc:
            logger.warning("LightGBM train failed: %s", exc)
            self.lgb_model = None
        try:
            base_nn = _sklearn_mlp(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=0.05,
                                   batch_size=64, learning_rate_init=0.001, max_iter=150, shuffle=False, random_state=42)
            cal_nn = _CalibratedClassifierCV(base_nn, method="sigmoid", cv=tscv)
            cal_nn.fit(Xs, y)
            self.nn_model = cal_nn
        except Exception as exc:
            logger.warning("MLP train failed: %s", exc)
            self.nn_model = None
        if self.lgb_model is None and self.nn_model is None:
            raise RuntimeError("Both models failed to train")

    def predict(self, features: list[float]) -> EnsemblePrediction:
        import numpy as np
        if not self.is_ready:
            return EnsemblePrediction(proba=0.5, edge=0.0, direction=0, detail="not ready")
        x = np.array(features, dtype=np.float64).reshape(1, -1)
        if not np.all(np.isfinite(x)):
            return EnsemblePrediction(proba=0.5, edge=0.0, direction=0, detail="bad features")
        lgb_p = nn_p = None
        if self.lgb_model is not None:
            try:
                lgb_p = float(self.lgb_model.predict_proba(x)[0, 1])
            except Exception:
                pass
        if self.nn_model is not None and self.scaler is not None:
            try:
                nn_p = float(self.nn_model.predict_proba(self.scaler.transform(x))[0, 1])
            except Exception:
                pass
        probs, weights = [], []
        if lgb_p is not None:
            probs.append(lgb_p); weights.append(ENSEMBLE_WEIGHT_LGB)
        if nn_p is not None:
            probs.append(nn_p); weights.append(ENSEMBLE_WEIGHT_NN)
        if not probs:
            return EnsemblePrediction(proba=0.5, edge=0.0, direction=0, detail="both failed")
        proba = sum(p * w for p, w in zip(probs, weights)) / sum(weights)
        edge = proba - 0.5
        agreement = False
        if lgb_p is not None and nn_p is not None:
            agreement = ((lgb_p - 0.5) * (nn_p - 0.5) > 0) and (abs(lgb_p - 0.5) >= MIN_AGREEMENT_EDGE) and (abs(nn_p - 0.5) >= MIN_AGREEMENT_EDGE)
        direction, confidence, abs_edge = 0, "Low", abs(edge)
        if agreement and abs_edge >= 0.035:
            direction = 1 if edge > 0 else -1; confidence = "High"
        elif agreement and abs_edge >= 0.02:
            direction = 1 if edge > 0 else -1; confidence = "Medium"
        elif abs_edge >= 0.04:
            direction = 1 if edge > 0 else -1; confidence = "Low"
        detail = f"lgb={lgb_p:.3f}" if lgb_p is not None else "lgb=n/a"
        detail += f" nn={nn_p:.3f} agree={agreement}" if nn_p is not None else f" nn=n/a agree={agreement}"
        return EnsemblePrediction(proba=proba, edge=edge, direction=direction, lgb_proba=lgb_p, nn_proba=nn_p, agreement=agreement, confidence=confidence, detail=detail)

    def status(self) -> str:
        return f"Ensemble(ready={self.is_ready}, trains={self.train_count}, samples={self.last_train_size}, LGB={'yes' if self.lgb_model else 'no'}, NN={'yes' if self.nn_model else 'no'})"
