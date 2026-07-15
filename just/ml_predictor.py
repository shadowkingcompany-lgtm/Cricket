"""
ml_predictor.py
---------------
Adaptive score predictor for T20 cricket.

• Trains a GradientBoostingRegressor (Ridge fallback for <20 samples).
• Re-trains after EVERY over using the live match state.
• Persists all training samples and completed-match records to
  match_history.json so the model improves with every match seen.

Feature vector (8 dimensions per snapshot):
  [overs_done, total_balls, score, wickets, crr,
   phase_enc (0/1/2), last3_rr, pitch_enc (0-3)]

Target: final innings score (runs).
"""

import json
import os
import time
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "match_history.json")

# ------------------------------------------------------------------
# Seed training data: [overs, balls, score, wkts, crr, phase, l3rr, pitch, final]
# Built from representative T20 match snapshots
# ------------------------------------------------------------------
SEED_SAMPLES = [
    [1,  6,   9, 0,  9.0, 0,  9.0, 0, 171],
    [2, 12,  16, 0,  8.0, 0,  8.1, 0, 158],
    [3, 18,  24, 1,  8.0, 0,  8.3, 0, 151],
    [4, 24,  35, 1,  8.8, 0,  9.5, 0, 172],
    [5, 30,  42, 2,  8.4, 0,  8.2, 0, 155],
    [6, 36,  52, 1,  8.7, 0,  9.0, 0, 168],
    [6, 36,  45, 2,  7.5, 0,  7.2, 1, 142],
    [6, 36,  62, 0, 10.3, 0, 11.2, 0, 196],
    [7, 42,  60, 2,  8.6, 1,  8.8, 0, 162],
    [8, 48,  68, 2,  8.5, 1,  8.0, 0, 161],
    [9, 54,  77, 3,  8.6, 1,  8.7, 2, 153],
    [10, 60,  89, 2,  8.9, 1,  9.1, 0, 175],
    [10, 60,  72, 4,  7.2, 1,  6.8, 2, 138],
    [10, 60,  95, 1,  9.5, 1, 10.5, 0, 191],
    [11, 66, 104, 2,  9.5, 1,  9.8, 0, 184],
    [12, 72, 104, 3,  8.7, 1,  9.2, 1, 172],
    [13, 78, 112, 3,  8.6, 1,  8.4, 0, 169],
    [14, 84, 120, 4,  8.6, 1,  8.9, 3, 162],
    [15, 90, 130, 3,  8.7, 1, 10.2, 0, 182],
    [15, 90, 115, 5,  7.7, 1,  8.5, 1, 152],
    [15, 90, 140, 2,  9.3, 1, 11.8, 0, 201],
    [16, 96, 135, 6,  8.4, 2,  9.0, 2, 158],
    [17, 102, 150, 4,  8.8, 2, 10.8, 0, 183],
    [18, 108, 158, 5,  8.8, 2,  9.5, 1, 178],
    [18, 108, 168, 3,  9.3, 2, 11.0, 0, 195],
    [19, 114, 172, 3,  9.1, 2, 11.2, 0, 193],
    [19, 114, 155, 6,  8.2, 2,  9.2, 2, 169],
    [20, 120, 186, 4,  9.3, 2, 11.5, 0, 186],
    [20, 120, 142, 8,  7.1, 2,  8.2, 2, 142],
    [3,  18,  22, 0,  7.3, 0,  7.5, 0, 148],
    [5,  30,  55, 0, 11.0, 0, 11.5, 0, 198],
    [7,  42,  58, 3,  8.3, 0,  7.8, 2, 140],
    [9,  54,  82, 1,  9.1, 1,  9.5, 3, 179],
    [12, 72, 108, 2,  9.0, 1,  9.8, 1, 180],
    [16, 96, 142, 4,  8.9, 2, 10.1, 0, 187],
    [6,  36,  38, 4,  6.3, 0,  5.8, 2, 118],
    [10, 60,  58, 6,  5.8, 1,  5.0, 1, 107],
    [15, 90, 112, 7,  7.5, 1,  7.8, 3, 131],
    [6,  36,  72, 0, 12.0, 0, 13.2, 0, 214],
    [10, 60, 108, 1, 10.8, 1, 12.0, 0, 207],
]


class CricketPredictor:
    """Adaptive ML model that learns from every over and every match."""

    FEATURE_NAMES = [
        "Overs Done", "Total Balls", "Score", "Wickets",
        "CRR", "Phase", "Last-3 RR", "Pitch Type"
    ]

    PHASE_ENC = {"powerplay": 0, "middle": 1, "death": 2}
    PITCH_ENC = {"flat": 0, "seaming": 1, "turning": 2, "bouncy": 3}

    def __init__(self):
        self.history = self._load_history()
        self.scaler = StandardScaler()
        self.model = None
        self._last_trained_over = -1
        self._fit()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        return {
            "training_samples": [list(s) for s in SEED_SAMPLES],
            "completed_matches": [],
            "total_matches_seen": 0,
        }

    def _save_history(self):
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2)

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def _fit(self):
        samples = self.history["training_samples"]
        if len(samples) < 5:
            return
        X = np.array([s[:8] for s in samples], dtype=float)
        y = np.array([s[8] for s in samples], dtype=float)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        if len(samples) >= 20:
            self.model = GradientBoostingRegressor(
                n_estimators=120, learning_rate=0.08,
                max_depth=4, subsample=0.85, random_state=42,
            )
        else:
            self.model = Ridge(alpha=1.0)
        self.model.fit(Xs, y)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _features(self, state):
        overs_str = str(state.get("overs", "0.0"))
        parts = overs_str.split(".")
        overs_done = int(parts[0])
        balls_in_over = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        total_balls = overs_done * 6 + balls_in_over

        score = float(state.get("score", 0))
        wickets = float(state.get("wickets", 0))
        crr = float(state.get("crr", 0))
        last3_rr = float(state.get("last3_runrate", crr))
        phase_enc = self.PHASE_ENC.get(state.get("phase", "middle"), 1)
        pitch_enc = self.PITCH_ENC.get(state.get("pitch_type", "flat"), 0)

        return [overs_done, total_balls, score, wickets,
                crr, phase_enc, last3_rr, pitch_enc]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, state):
        feats = self._features(state)
        overs_done = feats[0]

        if self.model is None:
            base = max(60, feats[2] / max(overs_done, 1) * 20)
            return {
                "predicted_score": int(base),
                "lower": int(base - 25),
                "upper": int(base + 25),
                "confidence": 0.20,
                "feature_importance": {},
                "model_type": "fallback",
                "model_samples": len(self.history["training_samples"]),
            }

        X = np.array([feats], dtype=float)
        Xs = self.scaler.transform(X)
        pred = float(self.model.predict(Xs)[0])
        pred = max(60.0, min(300.0, pred))

        # Uncertainty shrinks as overs advance
        uncertainty = max(5.0, 28.0 - overs_done * 1.25)
        confidence = round(min(0.97, 0.35 + overs_done * 0.031), 2)

        # Feature importance
        importance = {}
        if hasattr(self.model, "feature_importances_"):
            raw = self.model.feature_importances_
            importance = {
                n: round(float(v), 4)
                for n, v in zip(self.FEATURE_NAMES, raw)
            }
        elif hasattr(self.model, "coef_"):
            raw = np.abs(self.model.coef_)
            total = raw.sum() or 1
            importance = {
                n: round(float(v / total), 4)
                for n, v in zip(self.FEATURE_NAMES, raw)
            }

        model_type = (
            "GradientBoosting" if isinstance(self.model, GradientBoostingRegressor)
            else "Ridge"
        )

        return {
            "predicted_score": int(round(pred)),
            "lower": int(round(max(60, pred - uncertainty))),
            "upper": int(round(min(300, pred + uncertainty))),
            "confidence": confidence,
            "feature_importance": importance,
            "model_type": model_type,
            "model_samples": len(self.history["training_samples"]),
        }

    # ------------------------------------------------------------------
    # Online learning
    # ------------------------------------------------------------------

    def maybe_train_on_over(self, state):
        """Call after each API poll; trains whenever a new over completes."""
        overs_str = str(state.get("overs", "0.0"))
        parts = overs_str.split(".")
        overs_done = int(parts[0])
        balls_in_over = int(parts[1]) if len(parts) > 1 and parts[1] else 0

        over_complete = (balls_in_over == 0 and overs_done > 0
                         and state.get("over_just_completed", False))

        if over_complete and overs_done != self._last_trained_over:
            self._last_trained_over = overs_done
            feats = self._features(state)
            # Use current model prediction as proxy label (will be refined
            # when match completes with the real final score)
            proxy_label = self.predict(state)["predicted_score"]
            sample = feats + [proxy_label]
            self.history["training_samples"].append(sample)
            self._fit()

    def record_completed_match(self, match_record):
        """
        Called when a full innings finishes.
        Adds high-quality (ground-truth) training samples at each completed over.
        """
        ball_log = match_record.get("ball_log", [])
        final_score = match_record.get("final_score", 0)
        pitch_type = match_record.get("pitch_type", "flat")

        if final_score <= 0 or not ball_log:
            return

        pitch_enc = self.PITCH_ENC.get(pitch_type, 0)

        for over_num in range(1, 21):
            balls_up_to = [b for b in ball_log if b.get("over", 0) < over_num]
            if not balls_up_to:
                continue

            score_at = sum(b["runs"] for b in balls_up_to)
            wickets_at = sum(1 for b in balls_up_to if b.get("wicket"))
            total_balls = len(balls_up_to)
            crr = round(score_at / over_num, 2)

            recent = balls_up_to[-18:]
            last3_rr = round(
                sum(b["runs"] for b in recent) / len(recent) * 6, 2
            ) if recent else crr

            phase_enc = 0 if over_num <= 6 else (1 if over_num <= 15 else 2)

            sample = [
                over_num, total_balls, score_at, wickets_at,
                crr, phase_enc, last3_rr, pitch_enc,
                final_score,
            ]
            self.history["training_samples"].append(sample)

        self.history["completed_matches"].append({
            "match_name": match_record.get("match_name", ""),
            "venue": match_record.get("venue", ""),
            "pitch_type": pitch_type,
            "final_score": final_score,
            "timestamp": match_record.get("timestamp", time.time()),
        })
        self.history["total_matches_seen"] = (
            self.history.get("total_matches_seen", 0) + 1
        )

        self._fit()
        self._save_history()

    # ------------------------------------------------------------------
    # Public stats
    # ------------------------------------------------------------------

    def stats(self):
        return {
            "total_samples": len(self.history["training_samples"]),
            "completed_matches": len(self.history["completed_matches"]),
            "total_matches_seen": self.history.get("total_matches_seen", 0),
            "model_type": (
                "GradientBoosting"
                if isinstance(self.model, GradientBoostingRegressor)
                else "Ridge" if self.model else "None"
            ),
        }
