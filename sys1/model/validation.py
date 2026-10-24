"""Fixed-validation checkpoint selection and resumable early-stopping state."""
from __future__ import annotations

import math


class ValidationGuard:
    def __init__(self, weights: dict[str, float], patience: int = 8, min_delta: float = 0.01,
                 validation_id: str | None = None):
        if not weights or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
            raise ValueError("validation weights must be finite and positive")
        if patience < 0 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("invalid validation patience or minimum improvement")
        total = math.fsum(weights.values())
        self.weights = {key: value / total for key, value in weights.items()}
        self.patience, self.min_delta = patience, min_delta
        self.validation_id = validation_id
        self.best_score = self.best_step = self.progress_score = None
        self.bad_evaluations = self.evaluations = 0

    def policy(self) -> dict:
        return {"weights": self.weights, "patience": self.patience, "min_delta": self.min_delta,
                "validation_id": self.validation_id}

    def observe(self, step: int, losses: dict[str, float]) -> dict:
        if not self.weights.keys() <= losses.keys():
            raise ValueError("validation is missing a training-mixture source")
        if any(not math.isfinite(losses[key]) or losses[key] < 0 for key in self.weights):
            raise ValueError("validation losses must be finite and nonnegative")
        score = math.fsum(self.weights[key] * losses[key] for key in self.weights)
        improved = self.best_score is None or score < self.best_score
        if improved:
            self.best_score, self.best_step = score, step
        # Retain every genuine best. Patience resets on cumulative meaningful
        # improvement, so several small gains can jointly exceed min_delta.
        if self.progress_score is None or score < self.progress_score - self.min_delta:
            self.progress_score, self.bad_evaluations = score, 0
        else:
            self.bad_evaluations += 1
        self.evaluations += 1
        return {"weighted_loss": score, "improved": improved, "best_score": self.best_score,
                "best_step": self.best_step, "bad_evaluations": self.bad_evaluations,
                "should_stop": bool(self.patience and self.bad_evaluations >= self.patience)}

    def state(self) -> dict:
        return {"version": 1, "policy": self.policy(), "best_score": self.best_score,
                "best_step": self.best_step, "progress_score": self.progress_score,
                "bad_evaluations": self.bad_evaluations, "evaluations": self.evaluations}

    def load_state(self, state: dict) -> None:
        if state.get("version") != 1 or state.get("policy") != self.policy():
            raise ValueError("resume validation policy differs from the checkpoint")
        for key in ("best_score", "progress_score"):
            value = state[key]
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("invalid checkpoint validation score")
        for key in ("bad_evaluations", "evaluations"):
            if not isinstance(state[key], int) or state[key] < 0:
                raise ValueError("invalid checkpoint validation counter")
        for key in ("best_score", "best_step", "progress_score", "bad_evaluations", "evaluations"):
            setattr(self, key, state[key])
