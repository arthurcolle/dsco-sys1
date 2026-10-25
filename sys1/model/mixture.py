"""Bounded feedback control of training probabilities; evaluation weights stay fixed.

This is a declared heuristic, not a claim of optimal data selection. It uses
selection validation only, never a final test set or examples from validation.
"""
from __future__ import annotations

import copy
import math


def _nonnegative(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _positive(value):
    return _nonnegative(value) and value > 0


class AdaptiveMixture:
    def __init__(self, weights, validation_id, *, warmup=3, smoothing=0.3,
                 rate=1.0, max_factor=2.0, max_change=0.02, min_observations=32):
        if not isinstance(weights, dict) or not weights or any(not _positive(v) for v in weights.values()):
            raise ValueError("mixture weights must be finite and positive")
        try:
            total = math.fsum(weights.values())
        except OverflowError as error:
            raise ValueError("mixture weights overflow") from error
        if not _positive(total) or not isinstance(validation_id, str) or not validation_id:
            raise ValueError("adaptive mixture requires a fixed validation identity")
        if type(warmup) is not int or warmup < 1 or type(min_observations) is not int or min_observations < 1:
            raise ValueError("mixture warmup and observation counts must be positive integers")
        if not _positive(smoothing) or smoothing > 1 or not _positive(rate) or rate > 10:
            raise ValueError("invalid mixture smoothing or rate")
        if not _positive(max_factor) or not 1 <= max_factor <= 10 or not _positive(max_change) or max_change > 0.1:
            raise ValueError("invalid mixture probability bounds")
        self.base = {key: value / total for key, value in weights.items()}
        if any(not _positive(value) for value in self.base.values()):
            raise ValueError("mixture normalization underflow")
        self.weights = dict(self.base)
        self.validation_id = validation_id
        self.warmup, self.smoothing, self.rate = warmup, smoothing, rate
        self.max_factor, self.max_change, self.min_observations = max_factor, max_change, min_observations
        self.last_step, self.observations = None, 0
        self.baseline_validation = {key: 0.0 for key in self.base}
        self.baseline_training = dict(self.baseline_validation)
        self.validation_trend = dict(self.baseline_validation)
        self.training_trend = dict(self.baseline_validation)

    def policy(self):
        return {"version": 1, "base_weights": dict(self.base), "validation_id": self.validation_id,
                "warmup": self.warmup, "smoothing": self.smoothing, "rate": self.rate,
                "max_factor": self.max_factor, "max_change": self.max_change,
                "min_observations": self.min_observations}

    def observe(self, step, validation, training, counts):
        if type(step) is not int or step < 0 or (self.last_step is not None and step <= self.last_step):
            raise ValueError("adaptive mixture steps must increase strictly")
        if any(set(values) != set(self.base) for values in (validation, training, counts)):
            raise ValueError("adaptive mixture requires every source")
        if any(not _nonnegative(validation[k]) or type(counts[k]) is not int or counts[k] < 0 or
               (counts[k] and not _nonnegative(training[k])) for k in self.base):
            raise ValueError("invalid adaptive mixture observations")
        before = dict(self.weights)
        self.last_step = step
        result = {"previous_weights": before, "next_weights": dict(before), "updated": False,
                  "training_counts": dict(counts)}
        if any(counts[k] < self.min_observations for k in self.base):
            return {**result, "reason": "insufficient_source_observations"}
        self.observations += 1
        if self.observations <= self.warmup:
            for key in self.base:
                # Running means avoid a warmup sum overflowing on finite inputs.
                n = self.observations
                self.baseline_validation[key] = self.baseline_validation[key] * ((n - 1) / n) + validation[key] / n
                self.baseline_training[key] = self.baseline_training[key] * ((n - 1) / n) + training[key] / n
            return {**result, "reason": "calibrating", "calibration_evaluations": self.observations}
        scores, gaps = {}, {}
        for key in self.base:
            val = math.log(max(validation[key], 1e-12)) - math.log(max(self.baseline_validation[key], 1e-12))
            train = math.log(max(training[key], 1e-12)) - math.log(max(self.baseline_training[key], 1e-12))
            a = self.smoothing
            self.validation_trend[key] = (1-a) * self.validation_trend[key] + a * max(-4, min(4, val))
            self.training_trend[key] = (1-a) * self.training_trend[key] + a * max(-4, min(4, train))
            gaps[key] = max(0.0, self.validation_trend[key] - self.training_trend[key])
            scores[key] = self.validation_trend[key] - 2 * gaps[key]
        held_back = [key for key, gap in gaps.items() if gap > math.log(1.05)]
        raw = {key: self.base[key] * math.exp(max(-40, min(40, self.rate * scores[key]))) for key in self.base}
        lower = {key: max(self.base[key] / self.max_factor, before[key] - self.max_change) for key in self.base}
        upper = {key: min(1.0, self.base[key] * self.max_factor, before[key] + self.max_change,
                          before[key] if key in held_back else 1.0) for key in self.base}
        # Scale and clip onto the bounded probability simplex. The previous
        # vector is feasible, even when every source is held back.
        lo, hi = 0.0, 1.0
        def project(scale):
            return {key: max(lower[key], min(upper[key], scale * raw[key])) for key in self.base}
        while math.fsum(project(hi).values()) < 1 - 1e-14:
            hi *= 2
        for _ in range(80):
            mid = (lo + hi) / 2
            if math.fsum(project(mid).values()) < 1:
                lo = mid
            else:
                hi = mid
        self.weights = project((lo + hi) / 2)
        return {**result, "next_weights": dict(self.weights),
                "updated": any(abs(self.weights[k] - before[k]) > 1e-12 for k in self.base),
                "reason": "validation_feedback", "relative_validation_log_trend": dict(self.validation_trend),
                "relative_training_log_trend": dict(self.training_trend),
                "generalization_gap_signal": gaps, "sources_held_back": sorted(held_back)}

    def state(self):
        return copy.deepcopy({"policy": self.policy(), "weights": self.weights, "last_step": self.last_step,
                              "observations": self.observations, "baseline_validation": self.baseline_validation,
                              "baseline_training": self.baseline_training, "validation_trend": self.validation_trend,
                              "training_trend": self.training_trend})

    def load_state(self, state):
        if not isinstance(state, dict) or state.get("policy") != self.policy():
            raise ValueError("adaptive mixture policy differs from checkpoint")
        try:
            candidate = copy.deepcopy({key: state[key] for key in self.state() if key != "policy"})
            count, step = candidate["observations"], candidate["last_step"]
            if type(count) is not int or count < 0 or (step is not None and (type(step) is not int or step < 0)):
                raise ValueError("invalid adaptive mixture counters")
            if count and (step is None or count > step + 1):
                raise ValueError("inconsistent adaptive mixture history")
            for field in ("weights", "baseline_validation", "baseline_training", "validation_trend", "training_trend"):
                values = candidate[field]
                if not isinstance(values, dict) or set(values) != set(self.base):
                    raise ValueError("invalid adaptive mixture source state")
                for key, value in values.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("nonfinite adaptive mixture state")
                    if field == "weights" and not self.base[key] / self.max_factor - 1e-12 <= value <= min(1, self.base[key] * self.max_factor) + 1e-12:
                        raise ValueError("checkpoint mixture outside probability bounds")
                    if field.startswith("baseline") and (value < 0 if count else value != 0):
                        raise ValueError("invalid adaptive mixture baseline")
                    if field.endswith("trend") and (abs(value) > 4 or (count <= self.warmup and value != 0)):
                        raise ValueError("invalid adaptive mixture trend")
            if not math.isclose(math.fsum(candidate["weights"].values()), 1.0, abs_tol=1e-12):
                raise ValueError("checkpoint mixture does not sum to one")
            if count <= self.warmup and candidate["weights"] != self.base:
                raise ValueError("uncalibrated adaptive mixture changed weights")
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("malformed adaptive mixture checkpoint") from error
        for key, value in candidate.items():
            setattr(self, key, value)
