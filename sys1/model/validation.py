"""Fixed-validation checkpoint selection and resumable early-stopping state."""
from __future__ import annotations

import math
from numbers import Real


def _loss(value) -> bool:
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


class ValidationGuard:
    def __init__(self, weights: dict[str, float], patience: int = 8, min_delta: float = 0.01,
                 validation_id: str | None = None, max_domain_regression: float | None = None,
                 domain_patience: int = 4):
        if not weights or any(not _loss(w) or w <= 0 for w in weights.values()):
            raise ValueError("validation weights must be finite and positive")
        if type(patience) is not int or patience < 0 or not _loss(min_delta):
            raise ValueError("invalid validation patience or minimum improvement")
        if max_domain_regression is not None and not _loss(max_domain_regression):
            raise ValueError("invalid domain regression limit")
        if type(domain_patience) is not int or domain_patience < 1:
            raise ValueError("domain patience must be a positive integer")
        try:
            total = math.fsum(weights.values())
        except OverflowError as error:
            raise ValueError("validation weights overflow") from error
        self.weights = {key: value / total for key, value in weights.items()}
        self.patience, self.min_delta, self.validation_id = patience, min_delta, validation_id
        self.max_domain_regression, self.domain_patience = max_domain_regression, domain_patience
        self.best_score = self.best_step = self.progress_score = self.last_step = None
        self.best_losses = None
        self.domain_best_losses = {}
        self.domain_bad_evaluations = {key: 0 for key in self.weights}
        self.bad_evaluations = self.evaluations = 0
        self.legacy_score_only = False

    @property
    def enabled(self) -> bool:
        return bool(self.patience or self.max_domain_regression is not None)

    @property
    def stop_reason(self) -> str | None:
        if self.max_domain_regression is not None:
            failed = sorted(key for key, count in self.domain_bad_evaluations.items()
                            if count >= self.domain_patience)
            if failed:
                return "domain_regression:" + ",".join(failed)
        if self.patience and self.bad_evaluations >= self.patience:
            return "validation_patience"
        return None

    @property
    def should_stop(self) -> bool:
        return self.stop_reason is not None

    def policy(self) -> dict:
        return {"weights": dict(self.weights), "patience": self.patience, "min_delta": self.min_delta,
                "validation_id": self.validation_id, "max_domain_regression": self.max_domain_regression,
                "domain_patience": self.domain_patience}

    def observe(self, step: int, losses: dict[str, float]) -> dict:
        if self.should_stop:
            raise ValueError("validation guard already stopped; start a separately authorized run")
        if type(step) is not int or step < 0 or (self.last_step is not None and step <= self.last_step):
            raise ValueError("validation steps must increase strictly")
        if not self.weights.keys() <= losses.keys():
            raise ValueError("validation is missing a training-mixture source")
        if any(not _loss(losses[key]) for key in self.weights):
            raise ValueError("validation losses must be finite and nonnegative")
        current = {key: float(losses[key]) for key in self.weights}
        score = math.fsum(self.weights[key] * current[key] for key in self.weights)
        domain_bests = {key: min(self.domain_best_losses.get(key, value), value)
                        for key, value in current.items()}
        regressions = [] if self.max_domain_regression is None else [
            key for key, value in current.items()
            if value > domain_bests[key] * (1 + self.max_domain_regression) + 1e-12]
        eligible = not regressions
        improved = eligible and (self.best_score is None or score < self.best_score)
        if improved:
            self.best_score, self.best_step, self.best_losses = score, step, dict(current)
            self.legacy_score_only = False
        if eligible and (self.progress_score is None or score < self.progress_score - self.min_delta):
            self.progress_score, self.bad_evaluations = score, 0
        else:
            self.bad_evaluations += 1
        self.domain_best_losses = domain_bests
        self.domain_bad_evaluations = {
            key: self.domain_bad_evaluations[key] + 1 if key in regressions else 0
            for key in self.weights}
        self.evaluations += 1
        self.last_step = step
        return {"weighted_loss": score, "improved": improved, "eligible_for_best": eligible,
                "best_score": self.best_score, "best_step": self.best_step,
                "bad_evaluations": self.bad_evaluations, "should_stop": self.should_stop,
                "stop_reason": self.stop_reason, "domain_regressions": sorted(regressions),
                "domain_best_losses": dict(self.domain_best_losses),
                "domain_bad_evaluations": dict(self.domain_bad_evaluations)}

    def state(self) -> dict:
        return {"version": 2, "policy": self.policy(), "best_score": self.best_score,
                "best_step": self.best_step, "progress_score": self.progress_score,
                "bad_evaluations": self.bad_evaluations, "evaluations": self.evaluations,
                "last_step": self.last_step, "best_losses": dict(self.best_losses) if self.best_losses is not None else None,
                "domain_best_losses": dict(self.domain_best_losses),
                "domain_bad_evaluations": dict(self.domain_bad_evaluations),
                "legacy_score_only": self.legacy_score_only}

    def load_state(self, state: dict, step_hint: int | None = None) -> None:
        if not isinstance(state, dict):
            raise ValueError("malformed checkpoint validation state")
        version, policy = state.get("version"), state.get("policy")
        expected = self.policy()
        if version == 1:
            expected = {key: expected[key] for key in ("weights", "patience", "min_delta", "validation_id")}
            if self.max_domain_regression is not None:
                raise ValueError("resume validation policy differs from the checkpoint")
        if version not in (1, 2) or policy != expected:
            raise ValueError("resume validation policy differs from the checkpoint")
        try:
            candidate = {key: state[key] for key in ("best_score", "best_step", "progress_score",
                                                    "bad_evaluations", "evaluations")}
            for key in ("bad_evaluations", "evaluations"):
                if type(candidate[key]) is not int or candidate[key] < 0:
                    raise ValueError("invalid checkpoint validation counter")
            count = candidate["evaluations"]
            if count:
                if not _loss(candidate["best_score"]) or not _loss(candidate["progress_score"]):
                    raise ValueError("invalid checkpoint validation score")
                if candidate["progress_score"] < candidate["best_score"] or candidate["bad_evaluations"] >= count:
                    raise ValueError("inconsistent checkpoint validation history")
                if type(candidate["best_step"]) is not int or candidate["best_step"] < 0:
                    raise ValueError("invalid checkpoint best step")
            elif any(candidate[key] is not None for key in ("best_score", "best_step", "progress_score")) or candidate["bad_evaluations"]:
                raise ValueError("inconsistent empty validation history")
            if version == 1:
                last = None if not count else step_hint if step_hint is not None else candidate["best_step"] if count == 1 else None
                if count and last is None:
                    raise ValueError("legacy validation history requires the checkpoint step")
                candidate.update(last_step=last, best_losses=None, domain_best_losses={},
                                 domain_bad_evaluations={key: 0 for key in self.weights}, legacy_score_only=bool(count))
            else:
                candidate.update({key: state[key] for key in ("last_step", "best_losses", "domain_best_losses",
                                                               "domain_bad_evaluations", "legacy_score_only")})
            if count and (type(candidate["last_step"]) is not int or candidate["last_step"] < candidate["best_step"]):
                raise ValueError("invalid checkpoint last step")
            if count and count > candidate["last_step"] + 1:
                raise ValueError("validation count exceeds strictly increasing step history")
            if not count and candidate["last_step"] is not None:
                raise ValueError("inconsistent empty validation step")
            if not count and (candidate["best_losses"] is not None or candidate["domain_best_losses"]):
                raise ValueError("inconsistent empty domain history")
            if type(candidate["legacy_score_only"]) is not bool:
                raise ValueError("invalid legacy validation flag")
            for field in ("best_losses", "domain_best_losses"):
                values = candidate[field]
                if values:
                    if values.keys() != self.weights.keys() or any(not _loss(v) for v in values.values()):
                        raise ValueError("invalid checkpoint domain losses")
                elif count and not candidate["legacy_score_only"]:
                    raise ValueError("checkpoint domain history missing")
            counts = candidate["domain_bad_evaluations"]
            if counts.keys() != self.weights.keys() or any(type(v) is not int or v < 0 or (count and v >= count) or (not count and v) for v in counts.values()):
                raise ValueError("invalid checkpoint domain counters")
            if candidate["best_losses"] and not math.isclose(
                candidate["best_score"], math.fsum(self.weights[k] * candidate["best_losses"][k] for k in self.weights),
                rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("checkpoint best score disagrees with domain losses")
            if candidate["legacy_score_only"] and self.max_domain_regression is not None:
                raise ValueError("legacy checkpoint cannot initialize a domain regression guard")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("malformed checkpoint validation state") from error
        for key, value in candidate.items():
            setattr(self, key, dict(value) if isinstance(value, dict) else value)
