"""Pure, aggregate-only training diagnostics; never controls a trainer."""


def diagnose_training(rows, retained_steps, now, window=5, stale_seconds=1800):
    import math
    import statistics

    def finite(value):
        try:
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        except OverflowError:
            return False

    invalid_objects = sum(not isinstance(row, dict) for row in rows)
    rows = [row for row in rows if isinstance(row, dict)]
    starts = [r for r in rows if r.get("event") in {"start", "resume"} and r.get("weights")]
    report = {"status": "insufficient_data", "alerts": [], "automatic_intervention": False,
              "validation_identity_verified": False}
    if not starts:
        report["alerts"] = ["training_weights_unknown"]
        return report
    weights = starts[0]["weights"]
    if not isinstance(weights, dict) or not weights or any(not finite(v) or v <= 0 for v in weights.values()):
        report.update(status="invalid_metrics", alerts=["invalid_training_weights"])
        return report
    total = sum(weights.values())
    if not finite(total):
        report.update(status="invalid_metrics", alerts=["invalid_training_weights"])
        return report
    weights = {key: value / total for key, value in weights.items()}
    identities = set()
    for row in starts:
        policy = row.get("validation_policy", {})
        if not isinstance(policy, dict) or not isinstance(policy.get("validation_id"), (str, type(None))):
            report.update(status="invalid_metrics", alerts=["invalid_validation_identity"])
            return report
        identities.add(policy.get("validation_id"))
    if any(r["weights"] != starts[0]["weights"] for r in starts) or len(identities) > 1:
        report.update(status="incomparable_history", alerts=["validation_policy_changed"])
        return report
    report["validation_identity_verified"] = None not in identities
    if not report["validation_identity_verified"]:
        report["alerts"].append("legacy_validation_identity_unrecorded")
    evaluations, training, invalid, rewinds = {}, [], invalid_objects, 0
    for row in rows:
        if row.get("event") in {"start", "resume"} and type(row.get("step")) is int:
            if any(step > row["step"] for step in evaluations) or any(r["step"] > row["step"] for r in training):
                rewinds += 1
            evaluations = {step: item for step, item in evaluations.items() if step <= row["step"]}
            training = [item for item in training if item["step"] <= row["step"]]
        if row.get("event") == "eval":
            losses, step = row.get("heldout_loss"), row.get("step")
            if not isinstance(losses, dict) or type(step) is not int or any(not finite(losses.get(k)) or losses[k] < 0 for k in weights):
                invalid += 1
                continue
            clean = {key: float(losses[key]) for key in weights}
            evaluations[step] = {"step": step, "weighted_loss": sum(weights[k] * clean[k] for k in weights),
                                 "source_losses": clean}
        elif row.get("event") == "train":
            if type(row.get("step")) is int and finite(row.get("loss")) and row["loss"] >= 0 and finite(row.get("t")):
                training.append({key: row[key] for key in ("step", "loss", "t")})
            else:
                invalid += 1
    if invalid:
        report["alerts"].append("invalid_metric_records")
    if rewinds:
        report["alerts"].append("training_history_rewound")
    report["invalid_metric_records"] = invalid
    report["history_rewinds"] = rewinds
    ev = sorted(evaluations.values(), key=lambda row: row["step"])
    report["evaluation_count"] = len(ev)
    if not ev or not training:
        return report
    latest_train = training[-1]
    best = min(ev, key=lambda row: row["weighted_loss"])
    available = [row for row in ev if row["step"] in retained_steps]
    report.update(weights=weights, latest_training=latest_train, latest_validation=ev[-1],
                  best_validation=best, best_checkpoint_retained=best["step"] in retained_steps,
                  best_available_checkpoint=min(available, key=lambda row: row["weighted_loss"]) if available else None,
                  historical_source_best={key: min(row["source_losses"][key] for row in ev) for key in weights})
    if best["step"] not in retained_steps:
        report["alerts"].append("historical_best_checkpoint_not_locally_retained")
    report["historical_source_regression"] = [
        key for key in weights
        if ev[-1]["source_losses"][key] > report["historical_source_best"][key] * 1.15 + 1e-12]
    report["alerts"].extend("historical_domain_regression:" + key for key in report["historical_source_regression"])
    finished = next((r for r in reversed(rows) if r.get("event") in {"done", "early_stopped"}), None)
    last_start = max((i for i, r in enumerate(rows) if r.get("event") in {"start", "resume"}), default=-1)
    last_finish = max((i for i, r in enumerate(rows) if r.get("event") in {"done", "early_stopped"}), default=-1)
    if finished and last_finish > last_start:
        report["status"] = "completed" if finished["event"] == "done" else "early_stopped"
        return report
    age = max(0, now - latest_train["t"])
    report["latest_training_age_seconds"] = round(age, 1)
    if age > stale_seconds:
        report.update(status="stalled", alerts=report["alerts"] + ["training_metrics_stale"])
        return report
    if len(ev) < 2 * window:
        return report

    def block(start, end):
        subset = ev[start:end]
        lower = ev[start - 1]["step"] if start else 0
        losses = [r["loss"] for r in training if lower < r["step"] <= subset[-1]["step"]]
        return {"first_step": subset[0]["step"], "last_step": subset[-1]["step"],
                "validation_mean": statistics.mean(r["weighted_loss"] for r in subset),
                "training_mean": statistics.mean(losses) if losses else None,
                "source_means": {key: statistics.mean(r["source_losses"][key] for r in subset) for key in weights}}

    previous, recent = block(len(ev) - 2 * window, len(ev) - window), block(len(ev) - window, len(ev))
    report.update(previous_window=previous, recent_window=recent)
    delta = (recent["validation_mean"] - previous["validation_mean"]) / max(previous["validation_mean"], 1e-12)
    report["recent_relative_validation_change"] = delta
    falling_train = recent["training_mean"] is not None and previous["training_mean"] is not None and recent["training_mean"] < previous["training_mean"]
    if delta > 0.01:
        report["status"] = "possible_overfitting" if falling_train else "validation_regression"
        report["alerts"].append(report["status"])
    else:
        report["status"] = "improving" if delta < -0.01 else "stable"
    report["alerts"] = sorted(set(report["alerts"]))
    return report
