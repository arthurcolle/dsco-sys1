import json

from sys1.model.diagnostics import diagnose_training


def history(worsening=False):
    rows = [{"event": "start", "step": 0, "weights": {"en": 0.8, "agent": 0.2},
             "validation_policy": {"validation_id": "fixed-inputs"}}]
    for step in range(1, 11):
        value = 2 + step * 0.05 if worsening else 4 - step * 0.1
        rows += [{"event": "train", "step": step, "t": 100 + step, "loss": 5 - step * 0.1,
                  "private_prompt": "PRIVATE_EXAMPLE_MUST_NOT_LEAVE_NODE"},
                 {"event": "eval", "step": step, "heldout_loss": {"en": value, "agent": value}}]
    return rows


def test_improving_validation_is_not_called_overfitting():
    report = diagnose_training(history(), [8, 9, 10], 111)
    assert report["status"] == "improving"
    assert report["alerts"] == []
    assert report["best_checkpoint_retained"] is True
    assert report["automatic_intervention"] is False
    assert "PRIVATE_EXAMPLE" not in json.dumps(report)


def test_diverging_validation_and_falling_training_loss_are_flagged():
    report = diagnose_training(history(worsening=True), [8, 9, 10], 111)
    assert report["status"] == "possible_overfitting"
    assert "historical_best_checkpoint_not_locally_retained" in report["alerts"]
    assert report["best_available_checkpoint"]["step"] == 8
    assert report["best_validation"]["step"] == 1


def test_stale_metrics_are_not_reported_healthy():
    report = diagnose_training(history(), [10], 10000)
    assert report["status"] == "stalled"
    assert "training_metrics_stale" in report["alerts"]


def test_finished_run_does_not_become_a_stall():
    rows = history() + [{"event": "done", "step": 10}]
    assert diagnose_training(rows, [10], 10000)["status"] == "completed"


def test_changed_validation_identity_prevents_combining_histories():
    rows = history() + [{"event": "resume", "step": 10, "weights": {"en": 0.8, "agent": 0.2},
                         "validation_policy": {"validation_id": "different-inputs"}}]
    report = diagnose_training(rows, [10], 111)
    assert report["status"] == "incomparable_history"
    assert report["alerts"] == ["validation_policy_changed"]


def test_rewind_does_not_keep_metrics_from_abandoned_future():
    rows = history() + [{"event": "resume", "step": 2, "weights": {"en": 0.8, "agent": 0.2},
                         "validation_policy": {"validation_id": "fixed-inputs"}},
                        {"event": "train", "step": 3, "t": 111, "loss": 2},
                        {"event": "eval", "step": 3, "heldout_loss": {"en": 1, "agent": 1}}]
    report = diagnose_training(rows, [2, 3], 112)
    assert report["latest_validation"]["step"] == 3
    assert report["evaluation_count"] == 3
    assert report["history_rewinds"] == 1
    assert "training_history_rewound" in report["alerts"]


def test_invalid_objects_and_nonfinite_losses_are_visible():
    rows = history() + [None, {"event": "eval", "step": 11,
                              "heldout_loss": {"en": float("nan"), "agent": 1}}]
    report = diagnose_training(rows, [10], 111)
    assert report["invalid_metric_records"] == 2
    assert "invalid_metric_records" in report["alerts"]


def test_domain_regression_is_visible_even_if_weighted_score_improves():
    rows = history()
    for row in rows:
        if row["event"] == "eval":
            row["heldout_loss"]["agent"] = 1 + 0.05 * row["step"]
    report = diagnose_training(rows, [8, 9, 10], 111)
    assert report["status"] == "improving"
    assert "historical_domain_regression:agent" in report["alerts"]
