import copy
import math

import pytest

from sys1.model.validation import ValidationGuard


def test_weighted_selection_uses_validation_not_training_loss():
    guard = ValidationGuard({"en": 0.5, "code": 0.3, "agent": 0.2})
    first = guard.observe(200, {"en": 2.0, "code": 3.0, "agent": 4.0})
    assert first["weighted_loss"] == pytest.approx(2.7)
    worse = guard.observe(400, {"en": 2.1, "code": 3.1, "agent": 4.1})
    assert worse["improved"] is False
    assert worse["best_step"] == 200


def test_patience_stops_repeated_regression_and_survives_resume():
    guard = ValidationGuard({"en": 1}, patience=3, min_delta=0.01)
    guard.observe(1, {"en": 2})
    guard.observe(2, {"en": 2.1})
    restored = ValidationGuard({"en": 1}, patience=3, min_delta=0.01)
    restored.load_state(guard.state())
    assert restored.observe(3, {"en": 2.2})["should_stop"] is False
    result = restored.observe(4, {"en": 2.3})
    assert result["should_stop"] is True
    assert result["best_step"] == 1
    assert result["best_score"] == 2


def test_small_improvements_are_retained_and_accumulate_for_patience():
    guard = ValidationGuard({"en": 1}, patience=3, min_delta=0.01)
    guard.observe(1, {"en": 3})
    result = guard.observe(2, {"en": 2.996})
    assert result["improved"] is True
    assert result["bad_evaluations"] == 1
    result = guard.observe(3, {"en": 2.987})
    assert result["bad_evaluations"] == 0
    assert result["best_step"] == 3


@pytest.mark.parametrize("losses", [{}, {"en": math.nan}, {"en": math.inf}, {"en": -1}])
def test_invalid_validation_cannot_replace_best(losses):
    guard = ValidationGuard({"en": 1})
    guard.observe(1, {"en": 2})
    before = copy.deepcopy(guard.state())
    with pytest.raises(ValueError):
        guard.observe(2, losses)
    assert guard.state() == before


def test_resume_cannot_silently_change_stopping_policy():
    guard = ValidationGuard({"en": 1}, patience=3)
    guard.observe(1, {"en": 2})
    with pytest.raises(ValueError, match="policy differs"):
        ValidationGuard({"en": 1}, patience=4).load_state(guard.state())


def test_resume_rejects_a_different_validation_set():
    guard = ValidationGuard({"en": 1}, validation_id="original-heldout")
    guard.observe(1, {"en": 2})
    with pytest.raises(ValueError, match="policy differs"):
        ValidationGuard({"en": 1}, validation_id="changed-heldout").load_state(guard.state())


def test_weighted_improvement_cannot_hide_persistent_domain_regression():
    guard = ValidationGuard({"en": 0.9, "agent": 0.1}, patience=8,
                            max_domain_regression=0.15, domain_patience=2)
    initial = guard.observe(1, {"en": 4, "agent": 1})
    masked = guard.observe(2, {"en": 3, "agent": 1.2})
    assert masked["weighted_loss"] < initial["weighted_loss"]
    assert masked["eligible_for_best"] is False
    assert masked["best_step"] == 1
    stopped = guard.observe(3, {"en": 2, "agent": 1.3})
    assert stopped["should_stop"] is True
    assert stopped["stop_reason"] == "domain_regression:agent"


def test_domain_regression_patience_resets_on_recovery():
    guard = ValidationGuard({"en": 0.9, "agent": 0.1}, max_domain_regression=0.15, domain_patience=2)
    guard.observe(1, {"en": 4, "agent": 1})
    guard.observe(2, {"en": 3, "agent": 1.2})
    recovered = guard.observe(3, {"en": 2.5, "agent": 1.1})
    assert recovered["domain_bad_evaluations"]["agent"] == 0
    assert guard.observe(4, {"en": 2, "agent": 1.3})["should_stop"] is False


@pytest.mark.parametrize("step", [9, 10, True])
def test_duplicate_or_reversed_steps_do_not_consume_patience(step):
    guard = ValidationGuard({"en": 1})
    guard.observe(10, {"en": 2})
    before = guard.state()
    with pytest.raises(ValueError, match="increase strictly"):
        guard.observe(step, {"en": 3})
    assert guard.state() == before


def test_stopping_is_terminal_even_if_a_later_score_would_improve():
    guard = ValidationGuard({"en": 1}, patience=1)
    guard.observe(1, {"en": 2})
    guard.observe(2, {"en": 3})
    restored = ValidationGuard({"en": 1}, patience=1)
    restored.load_state(guard.state())
    with pytest.raises(ValueError, match="already stopped"):
        restored.observe(3, {"en": 1})


@pytest.mark.parametrize("field,value", [
    ("evaluations", True), ("evaluations", 100), ("bad_evaluations", 100), ("last_step", 0),
    ("best_step", "one"), ("progress_score", 0.1),
    ("best_losses", {"en": 9}), ("domain_best_losses", {"en": math.nan}),
    ("domain_bad_evaluations", {"en": -1}),
])
def test_corrupt_checkpoint_restore_is_atomic(field, value):
    guard = ValidationGuard({"en": 1})
    guard.observe(1, {"en": 2})
    guard.observe(2, {"en": 3})
    before = guard.state()
    corrupt = copy.deepcopy(before)
    corrupt[field] = value
    with pytest.raises(ValueError):
        guard.load_state(corrupt)
    assert guard.state() == before


def test_exported_state_does_not_alias_live_guard():
    guard = ValidationGuard({"en": 1})
    guard.observe(1, {"en": 2})
    state = guard.state()
    state["policy"]["weights"]["en"] = 0
    state["best_losses"]["en"] = 100
    assert guard.weights == {"en": 1}
    assert guard.best_losses == {"en": 2}


def test_v1_migration_preserves_patience_with_checkpoint_step():
    guard = ValidationGuard({"en": 1}, patience=3)
    guard.observe(1, {"en": 2})
    guard.observe(2, {"en": 3})
    current = guard.state()
    legacy = {key: current[key] for key in ("best_score", "best_step", "progress_score", "evaluations", "bad_evaluations")}
    legacy.update(version=1, policy={key: current["policy"][key] for key in ("weights", "patience", "min_delta", "validation_id")})
    restored = ValidationGuard({"en": 1}, patience=3)
    with pytest.raises(ValueError, match="requires the checkpoint step"):
        restored.load_state(legacy)
    restored.load_state(legacy, step_hint=100)
    assert restored.observe(101, {"en": 3.1})["bad_evaluations"] == 2


def test_v1_checkpoint_before_first_validation_can_resume():
    guard = ValidationGuard({"en": 1})
    legacy = {key: guard.state()[key] for key in ("best_score", "best_step", "progress_score", "evaluations", "bad_evaluations")}
    legacy.update(version=1, policy={key: guard.policy()[key] for key in ("weights", "patience", "min_delta", "validation_id")})
    guard.load_state(legacy, step_hint=100)
    assert guard.last_step is None
    assert guard.observe(200, {"en": 2})["improved"] is True
