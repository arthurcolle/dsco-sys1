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
