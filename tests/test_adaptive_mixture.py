import copy
import math
import random

import pytest

from sys1.model.mixture import AdaptiveMixture


def controller(**kwargs):
    return AdaptiveMixture({"en": 0.5, "code": 0.3, "agent": 0.2}, "fixed-validation",
                           warmup=1, smoothing=1.0, min_observations=2, **kwargs)


def observe(ctl, step, val, train, count=100):
    return ctl.observe(step, dict(zip(ctl.base, val)), dict(zip(ctl.base, train)),
                       {key: count for key in ctl.base})


def test_focuses_on_underlearned_source_without_favoring_large_absolute_losses():
    ctl = controller()
    observe(ctl, 10, [2, 20, 200], [2, 20, 200])
    same_progress = observe(ctl, 20, [1, 10, 100], [1, 10, 100])
    assert same_progress["updated"] is False
    later = observe(ctl, 30, [0.5, 5, 100], [0.5, 5, 100])
    assert later["next_weights"]["agent"] > 0.2
    assert later["next_weights"]["en"] < 0.5


def test_falling_train_with_worsening_validation_cannot_increase_source_weight():
    ctl = controller()
    observe(ctl, 10, [4, 4, 4], [3, 3, 3])
    result = observe(ctl, 20, [3, 3, 5], [2.25, 2.25, 1])
    assert "agent" in result["sources_held_back"]
    assert result["next_weights"]["agent"] < 0.2


def test_bounds_normalization_and_no_starvation_across_many_feedback_updates():
    ctl, rng = controller(), random.Random(99)
    observe(ctl, 0, [4, 4, 4], [4, 4, 4])
    for step in range(1, 201):
        before = dict(ctl.weights)
        result = observe(ctl, step, [10**rng.uniform(-5, 5) for _ in ctl.base],
                         [10**rng.uniform(-5, 5) for _ in ctl.base])
        assert sum(ctl.weights.values()) == pytest.approx(1, abs=1e-12)
        for name, weight in ctl.weights.items():
            assert ctl.base[name] / 2 - 1e-12 <= weight <= ctl.base[name] * 2 + 1e-12
            assert abs(weight - before[name]) <= 0.02 + 1e-12
            if name in result["sources_held_back"]:
                assert weight <= before[name] + 1e-12


def test_insufficient_source_coverage_does_not_calibrate_or_adjust():
    ctl = controller()
    result = observe(ctl, 10, [4, 4, 4], [3, 3, 3], count=1)
    assert result["reason"] == "insufficient_source_observations"
    assert ctl.observations == 0
    assert ctl.weights == ctl.base


def test_roundtrip_continues_identical_decisions_and_export_is_not_aliased():
    ctl = controller()
    observe(ctl, 10, [4, 4, 4], [3, 3, 3])
    observe(ctl, 20, [3, 3, 4], [2, 2, 3])
    restored = controller()
    restored.load_state(ctl.state())
    assert observe(restored, 30, [3, 3, 5], [2, 2, 3]) == observe(ctl, 30, [3, 3, 5], [2, 2, 3])
    state = ctl.state()
    state["weights"]["agent"] = 0
    assert ctl.weights["agent"] > 0


@pytest.mark.parametrize("step", [10, 9, True])
def test_invalid_steps_leave_state_unchanged(step):
    ctl = controller()
    observe(ctl, 10, [4, 4, 4], [3, 3, 3])
    before = ctl.state()
    with pytest.raises(ValueError):
        observe(ctl, step, [4, 4, 4], [3, 3, 3])
    assert ctl.state() == before


@pytest.mark.parametrize("field,value", [("weights", {"en": 0, "code": 1, "agent": 0}),
                                         ("observations", True), ("last_step", -1),
                                         ("validation_trend", {"en": math.nan, "code": 0, "agent": 0})])
def test_malformed_checkpoint_restore_is_atomic(field, value):
    ctl = controller()
    observe(ctl, 10, [4, 4, 4], [3, 3, 3])
    before = ctl.state()
    corrupt = copy.deepcopy(before)
    corrupt[field] = value
    with pytest.raises(ValueError):
        ctl.load_state(corrupt)
    assert ctl.state() == before


def test_resume_rejects_changed_identity_or_policy():
    ctl = controller()
    for changed in (controller(max_change=0.01), AdaptiveMixture(ctl.base, "different")):
        with pytest.raises(ValueError, match="policy differs"):
            changed.load_state(ctl.state())


def test_zero_loss_source_is_valid_and_cannot_destabilize_controller():
    ctl = controller()
    observe(ctl, 10, [0, 4, 4], [0, 3, 3])
    result = observe(ctl, 20, [0, 3, 3], [0, 2, 2])
    assert all(math.isfinite(weight) for weight in result["next_weights"].values())
