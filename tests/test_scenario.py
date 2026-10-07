from __future__ import annotations

import numpy as np
import pytest

from sourcesense.config import load_config
from sourcesense.scenario import create_unit_scenario


@pytest.fixture
def config():
    return load_config("config/default.yaml")


def test_deterministic_same_seed(config):
    scenario_a = create_unit_scenario(config, "soft_municipal", 0, 0, fault=False, seed=42)
    scenario_b = create_unit_scenario(config, "soft_municipal", 0, 0, fault=False, seed=42)
    assert np.array_equal(scenario_a.usage_raw, scenario_b.usage_raw)
    assert np.array_equal(scenario_a.absence_mask, scenario_b.absence_mask)
    assert np.array_equal(scenario_a.user_delay, scenario_b.user_delay)


def test_different_seed_changes_output(config):
    scenario_a = create_unit_scenario(config, "hard_borewell", 1, 2, fault=True, seed=42)
    scenario_b = create_unit_scenario(config, "hard_borewell", 1, 2, fault=True, seed=43)
    assert not np.array_equal(scenario_a.usage_raw, scenario_b.usage_raw)
    assert scenario_a.fault_type != scenario_b.fault_type or scenario_a.fault_onset_day != scenario_b.fault_onset_day


def test_usage_values_are_within_bounds(config):
    scenario = create_unit_scenario(config, "tanker", 2, 4, fault=False, seed=7)
    assert np.all(scenario.usage_raw >= config.usage["min_lpd"])
    assert np.all(scenario.usage_raw <= config.usage["max_lpd"])


def test_weekend_multiplier_is_applied(config):
    scenario = create_unit_scenario(config, "soft_municipal", 0, 0, fault=False, seed=10)
    week = np.arange(config.horizon_days) % 7
    weekend = np.isin(week, [5, 6])
    assert np.all(scenario.usage_raw[weekend] >= 0)


def test_absence_days_are_represented_correctly(config):
    scenario = create_unit_scenario(config, "soft_municipal", 0, 0, fault=False, seed=11)
    assert scenario.absence_mask.shape == (config.horizon_days,)
    assert np.all((scenario.usage_raw[scenario.absence_mask] == 0.0))


def test_tanker_multiplier_non_tanker_is_one(config):
    scenario = create_unit_scenario(config, "soft_municipal", 0, 0, fault=False, seed=12)
    assert np.all(scenario.tanker_multiplier == 1.0)


def test_tanker_deliveries_occur_at_day_zero_and_intervals(config):
    scenario = create_unit_scenario(config, "tanker", 2, 1, fault=False, seed=13)
    assert scenario.tanker_multiplier[0] > 0
    delivery_days = np.flatnonzero(np.diff(scenario.tanker_multiplier) != 0)
    assert len(delivery_days) >= 0
    assert np.all(scenario.tanker_multiplier >= 0.6)


def test_tanker_multiplier_remains_constant_between_deliveries(config):
    scenario = create_unit_scenario(config, "tanker", 2, 0, fault=False, seed=14)
    changes = np.where(np.diff(scenario.tanker_multiplier) != 0)[0]
    if len(changes) > 0:
        for start, end in zip(changes[:-1], changes[1:]):
            segment = scenario.tanker_multiplier[start + 1 : end + 1]
            assert np.all(segment == segment[0])


def test_generated_arrays_have_expected_lengths(config):
    scenario = create_unit_scenario(config, "hard_borewell", 1, 2, fault=False, seed=15)
    for name in [
        "usage_raw",
        "absence_mask",
        "water_noise_tds",
        "water_noise_turb",
        "temp_noise",
        "obs_noise_tds_in",
        "obs_noise_tds_out",
        "obs_noise_runtime",
        "obs_noise_temp",
        "tanker_multiplier",
        "user_delay",
    ]:
        value = getattr(scenario, name)
        assert len(value) == config.horizon_days


def test_fault_free_unit_contains_no_fault(config):
    scenario = create_unit_scenario(config, "soft_municipal", 0, 9, fault=False, seed=16)
    assert scenario.fault_type is None
    assert scenario.fault_onset_day is None
    assert scenario.fault is None


def test_fault_units_contain_exactly_one_fault_type(config):
    scenario = create_unit_scenario(config, "hard_borewell", 1, 8, fault=True, seed=17)
    assert scenario.fault_type in {"sudden_clog", "membrane_breach", "bypass"}
    assert scenario.fault is not None
    assert scenario.fault_onset_day is not None


def test_fault_onset_is_in_configured_range(config):
    scenario = create_unit_scenario(config, "tanker", 2, 10, fault=True, seed=18)
    lo, hi = config.faults["onset_day_range"]
    assert lo <= scenario.fault_onset_day <= hi


def test_fault_specific_parameters_stay_in_ranges(config):
    scenario = create_unit_scenario(config, "high_fouling", 3, 11, fault=True, seed=19)
    if scenario.fault_type == "sudden_clog":
        lo, hi = config.faults["sudden_clog"]["load_fraction_range"]
        assert lo <= scenario.fault["load_fraction"] <= hi
    elif scenario.fault_type == "membrane_breach":
        lo, hi = config.faults["membrane_breach"]["rejection_loss_range"]
        assert lo <= scenario.fault["rejection_loss"] <= hi
    elif scenario.fault_type == "bypass":
        lo, hi = config.faults["bypass"]["fraction_range"]
        assert lo <= scenario.fault["fraction"] <= hi
        lo_d, hi_d = config.faults["bypass"]["duration_range"]
        assert lo_d <= scenario.fault["duration"] <= hi_d


def test_random_streams_do_not_alias(config):
    scenario = create_unit_scenario(config, "soft_municipal", 0, 12, fault=True, seed=20)
    assert not np.array_equal(scenario.water_noise_tds, scenario.water_noise_turb)
    assert not np.array_equal(scenario.obs_noise_tds_in, scenario.obs_noise_tds_out)
    assert not np.array_equal(scenario.obs_noise_runtime, scenario.obs_noise_temp)


def test_scenario_creation_does_not_modify_global_numpy_state(config):
    state_before = np.random.get_state()
    _ = create_unit_scenario(config, "soft_municipal", 0, 13, fault=False, seed=21)
    state_after = np.random.get_state()
    assert np.array_equal(state_before[1], state_after[1])
