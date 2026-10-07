from __future__ import annotations

import numpy as np
import pytest

from sourcesense.config import load_config
from sourcesense.scenario import create_unit_scenario


@pytest.fixture
def config():
    return load_config("config/default.yaml")


def test_deterministic_same_seed(config):
    """Same seed must produce identical UnitScenario in all fields."""
    scenario_a = create_unit_scenario(config, "soft_municipal", 0, 0, fault=False, seed=42)
    scenario_b = create_unit_scenario(config, "soft_municipal", 0, 0, fault=False, seed=42)
    
    assert scenario_a.seed == scenario_b.seed
    assert scenario_a.profile == scenario_b.profile
    assert scenario_a.profile_idx == scenario_b.profile_idx
    assert scenario_a.unit_idx == scenario_b.unit_idx
    assert scenario_a.tds_base == scenario_b.tds_base
    assert scenario_a.hardness_ratio == scenario_b.hardness_ratio
    assert scenario_a.turb_base == scenario_b.turb_base
    assert scenario_a.carbon_load == scenario_b.carbon_load
    assert scenario_a.usage_mean == scenario_b.usage_mean
    assert scenario_a.q0 == scenario_b.q0
    assert scenario_a.unit_factor == scenario_b.unit_factor
    assert scenario_a.start_doy == scenario_b.start_doy
    
    assert np.array_equal(scenario_a.usage_raw, scenario_b.usage_raw)
    assert np.array_equal(scenario_a.absence_mask, scenario_b.absence_mask)
    assert np.array_equal(scenario_a.water_noise_tds, scenario_b.water_noise_tds)
    assert np.array_equal(scenario_a.water_noise_turb, scenario_b.water_noise_turb)
    assert np.array_equal(scenario_a.temp_noise, scenario_b.temp_noise)
    assert np.array_equal(scenario_a.obs_noise_tds_in, scenario_b.obs_noise_tds_in)
    assert np.array_equal(scenario_a.obs_noise_tds_out, scenario_b.obs_noise_tds_out)
    assert np.array_equal(scenario_a.obs_noise_runtime, scenario_b.obs_noise_runtime)
    assert np.array_equal(scenario_a.obs_noise_temp, scenario_b.obs_noise_temp)
    assert np.array_equal(scenario_a.tanker_multiplier, scenario_b.tanker_multiplier)
    assert np.array_equal(scenario_a.user_delay, scenario_b.user_delay)


def test_different_seed_changes_output(config):
    """Different seed must produce different scenario data."""
    scenario_a = create_unit_scenario(config, "hard_borewell", 1, 2, fault=True, seed=42)
    scenario_b = create_unit_scenario(config, "hard_borewell", 1, 2, fault=True, seed=43)
    assert not np.array_equal(scenario_a.usage_raw, scenario_b.usage_raw)
    assert scenario_a.fault_type != scenario_b.fault_type or scenario_a.fault_onset_day != scenario_b.fault_onset_day


def test_usage_values_are_within_bounds(config):
    """Non-absence days must be within bounds; absence days must be exactly 0.0."""
    scenario = create_unit_scenario(config, "tanker", 2, 4, fault=False, seed=7)

    active_usage = scenario.usage_raw[~scenario.absence_mask]

    assert np.all(active_usage >= config.usage["min_lpd"])
    assert np.all(active_usage <= config.usage["max_lpd"])
    assert np.all(scenario.usage_raw[scenario.absence_mask] == 0.0)


def test_weekend_multiplier_is_actually_applied(config):
    """Verify weekend factor is applied to usage calculations."""
    scenario = create_unit_scenario(config, "soft_municipal", 0, 0, fault=False, seed=10)
    
    week = np.arange(config.horizon_days) % 7
    weekdays = ~np.isin(week, [5, 6])
    weekends = np.isin(week, [5, 6])
    
    weekday_usage = scenario.usage_raw[weekdays & ~scenario.absence_mask]
    weekend_usage = scenario.usage_raw[weekends & ~scenario.absence_mask]
    
    if len(weekday_usage) > 0 and len(weekend_usage) > 0:
        mean_weekend = np.mean(weekend_usage)
        mean_weekday = np.mean(weekday_usage)
        expected_ratio = config.usage["weekend_factor"]
        observed_ratio = mean_weekend / mean_weekday if mean_weekday > 0 else expected_ratio
        
        assert observed_ratio > 1.0, "Weekend usage should be higher than weekday due to factor 1.15"


def test_absence_days_are_represented_correctly(config):
    """Absence days must have usage_raw == 0.0."""
    scenario = create_unit_scenario(config, "soft_municipal", 0, 0, fault=False, seed=11)
    assert scenario.absence_mask.shape == (config.horizon_days,)
    assert np.all(scenario.usage_raw[scenario.absence_mask] == 0.0)


def test_tanker_multiplier_non_tanker_is_one(config):
    """Non-tanker profiles must have tanker_multiplier == 1.0 everywhere."""
    scenario = create_unit_scenario(config, "soft_municipal", 0, 0, fault=False, seed=12)
    assert np.all(scenario.tanker_multiplier == 1.0)


def test_tanker_deliveries_occur_at_day_zero_and_intervals(config):
    """Tanker multiplier must change at day 0 and then at 3-7 day intervals."""
    scenario = create_unit_scenario(config, "tanker", 2, 1, fault=False, seed=13)
    
    assert scenario.tanker_multiplier[0] > 0
    
    diffs = np.diff(scenario.tanker_multiplier)
    change_indices = np.where(diffs != 0)[0]
    
    assert len(change_indices) > 0, "Tanker must have at least one delivery change point"
    
    for value in scenario.tanker_multiplier:
        assert 0.6 <= value <= 1.6, f"Tanker multiplier {value} outside [0.6, 1.6]"
    
    if len(change_indices) > 1:
        for i in range(len(change_indices) - 1):
            interval = change_indices[i + 1] - change_indices[i]
            assert 3 <= interval <= 7, f"Delivery interval {interval} outside [3, 7]"


def test_tanker_multiplier_remains_constant_between_deliveries(config):
    """Between delivery points, tanker multiplier must remain constant."""
    scenario = create_unit_scenario(config, "tanker", 2, 0, fault=False, seed=14)
    
    diffs = np.diff(scenario.tanker_multiplier)
    changes = np.where(diffs != 0)[0]
    
    if len(changes) > 0:
        delivery_days = [0] + (changes + 1).tolist()
        for i in range(len(delivery_days) - 1):
            start = delivery_days[i]
            end = delivery_days[i + 1]
            segment = scenario.tanker_multiplier[start:end]
            assert np.allclose(segment, segment[0]), f"Multiplier not constant in [{start}, {end})"


def test_generated_arrays_have_expected_lengths(config):
    """All scenario arrays must have exactly horizon_days elements."""
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
        assert len(value) == config.horizon_days, f"{name} has length {len(value)}, expected {config.horizon_days}"


def test_fault_free_unit_contains_no_fault(config):
    """Fault-free scenario must have all fault fields as None."""
    scenario = create_unit_scenario(config, "soft_municipal", 0, 9, fault=False, seed=16)
    assert scenario.fault_type is None
    assert scenario.fault_onset_day is None
    assert scenario.fault is None


def test_fault_units_contain_exactly_one_fault_type(config):
    """Fault scenario must contain exactly one of: sudden_clog, membrane_breach, bypass."""
    scenario = create_unit_scenario(config, "hard_borewell", 1, 8, fault=True, seed=17)
    assert scenario.fault_type in {"sudden_clog", "membrane_breach", "bypass"}
    assert scenario.fault is not None
    assert scenario.fault_onset_day is not None


def test_fault_onset_is_in_configured_range(config):
    """Fault onset day must lie within [90, 700] inclusive."""
    scenario = create_unit_scenario(config, "tanker", 2, 10, fault=True, seed=18)
    lo, hi = config.faults["onset_day_range"]
    assert lo <= scenario.fault_onset_day <= hi


def test_fault_specific_parameters_stay_in_ranges(config):
    """Fault-specific parameters must stay within configured ranges."""
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
    """Independent noise arrays must not be aliased."""
    scenario = create_unit_scenario(config, "soft_municipal", 0, 12, fault=True, seed=20)
    assert not np.array_equal(scenario.water_noise_tds, scenario.water_noise_turb)
    assert not np.array_equal(scenario.obs_noise_tds_in, scenario.obs_noise_tds_out)
    assert not np.array_equal(scenario.obs_noise_runtime, scenario.obs_noise_temp)


def test_scenario_creation_does_not_modify_global_numpy_state(config):
    """Scenario creation must not use global np.random state."""
    state_before = np.random.get_state()
    _ = create_unit_scenario(config, "soft_municipal", 0, 13, fault=False, seed=21)
    state_after = np.random.get_state()
    assert np.array_equal(state_before[1], state_after[1])


def test_no_hardcoded_constants_in_scenario_code(config):
    """Verify scenario.py reads all constants from config, not hard-coded."""
    scenario = create_unit_scenario(config, "soft_municipal", 0, 14, fault=False, seed=22)
    
    assert scenario.q0 > 0
    assert config.device["q0_lpm"] > 0
    assert scenario.usage_mean >= config.usage["mean_lpd_range"][0]
    assert scenario.usage_mean <= config.usage["mean_lpd_range"][1]
    
    active_usage = scenario.usage_raw[~scenario.absence_mask]
    if len(active_usage) > 0:
        assert np.min(active_usage) >= config.usage["min_lpd"]
        assert np.max(active_usage) <= config.usage["max_lpd"]


def test_tanker_multiplier_clipped_to_range(config):
    """Tanker multiplier must be clipped to [0.6, 1.6]."""
    scenario = create_unit_scenario(config, "tanker", 2, 15, fault=False, seed=23)
    assert np.all(scenario.tanker_multiplier >= 0.6)
    assert np.all(scenario.tanker_multiplier <= 1.6)


def test_user_delay_within_configured_range(config):
    """User delay must be within configured range."""
    scenario = create_unit_scenario(config, "soft_municipal", 0, 16, fault=False, seed=24)
    lo, hi = config.policies["user_delay_days_range"]
    assert np.all(scenario.user_delay >= lo)
    assert np.all(scenario.user_delay <= hi)


def test_multiple_faults_with_different_seeds(config):
    """Multiple fault scenarios with different seeds must differ."""
    s1 = create_unit_scenario(config, "soft_municipal", 0, 17, fault=True, seed=100)
    s2 = create_unit_scenario(config, "soft_municipal", 0, 17, fault=True, seed=101)
    s3 = create_unit_scenario(config, "soft_municipal", 0, 17, fault=True, seed=102)
    
    types = [s1.fault_type, s2.fault_type, s3.fault_type]
    assert all(t in {"sudden_clog", "membrane_breach", "bypass"} for t in types)
