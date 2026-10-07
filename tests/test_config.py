from __future__ import annotations

import pytest

from sourcesense.config import load_config


@pytest.fixture
def config():
    return load_config("config/default.yaml")


def test_valid_configuration_loads(config):
    assert config.seed == 42
    assert config.horizon_days == 1600
    assert config.stages == ("sediment", "carbon", "uf", "ro_membrane")
    assert "soft_municipal" in config.water_profiles


def test_invalid_profile_raises_value_error():
    with pytest.raises(ValueError, match="water_profiles.*does not exist"):
        bad = {
            "seed": 1,
            "start_date": "2026-01-01",
            "horizon_days": 10,
            "policy_horizon_days": 5,
            "stages": ["sediment", "carbon"],
            "profiles": ["soft_municipal", "bad_profile"],
            "device": {"q0_lpm": 0.3, "q0_unit_sigma": 0.05, "recovery": 0.3, "max_runtime_fraction": 0.5, "tcf_per_degC": 0.03},
            "filters": {"tss_mg_per_ntu": 1.5, "sediment_ref_g": 60.0, "sediment_eff": 0.85, "sediment_eff_after_eol": 0.5},
            "coupling": {"enabled": True, "carbon_exhausted_ro_aging_multiplier": 1.5},
            "usage": {"mean_lpd_range": [10.0, 35.0], "weekend_factor": 1.15, "daily_sigma": 0.20, "min_lpd": 2.0, "max_lpd": 80.0, "absence_start_prob": 0.005, "absence_len_range": [3, 10]},
            "climate": {"temp_mean_c": 26.0, "temp_amp_c": 7.0, "temp_peak_doy": 136, "temp_noise_sigma": 1.0, "monsoon_peak_doy": 213},
            "noise": {"tds_in_rel": 0.02, "tds_out_rel": 0.03, "tds_out_abs_ppm": 2.0, "runtime_rel": 0.02, "temp_obs_sigma": 0.5, "tds_daily_rel": 0.03, "turbidity_daily_sigma": 0.30},
            "water_profiles": {
                "soft_municipal": {"tds": [80, 200], "hardness_ratio": [0.40, 0.60], "turbidity": [0.5, 2.5], "carbon_load": [0.8, 1.6], "monsoon_tds_drop": 0.10, "monsoon_turb_gain": 1.0},
                "tanker": {"tds": [300, 900], "hardness_ratio": [0.35, 0.55], "turbidity": [3.0, 13.0], "carbon_load": [1.0, 2.0], "monsoon_tds_drop": 0.08, "monsoon_turb_gain": 0.5, "delivery_interval_days": [3, 7], "delivery_sigma": 0.25, "delivery_clip": [0.6, 1.6]},
            },
            "unit_heterogeneity": {"ro_wear_sigma": 0.15},
            "faults": {"fraction_units_with_fault": 1.0, "onset_day_range": [90, 700], "types": ["sudden_clog", "membrane_breach", "bypass"], "sudden_clog": {"load_fraction_range": [0.5, 1.0], "active_days": 30}, "membrane_breach": {"rejection_loss_range": [0.08, 0.20], "flux_boost": 1.10}, "bypass": {"fraction_range": [0.3, 0.7], "duration_range": [20, 60]}},
            "dataset": {"units_per_profile": 2, "fault_units_per_profile": 1, "split": {"train": 1, "val": 1, "test": 0}, "profile_window_days": 14, "profile_min_valid_days": 7, "sample_stride_days": 5, "min_age_for_rul_days": 14},
            "rul": {"quantiles": [0.1, 0.5, 0.9], "conformal_alpha": 0.2, "max_rul_days": 1600, "phys_clip_days": [0, 3000]},
            "alerts": {"replace_soon_days": 14, "replace_now_days": 3},
            "anomaly": {"persistence": [3, 5], "threshold_quantile": 0.99, "detect_within_days": 14},
            "policies": {"fixed_days": {"sediment": 180, "carbon": 270}, "rated_litres": {"sediment": 3600, "carbon": 5400}, "user_delay_days_range": [0, 5]},
        }
        from sourcesense.config import _validate_config
        _validate_config(bad)


def test_invalid_range_raises_value_error():
    bad = {
        "seed": 1,
        "start_date": "2026-01-01",
        "horizon_days": 10,
        "policy_horizon_days": 5,
        "stages": ["sediment", "carbon"],
        "profiles": ["soft_municipal"],
        "device": {"q0_lpm": 0.3, "q0_unit_sigma": 0.05, "recovery": 0.3, "max_runtime_fraction": 0.5, "tcf_per_degC": 0.03},
        "filters": {"tss_mg_per_ntu": 1.5, "sediment_ref_g": 60.0, "sediment_eff": 0.85, "sediment_eff_after_eol": 0.5},
        "coupling": {"enabled": True, "carbon_exhausted_ro_aging_multiplier": 1.5},
        "usage": {"mean_lpd_range": [35.0, 10.0], "weekend_factor": 1.15, "daily_sigma": 0.20, "min_lpd": 2.0, "max_lpd": 80.0, "absence_start_prob": 0.005, "absence_len_range": [3, 10]},
        "climate": {"temp_mean_c": 26.0, "temp_amp_c": 7.0, "temp_peak_doy": 136, "temp_noise_sigma": 1.0, "monsoon_peak_doy": 213},
        "noise": {"tds_in_rel": 0.02, "tds_out_rel": 0.03, "tds_out_abs_ppm": 2.0, "runtime_rel": 0.02, "temp_obs_sigma": 0.5, "tds_daily_rel": 0.03, "turbidity_daily_sigma": 0.30},
        "water_profiles": {"soft_municipal": {"tds": [150, 80], "hardness_ratio": [0.40, 0.60], "turbidity": [0.5, 2.5], "carbon_load": [0.8, 1.6], "monsoon_tds_drop": 0.10, "monsoon_turb_gain": 1.0}},
        "unit_heterogeneity": {"ro_wear_sigma": 0.15},
        "faults": {"fraction_units_with_fault": 1.0, "onset_day_range": [90, 700], "types": ["sudden_clog", "membrane_breach", "bypass"], "sudden_clog": {"load_fraction_range": [0.5, 1.0], "active_days": 30}, "membrane_breach": {"rejection_loss_range": [0.08, 0.20], "flux_boost": 1.10}, "bypass": {"fraction_range": [0.3, 0.7], "duration_range": [20, 60]}},
        "dataset": {"units_per_profile": 2, "fault_units_per_profile": 1, "split": {"train": 1, "val": 1, "test": 0}, "profile_window_days": 14, "profile_min_valid_days": 7, "sample_stride_days": 5, "min_age_for_rul_days": 14},
        "rul": {"quantiles": [0.1, 0.5, 0.9], "conformal_alpha": 0.2, "max_rul_days": 1600, "phys_clip_days": [0, 3000]},
        "alerts": {"replace_soon_days": 14, "replace_now_days": 3},
        "anomaly": {"persistence": [3, 5], "threshold_quantile": 0.99, "detect_within_days": 14},
        "policies": {"fixed_days": {"sediment": 180, "carbon": 270}, "rated_litres": {"sediment": 3600, "carbon": 5400}, "user_delay_days_range": [0, 5]},
    }
    with pytest.raises(ValueError, match="lo > hi|must be a 2-item numeric range"):
        from sourcesense.config import _validate_config
        _validate_config(bad)


def test_invalid_split_raises_value_error():
    bad = {
        "seed": 1,
        "start_date": "2026-01-01",
        "horizon_days": 10,
        "policy_horizon_days": 5,
        "stages": ["sediment", "carbon"],
        "profiles": ["soft_municipal"],
        "device": {"q0_lpm": 0.3, "q0_unit_sigma": 0.05, "recovery": 0.3, "max_runtime_fraction": 0.5, "tcf_per_degC": 0.03},
        "filters": {"tss_mg_per_ntu": 1.5, "sediment_ref_g": 60.0, "sediment_eff": 0.85, "sediment_eff_after_eol": 0.5},
        "coupling": {"enabled": True, "carbon_exhausted_ro_aging_multiplier": 1.5},
        "usage": {"mean_lpd_range": [10.0, 35.0], "weekend_factor": 1.15, "daily_sigma": 0.20, "min_lpd": 2.0, "max_lpd": 80.0, "absence_start_prob": 0.005, "absence_len_range": [3, 10]},
        "climate": {"temp_mean_c": 26.0, "temp_amp_c": 7.0, "temp_peak_doy": 136, "temp_noise_sigma": 1.0, "monsoon_peak_doy": 213},
        "noise": {"tds_in_rel": 0.02, "tds_out_rel": 0.03, "tds_out_abs_ppm": 2.0, "runtime_rel": 0.02, "temp_obs_sigma": 0.5, "tds_daily_rel": 0.03, "turbidity_daily_sigma": 0.30},
        "water_profiles": {"soft_municipal": {"tds": [80, 200], "hardness_ratio": [0.40, 0.60], "turbidity": [0.5, 2.5], "carbon_load": [0.8, 1.6], "monsoon_tds_drop": 0.10, "monsoon_turb_gain": 1.0}},
        "unit_heterogeneity": {"ro_wear_sigma": 0.15},
        "faults": {"fraction_units_with_fault": 1.0, "onset_day_range": [90, 700], "types": ["sudden_clog", "membrane_breach", "bypass"], "sudden_clog": {"load_fraction_range": [0.5, 1.0], "active_days": 30}, "membrane_breach": {"rejection_loss_range": [0.08, 0.20], "flux_boost": 1.10}, "bypass": {"fraction_range": [0.3, 0.7], "duration_range": [20, 60]}},
        "dataset": {"units_per_profile": 4, "fault_units_per_profile": 1, "split": {"train": 1, "val": 1, "test": 1}, "profile_window_days": 14, "profile_min_valid_days": 7, "sample_stride_days": 5, "min_age_for_rul_days": 14},
        "rul": {"quantiles": [0.1, 0.5, 0.9], "conformal_alpha": 0.2, "max_rul_days": 1600, "phys_clip_days": [0, 3000]},
        "alerts": {"replace_soon_days": 14, "replace_now_days": 3},
        "anomaly": {"persistence": [3, 5], "threshold_quantile": 0.99, "detect_within_days": 14},
        "policies": {"fixed_days": {"sediment": 180, "carbon": 270}, "rated_litres": {"sediment": 3600, "carbon": 5400}, "user_delay_days_range": [0, 5]},
    }
    with pytest.raises(ValueError, match="dataset.split must sum to units_per_profile"):
        from sourcesense.config import _validate_config
        _validate_config(bad)


def test_missing_stage_configuration_raises_value_error():
    bad = {
        "seed": 1,
        "start_date": "2026-01-01",
        "horizon_days": 10,
        "policy_horizon_days": 5,
        "stages": ["sediment", "carbon", "uf", "ro_membrane"],
        "profiles": ["soft_municipal"],
        "device": {"q0_lpm": 0.3, "q0_unit_sigma": 0.05, "recovery": 0.3, "max_runtime_fraction": 0.5, "tcf_per_degC": 0.03},
        "filters": {"tss_mg_per_ntu": 1.5, "sediment_ref_g": 60.0, "sediment_eff": 0.85, "sediment_eff_after_eol": 0.5},
        "coupling": {"enabled": True, "carbon_exhausted_ro_aging_multiplier": 1.5},
        "usage": {"mean_lpd_range": [10.0, 35.0], "weekend_factor": 1.15, "daily_sigma": 0.20, "min_lpd": 2.0, "max_lpd": 80.0, "absence_start_prob": 0.005, "absence_len_range": [3, 10]},
        "climate": {"temp_mean_c": 26.0, "temp_amp_c": 7.0, "temp_peak_doy": 136, "temp_noise_sigma": 1.0, "monsoon_peak_doy": 213},
        "noise": {"tds_in_rel": 0.02, "tds_out_rel": 0.03, "tds_out_abs_ppm": 2.0, "runtime_rel": 0.02, "temp_obs_sigma": 0.5, "tds_daily_rel": 0.03, "turbidity_daily_sigma": 0.30},
        "water_profiles": {"soft_municipal": {"tds": [80, 200], "hardness_ratio": [0.40, 0.60], "turbidity": [0.5, 2.5], "carbon_load": [0.8, 1.6], "monsoon_tds_drop": 0.10, "monsoon_turb_gain": 1.0}},
        "unit_heterogeneity": {"ro_wear_sigma": 0.15},
        "faults": {"fraction_units_with_fault": 1.0, "onset_day_range": [90, 700], "types": ["sudden_clog", "membrane_breach", "bypass"], "sudden_clog": {"load_fraction_range": [0.5, 1.0], "active_days": 30}, "membrane_breach": {"rejection_loss_range": [0.08, 0.20], "flux_boost": 1.10}, "bypass": {"fraction_range": [0.3, 0.7], "duration_range": [20, 60]}},
        "dataset": {"units_per_profile": 2, "fault_units_per_profile": 1, "split": {"train": 1, "val": 1, "test": 0}, "profile_window_days": 14, "profile_min_valid_days": 7, "sample_stride_days": 5, "min_age_for_rul_days": 14},
        "rul": {"quantiles": [0.1, 0.5, 0.9], "conformal_alpha": 0.2, "max_rul_days": 1600, "phys_clip_days": [0, 3000]},
        "alerts": {"replace_soon_days": 14, "replace_now_days": 3},
        "anomaly": {"persistence": [3, 5], "threshold_quantile": 0.99, "detect_within_days": 14},
        "policies": {"fixed_days": {"sediment": 180, "carbon": 270}, "rated_litres": {"sediment": 3600, "carbon": 5400}, "user_delay_days_range": [0, 5]},
    }
    with pytest.raises(ValueError, match="Missing configuration for stage"):
        from sourcesense.config import _validate_config
        _validate_config(bad)
