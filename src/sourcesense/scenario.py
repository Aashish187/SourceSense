from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class UnitScenario:
    """
    Pre-generated randomness for one simulated purifier unit.
    
    All random draws are deterministic based on seed, profile_idx, unit_idx.
    The simulator must only READ these arrays and never regenerate them.
    Replacing filters must never modify scenario arrays.
    """

    seed: int
    profile: str
    profile_idx: int
    unit_idx: int
    horizon_days: int

    tds_base: float
    hardness_ratio: float
    turb_base: float
    carbon_load: float
    usage_mean: float
    q0: float
    unit_factor: float
    start_doy: int

    usage_raw: np.ndarray
    absence_mask: np.ndarray

    water_noise_tds: np.ndarray
    water_noise_turb: np.ndarray
    temp_noise: np.ndarray

    obs_noise_tds_in: np.ndarray
    obs_noise_tds_out: np.ndarray
    obs_noise_runtime: np.ndarray
    obs_noise_temp: np.ndarray

    tanker_multiplier: np.ndarray
    user_delay: np.ndarray

    fault_type: str | None
    fault_onset_day: int | None
    fault: Mapping[str, Any] | None


def _child_rng(seed: int, profile_idx: int, unit_idx: int, kind: int) -> np.random.Generator:
    """
    Create a deterministic independent child RNG stream.
    
    Uses SeedSequence to derive independent streams from a base seed.
    Same (seed, profile_idx, unit_idx, kind) always produces the same stream.
    """
    return np.random.default_rng(np.random.SeedSequence([seed, profile_idx, unit_idx, kind]))


def _draw_tanker_multiplier(rng: np.random.Generator, horizon_days: int) -> np.ndarray:
    """
    Generate tanker water quality multiplier array.
    
    Deliveries occur at day 0, then at intervals of 3-7 days.
    Each delivery's multiplier is drawn from lognormal(0, 0.25) and clipped to [0.6, 1.6].
    Multiplier remains constant between deliveries.
    """
    multiplier = np.ones(horizon_days, dtype=np.float64)
    if horizon_days <= 0:
        return multiplier

    delivery_days = [0]
    next_day = 0
    value = rng.lognormal(mean=0.0, sigma=0.25)
    value = float(np.clip(value, 0.6, 1.6))
    values = [value]

    while True:
        interval = int(rng.integers(3, 8))
        next_day += interval
        if next_day >= horizon_days:
            break
        delivery_days.append(next_day)
        value = rng.lognormal(mean=0.0, sigma=0.25)
        value = float(np.clip(value, 0.6, 1.6))
        values.append(value)

    for day in range(1, horizon_days):
        prior_day = max(d for d in delivery_days if d <= day)
        idx = delivery_days.index(prior_day)
        multiplier[day] = values[idx]
    multiplier[0] = values[0]
    return multiplier


def create_unit_scenario(
    config,
    profile: str,
    profile_idx: int,
    unit_idx: int,
    fault: bool = False,
    seed: int | None = None,
) -> UnitScenario:
    """
    Create a deterministic UnitScenario.
    
    All randomness is pre-generated using independent RNG streams.
    Same seed + profile_idx + unit_idx + fault flag produces identical scenario.
    Different seed produces different scenario.
    
    Args:
        config: Configuration object with water_profiles, device, usage, etc.
        profile: Profile name (e.g., "soft_municipal", "tanker")
        profile_idx: Profile index (0, 1, 2, 3)
        unit_idx: Unit index within profile
        fault: Whether to include a fault
        seed: Random seed (defaults to config.seed)
    
    Returns:
        UnitScenario with all pre-generated arrays
    """
    if seed is None:
        seed = int(config.seed)

    profile_cfg = config.water_profiles[profile]
    horizon_days = int(config.horizon_days)

    usage_rng = _child_rng(int(seed), int(profile_idx), int(unit_idx), 0)
    water_rng = _child_rng(int(seed), int(profile_idx), int(unit_idx), 1)
    noise_rng = _child_rng(int(seed), int(profile_idx), int(unit_idx), 2)
    fault_rng = _child_rng(int(seed), int(profile_idx), int(unit_idx), 3)
    delay_rng = _child_rng(int(seed), int(profile_idx), int(unit_idx), 4)
    tanker_rng = _child_rng(int(seed), int(profile_idx), int(unit_idx), 5)

    tds_base = float(usage_rng.uniform(profile_cfg["tds"][0], profile_cfg["tds"][1]))
    hardness_ratio = float(usage_rng.uniform(profile_cfg["hardness_ratio"][0], profile_cfg["hardness_ratio"][1]))
    turb_base = float(usage_rng.uniform(profile_cfg["turbidity"][0], profile_cfg["turbidity"][1]))
    carbon_load = float(usage_rng.uniform(profile_cfg["carbon_load"][0], profile_cfg["carbon_load"][1]))
    usage_mean = float(usage_rng.uniform(config.usage["mean_lpd_range"][0], config.usage["mean_lpd_range"][1]))
    q0 = float(config.device["q0_lpm"] * usage_rng.lognormal(mean=0.0, sigma=config.device["q0_unit_sigma"]))
    unit_factor = float(usage_rng.lognormal(mean=0.0, sigma=config.unit_heterogeneity["ro_wear_sigma"]))
    start_doy = int(usage_rng.integers(0, 365))

    weekend_factor = config.usage["weekend_factor"]
    usage_raw = np.empty(horizon_days, dtype=np.float64)
    for day in range(horizon_days):
        day_factor = weekend_factor if (day % 7) in (5, 6) else 1.0
        value = usage_mean * day_factor * usage_rng.lognormal(mean=0.0, sigma=config.usage["daily_sigma"])
        usage_raw[day] = float(np.clip(value, config.usage["min_lpd"], config.usage["max_lpd"]))

    absence_mask = np.zeros(horizon_days, dtype=bool)
    day = 0
    while day < horizon_days:
        if absence_mask[day]:
            day += 1
            continue
        if usage_rng.random() < config.usage["absence_start_prob"]:
            low, high = config.usage["absence_len_range"]
            duration = int(usage_rng.integers(low, high + 1))
            end = min(horizon_days, day + duration)
            absence_mask[day:end] = True
            day = end
        else:
            day += 1
    usage_raw[absence_mask] = 0.0

    water_noise_tds = water_rng.normal(size=horizon_days)
    water_noise_turb = water_rng.normal(size=horizon_days)
    temp_noise = noise_rng.normal(size=horizon_days)

    obs_noise_tds_in = noise_rng.normal(size=horizon_days)
    obs_noise_tds_out = noise_rng.normal(size=horizon_days)
    obs_noise_runtime = noise_rng.normal(size=horizon_days)
    obs_noise_temp = noise_rng.normal(size=horizon_days)

    if profile == "tanker":
        tanker_multiplier = _draw_tanker_multiplier(tanker_rng, horizon_days)
    else:
        tanker_multiplier = np.ones(horizon_days, dtype=np.float64)

    user_delay = delay_rng.integers(
        low=config.policies["user_delay_days_range"][0],
        high=config.policies["user_delay_days_range"][1] + 1,
        size=horizon_days,
        dtype=np.int64,
    )

    fault_type = None
    fault_onset_day = None
    fault_value: dict[str, Any] | None = None
    if fault:
        fault_type = fault_rng.choice(["sudden_clog", "membrane_breach", "bypass"]).item()
        onset_low, onset_high = config.faults["onset_day_range"]
        fault_onset_day = int(fault_rng.integers(onset_low, onset_high + 1))

        if fault_type == "sudden_clog":
            low, high = config.faults["sudden_clog"]["load_fraction_range"]
            fault_value = {
                "load_fraction": float(fault_rng.uniform(low, high)),
                "active_days": int(config.faults["sudden_clog"]["active_days"]),
            }
        elif fault_type == "membrane_breach":
            low, high = config.faults["membrane_breach"]["rejection_loss_range"]
            fault_value = {
                "rejection_loss": float(fault_rng.uniform(low, high)),
                "flux_boost": float(config.faults["membrane_breach"]["flux_boost"]),
            }
        elif fault_type == "bypass":
            low, high = config.faults["bypass"]["fraction_range"]
            bypass_fraction = float(fault_rng.uniform(low, high))
            low_d, high_d = config.faults["bypass"]["duration_range"]
            bypass_duration = int(fault_rng.integers(low_d, high_d + 1))
            fault_value = {
                "fraction": bypass_fraction,
                "duration": bypass_duration,
            }

    return UnitScenario(
        seed=int(seed),
        profile=str(profile),
        profile_idx=int(profile_idx),
        unit_idx=int(unit_idx),
        horizon_days=horizon_days,
        tds_base=tds_base,
        hardness_ratio=hardness_ratio,
        turb_base=turb_base,
        carbon_load=carbon_load,
        usage_mean=usage_mean,
        q0=q0,
        unit_factor=unit_factor,
        start_doy=start_doy,
        usage_raw=usage_raw,
        absence_mask=absence_mask,
        water_noise_tds=water_noise_tds,
        water_noise_turb=water_noise_turb,
        temp_noise=temp_noise,
        obs_noise_tds_in=obs_noise_tds_in,
        obs_noise_tds_out=obs_noise_tds_out,
        obs_noise_runtime=obs_noise_runtime,
        obs_noise_temp=obs_noise_temp,
        tanker_multiplier=tanker_multiplier,
        user_delay=user_delay,
        fault_type=fault_type,
        fault_onset_day=fault_onset_day,
        fault=fault_value,
    )
