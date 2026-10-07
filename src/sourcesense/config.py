from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
REQUIRED_TOP_LEVEL = {
    "stages",
    "profiles",
    "device",
    "filters",
    "coupling",
    "usage",
    "climate",
    "noise",
    "water_profiles",
    "unit_heterogeneity",
    "faults",
    "dataset",
    "rul",
    "alerts",
    "anomaly",
    "policies",
}


@dataclass(frozen=True, slots=True)
class Config:
    seed: int
    start_date: str
    horizon_days: int
    policy_horizon_days: int
    stages: tuple[str, ...]
    profiles: tuple[str, ...]
    device: Mapping[str, Any]
    filters: Mapping[str, Any]
    coupling: Mapping[str, Any]
    usage: Mapping[str, Any]
    climate: Mapping[str, Any]
    noise: Mapping[str, Any]
    water_profiles: Mapping[str, Mapping[str, Any]]
    unit_heterogeneity: Mapping[str, Any]
    faults: Mapping[str, Any]
    dataset: Mapping[str, Any]
    rul: Mapping[str, Any]
    alerts: Mapping[str, Any]
    anomaly: Mapping[str, Any]
    policies: Mapping[str, Any]


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _require_keys(mapping: Mapping[str, Any], required: set[str], section_name: str) -> None:
    missing = sorted(set(required) - set(mapping.keys()))
    if missing:
        suffix = ", ".join(missing)
        raise ValueError(f"Missing configuration for {section_name}: {suffix}")


def _require_range(name: str, value: Any) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a 2-item numeric range")
    lo, hi = value
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        raise ValueError(f"{name} must contain numeric bounds")
    if lo > hi:
        raise ValueError(f"{name} has lo > hi")


def _validate_water_profiles(cfg: dict[str, Any]) -> None:
    water_profiles = cfg.get("water_profiles")
    if not isinstance(water_profiles, dict):
        raise ValueError("water_profiles must be a mapping")
    required_profile_keys = {
        "tds",
        "hardness_ratio",
        "turbidity",
        "carbon_load",
        "monsoon_tds_drop",
        "monsoon_turb_gain",
    }
    for profile_name in cfg.get("profiles", []):
        if profile_name not in water_profiles:
            raise ValueError(f"water_profiles.{profile_name} does not exist")
        profile_cfg = water_profiles[profile_name]
        _require_keys(profile_cfg, required_profile_keys, f"water_profiles.{profile_name}")
        if profile_name == "tanker":
            _require_keys(profile_cfg, {"delivery_interval_days", "delivery_sigma", "delivery_clip"}, f"water_profiles.{profile_name}")
        for key, value in profile_cfg.items():
            if key in {"tds", "hardness_ratio", "turbidity", "carbon_load", "delivery_interval_days", "delivery_clip"}:
                _require_range(f"water_profiles.{profile_name}.{key}", value)
            elif key in {"monsoon_tds_drop", "monsoon_turb_gain", "delivery_sigma"}:
                if not isinstance(value, (int, float)):
                    raise ValueError(f"water_profiles.{profile_name}.{key} must be numeric")


def _validate_split(cfg: dict[str, Any]) -> None:
    dataset = cfg.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a mapping")
    split = dataset.get("split")
    if not isinstance(split, dict):
        raise ValueError("dataset.split must exist")
    _require_keys(split, {"train", "val", "test"}, "dataset.split")
    total = split["train"] + split["val"] + split["test"]
    units_per_profile = dataset.get("units_per_profile")
    if units_per_profile is None:
        raise ValueError("dataset.units_per_profile is required")
    if total != units_per_profile:
        raise ValueError("dataset.split must sum to units_per_profile")


def _validate_fault_ranges(cfg: dict[str, Any]) -> None:
    faults = cfg.get("faults")
    if not isinstance(faults, dict):
        raise ValueError("faults must be a mapping")
    _require_keys(faults, {"fraction_units_with_fault", "onset_day_range", "types", "sudden_clog", "membrane_breach", "bypass"}, "faults")
    _require_range("faults.onset_day_range", faults["onset_day_range"])
    sudden = faults["sudden_clog"]
    membrane = faults["membrane_breach"]
    bypass = faults["bypass"]
    _require_keys(sudden, {"load_fraction_range", "active_days"}, "faults.sudden_clog")
    _require_keys(membrane, {"rejection_loss_range", "flux_boost"}, "faults.membrane_breach")
    _require_keys(bypass, {"fraction_range", "duration_range"}, "faults.bypass")
    _require_range("faults.sudden_clog.load_fraction_range", sudden["load_fraction_range"])
    _require_range("faults.membrane_breach.rejection_loss_range", membrane["rejection_loss_range"])
    _require_range("faults.bypass.fraction_range", bypass["fraction_range"])
    _require_range("faults.bypass.duration_range", bypass["duration_range"])


def _validate_policies(cfg: dict[str, Any]) -> None:
    policies = cfg.get("policies")
    if not isinstance(policies, dict):
        raise ValueError("policies must be a mapping")
    _require_keys(policies, {"fixed_days", "rated_litres", "user_delay_days_range"}, "policies")
    fixed = policies["fixed_days"]
    rated = policies["rated_litres"]
    if not isinstance(fixed, dict) or not isinstance(rated, dict):
        raise ValueError("policies.fixed_days and policies.rated_litres must be mappings")
    for stage in cfg["stages"]:
        if stage not in fixed:
            raise ValueError(f"Missing configuration for stage {stage}")
        if stage not in rated:
            raise ValueError(f"Missing configuration for stage {stage}")
    _require_range("policies.user_delay_days_range", policies["user_delay_days_range"])


def _validate_quantiles(cfg: dict[str, Any]) -> None:
    rul = cfg.get("rul")
    if not isinstance(rul, dict):
        raise ValueError("rul must be a mapping")
    quantiles = rul.get("quantiles")
    if not isinstance(quantiles, (list, tuple)) or not quantiles:
        raise ValueError("rul.quantiles must be a non-empty list")
    values = list(quantiles)
    if any(not isinstance(v, (int, float)) for v in values):
        raise ValueError("rul.quantiles must be numeric")
    if any(v <= 0 or v >= 1 for v in values):
        raise ValueError("rul.quantiles must be between 0 and 1")
    if values != sorted(values):
        raise ValueError("rul.quantiles must be ordered ascending")


def _validate_config(cfg: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL - set(cfg.keys()))
    if missing:
        raise ValueError(f"Missing required configuration section(s): {', '.join(missing)}")

    stages = cfg.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a non-empty list")
    profiles = cfg.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profiles must be a non-empty list")

    for stage in stages:
        if stage not in cfg.get("policies", {}).get("fixed_days", {}):
            raise ValueError(f"Missing configuration for stage {stage}")
        if stage not in cfg.get("policies", {}).get("rated_litres", {}):
            raise ValueError(f"Missing configuration for stage {stage}")

    for profile_name in profiles:
        if profile_name not in cfg.get("water_profiles", {}):
            raise ValueError(f"water_profiles.{profile_name} does not exist")

    _validate_water_profiles(cfg)
    _validate_split(cfg)
    _validate_fault_ranges(cfg)
    _validate_policies(cfg)
    _validate_quantiles(cfg)

    for range_name, value in {
        "usage.mean_lpd_range": cfg["usage"].get("mean_lpd_range"),
        "usage.absence_len_range": cfg["usage"].get("absence_len_range"),
        "policies.user_delay_days_range": cfg["policies"].get("user_delay_days_range"),
        "dataset.phys_clip_days": cfg["rul"].get("phys_clip_days"),
        "anomaly.persistence": cfg["anomaly"].get("persistence"),
    }.items():
        if value is not None:
            _require_range(range_name, value)

    if "absence_len_range" in cfg["usage"]:
        lo, hi = cfg["usage"]["absence_len_range"]
        if not (isinstance(lo, int) and isinstance(hi, int)):
            raise ValueError("usage.absence_len_range must contain integer bounds")
    if "onset_day_range" in cfg["faults"]:
        lo, hi = cfg["faults"]["onset_day_range"]
        if not (isinstance(lo, int) and isinstance(hi, int)):
            raise ValueError("faults.onset_day_range must contain integer bounds")

    for profile_name, profile_cfg in cfg["water_profiles"].items():
        for key in ("tds", "hardness_ratio", "turbidity", "carbon_load"):
            if key in profile_cfg:
                _require_range(f"water_profiles.{profile_name}.{key}", profile_cfg[key])

    for key in ("train", "val", "test"):
        if not isinstance(cfg["dataset"]["split"][key], int):
            raise ValueError(f"dataset.split.{key} must be an integer")


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration file must contain a YAML mapping")
    _validate_config(raw)

    dataclass_fields = {
        "seed": raw["seed"],
        "start_date": raw["start_date"],
        "horizon_days": raw["horizon_days"],
        "policy_horizon_days": raw["policy_horizon_days"],
        "stages": tuple(raw["stages"]),
        "profiles": tuple(raw["profiles"]),
        "device": _freeze(raw["device"]),
        "filters": _freeze(raw["filters"]),
        "coupling": _freeze(raw["coupling"]),
        "usage": _freeze(raw["usage"]),
        "climate": _freeze(raw["climate"]),
        "noise": _freeze(raw["noise"]),
        "water_profiles": _freeze(raw["water_profiles"]),
        "unit_heterogeneity": _freeze(raw["unit_heterogeneity"]),
        "faults": _freeze(raw["faults"]),
        "dataset": _freeze(raw["dataset"]),
        "rul": _freeze(raw["rul"]),
        "alerts": _freeze(raw["alerts"]),
        "anomaly": _freeze(raw["anomaly"]),
        "policies": _freeze(raw["policies"]),
    }
    return Config(**dataclass_fields)
