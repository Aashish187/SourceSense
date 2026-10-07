# SourceSense

SourceSense is a synthetic system for studying filter-health behavior in a connected RO/UF purifier.

This repository is intentionally focused on the deterministic configuration and scenario layer used before simulator logic. The project does not use real-world telemetry or operational measurements.

UnitScenario pre-generates all randomness for each unit so that later policy comparisons can operate on common random inputs. This design keeps simulations deterministic, fair, and reproducible.

This phase does not implement the purifier simulator, dataset generation, or downstream models.
