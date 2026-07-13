from __future__ import annotations


def dataset_sample_interval_steps(sample_interval_s: float = 0.05, physics_dt: float = 1.0 / 120.0) -> int:
    if sample_interval_s <= 0.0:
        raise ValueError("sample_interval_s must be positive")
    if physics_dt <= 0.0:
        raise ValueError("physics_dt must be positive")
    return max(1, round(sample_interval_s / physics_dt))


def should_continue_collection(recorded: int, target_demos: int, attempts: int, max_attempts: int | None) -> bool:
    if recorded >= target_demos:
        return False
    if max_attempts is not None and attempts >= max_attempts:
        return False
    return True
