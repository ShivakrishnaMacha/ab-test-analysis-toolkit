"""Experiment design: sample-size / power calculations and runtime estimates.

Normal-approximation formulas for two-sample tests, with support for
unequal allocation ratios. All functions are pure and unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


def _z(alpha: float, two_sided: bool = True) -> float:
    p = 1.0 - alpha / 2.0 if two_sided else 1.0 - alpha
    return stats.norm.ppf(p)


def _z_power(power: float) -> float:
    return stats.norm.ppf(power)


def _split_allocation(n_equal_per_arm: float, allocation: float) -> tuple[int, int]:
    """Split an equal-allocation per-arm N across an allocation ratio.

    allocation = n_treatment / n_control. Uses the standard approximation
    n_control = n_eq * (1 + k) / (2k), n_treatment = n_eq * (1 + k) / 2.
    """
    if allocation <= 0:
        raise ValueError("allocation must be positive")
    k = allocation
    n_control = n_equal_per_arm * (1.0 + k) / (2.0 * k)
    n_treatment = n_equal_per_arm * (1.0 + k) / 2.0
    return math.ceil(n_control), math.ceil(n_treatment)


@dataclass
class DesignResult:
    metric: str
    baseline: float
    mde_relative: float
    alpha: float
    power: float
    allocation: float  # n_treatment / n_control
    n_control: int
    n_treatment: int
    total_n: int
    daily_traffic: int
    runtime_days: int


def sample_size_proportions(
    p1: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 1.0,
    daily_traffic: int = 1000,
    metric: str = "conversion rate",
) -> DesignResult:
    """Required sample size for a two-proportion test (normal approximation).

    Args:
        p1: baseline rate (control).
        mde_relative: minimum detectable effect as a fraction of baseline,
            e.g. 0.05 for a +5% relative lift.
        alpha: significance level (two-sided).
        power: desired statistical power (1 - beta).
        allocation: n_treatment / n_control.
        daily_traffic: users/day available for the experiment.
    """
    if not 0 < p1 < 1:
        raise ValueError("p1 must be in (0, 1)")
    if mde_relative <= 0:
        raise ValueError("mde_relative must be positive")
    p2 = p1 * (1.0 + mde_relative)
    if not 0 < p2 < 1:
        raise ValueError("baseline + MDE leaves (0, 1); pick a smaller MDE")

    za = _z(alpha)
    zb = _z_power(power)
    p_bar = (p1 + p2) / 2.0
    n_eq = (
        (za * math.sqrt(2 * p_bar * (1 - p_bar))
         + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
        / (p2 - p1)
    ) ** 2
    n_control, n_treatment = _split_allocation(n_eq, allocation)
    total = n_control + n_treatment
    return DesignResult(
        metric=metric,
        baseline=p1,
        mde_relative=mde_relative,
        alpha=alpha,
        power=power,
        allocation=allocation,
        n_control=n_control,
        n_treatment=n_treatment,
        total_n=total,
        daily_traffic=daily_traffic,
        runtime_days=runtime_days(total, daily_traffic),
    )


def sample_size_means(
    sd: float,
    mde_relative: float,
    baseline_mean: float,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 1.0,
    daily_traffic: int = 1000,
    metric: str = "revenue per user",
) -> DesignResult:
    """Required sample size for a two-sample means test (normal approximation).

    Args:
        sd: pooled standard deviation of the metric.
        mde_relative: minimum detectable effect as a fraction of baseline_mean.
        baseline_mean: control mean (also used to express the absolute MDE).
    """
    if sd <= 0:
        raise ValueError("sd must be positive")
    if baseline_mean <= 0 or mde_relative <= 0:
        raise ValueError("baseline_mean and mde_relative must be positive")

    delta = baseline_mean * mde_relative
    za = _z(alpha)
    zb = _z_power(power)
    n_eq = 2.0 * ((za + zb) * sd / delta) ** 2
    n_control, n_treatment = _split_allocation(n_eq, allocation)
    total = n_control + n_treatment
    return DesignResult(
        metric=metric,
        baseline=baseline_mean,
        mde_relative=mde_relative,
        alpha=alpha,
        power=power,
        allocation=allocation,
        n_control=n_control,
        n_treatment=n_treatment,
        total_n=total,
        daily_traffic=daily_traffic,
        runtime_days=runtime_days(total, daily_traffic),
    )


def power_for_n_proportions(
    n_per_arm: int, p1: float, mde_relative: float, alpha: float = 0.05
) -> float:
    """Achieved power for a given per-arm N (two-proportion, equal allocation)."""
    p2 = p1 * (1.0 + mde_relative)
    se = math.sqrt(p1 * (1 - p1) / n_per_arm + p2 * (1 - p2) / n_per_arm)
    za = _z(alpha)
    return float(stats.norm.cdf(abs(p2 - p1) / se - za))


def runtime_days(total_n: int, daily_traffic: int) -> int:
    """Calendar days to accumulate total_n users at daily_traffic users/day."""
    if daily_traffic <= 0:
        raise ValueError("daily_traffic must be positive")
    return math.ceil(total_n / daily_traffic)


def sample_size_means_statsmodels(
    sd: float, delta: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Cross-check per-arm N for two-sample means via statsmodels.

    Independent implementation of the same normal approximation, used to
    validate :func:`sample_size_means` in tests.
    """
    from statsmodels.stats.power import NormalIndPower

    n = NormalIndPower().solve_power(
        effect_size=delta / sd, alpha=alpha, power=power, alternative="two-sided"
    )
    return math.ceil(float(n))
