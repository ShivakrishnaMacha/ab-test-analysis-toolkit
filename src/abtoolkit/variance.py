"""CUPED — Controlled-experiment Using Pre-Experiment Data.

Regress the post-period metric on a pre-period covariate (measured before
randomization, so it cannot be affected by treatment) and analyze the
residuals. When the covariate correlates with the outcome, variance drops
by a factor of (1 - rho^2), which shrinks confidence intervals and boosts
power — without changing the point estimate in expectation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import TestResult, ttest_two_sample


@dataclass
class CupedResult:
    theta: float
    correlation: float
    variance_reduction: float  # 1 - Var(adjusted) / Var(raw), pooled
    adjusted_control: np.ndarray
    adjusted_treatment: np.ndarray


def _as_float(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def cuped_adjust(pre: np.ndarray, post: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Apply the CUPED adjustment.

    Returns (adjusted_post, theta, variance_reduction_ratio).
    theta = Cov(post, pre) / Var(pre); the covariate is centered on its
    pooled mean so the adjustment is mean-zero.
    """
    pre = _as_float(pre)
    post = _as_float(post)
    var_pre = float(np.var(pre, ddof=1))
    if var_pre == 0:
        raise ValueError("pre-period covariate has zero variance")
    theta = float(np.cov(post, pre, ddof=1)[0, 1] / var_pre)
    adjusted = post - theta * (pre - float(np.mean(pre)))
    var_ratio = float(np.var(adjusted, ddof=1) / np.var(post, ddof=1))
    variance_reduction = 1.0 - var_ratio
    correlation = float(np.corrcoef(post, pre)[0, 1])
    return adjusted, theta, variance_reduction, correlation


def cuped_ttest(
    df: pd.DataFrame,
    pre_col: str,
    post_col: str,
    variant_col: str = "variant",
    alpha: float = 0.05,
) -> tuple[TestResult, CupedResult]:
    """CUPED-adjusted two-sample t-test.

    Fits theta on pooled pre-period data, adjusts both arms with the same
    theta (the standard CUPED recipe), then runs Welch's t-test on the
    adjusted metric. Returns (test_result, cuped_diagnostics).
    """
    control_mask = df[variant_col] == "control"
    treatment_mask = df[variant_col] == "treatment"
    pre = df[pre_col].to_numpy(dtype=float)
    post = df[post_col].to_numpy(dtype=float)

    pooled_pre = pre
    adjusted, theta, variance_reduction, correlation = cuped_adjust(pooled_pre, post)

    adj_control = adjusted[control_mask.to_numpy()]
    adj_treatment = adjusted[treatment_mask.to_numpy()]

    result = ttest_two_sample(adj_control, adj_treatment, alpha=alpha,
                              metric=f"{post_col} (CUPED-adjusted)")
    diagnostics = CupedResult(
        theta=theta,
        correlation=correlation,
        variance_reduction=variance_reduction,
        adjusted_control=adj_control,
        adjusted_treatment=adj_treatment,
    )
    return result, diagnostics
