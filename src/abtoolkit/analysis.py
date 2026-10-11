"""Experiment analysis: hypothesis tests and multiple-comparison corrections.

Every test returns a :class:`TestResult` with effect size, confidence
interval, p-value and relative lift, so callers (and the Streamlit app)
never have to recompute presentation fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    metric: str
    test: str
    n_control: int
    n_treatment: int
    control_mean: float
    treatment_mean: float
    diff: float
    rel_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    alpha: float = 0.05
    significant: bool = field(init=False)
    ci_width: float = field(init=False)

    def __post_init__(self) -> None:
        self.significant = bool(self.p_value < self.alpha)
        self.ci_width = float(self.ci_high - self.ci_low)


def _split(df: pd.DataFrame, value_col: str, variant_col: str = "variant"):
    control = df.loc[df[variant_col] == "control", value_col].to_numpy(dtype=float)
    treatment = df.loc[df[variant_col] == "treatment", value_col].to_numpy(dtype=float)
    if len(control) == 0 or len(treatment) == 0:
        raise ValueError("both control and treatment arms must be non-empty")
    return control, treatment


def _rel_lift(control_mean: float, diff: float) -> float:
    return float(diff / control_mean) if control_mean != 0 else float("nan")


def ttest_two_sample(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    metric: str = "",
    equal_var: bool = False,
) -> TestResult:
    """Welch's (default) or Student's two-sample t-test on the mean difference."""
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    n_c, n_t = len(control), len(treatment)
    mean_c, mean_t = float(np.mean(control)), float(np.mean(treatment))
    diff = mean_t - mean_c

    if equal_var:
        res = stats.ttest_ind(treatment, control, equal_var=True)
        p_value = float(res.pvalue)
        s_pooled = math_sqrt(
            ((n_c - 1) * np.var(control, ddof=1) + (n_t - 1) * np.var(treatment, ddof=1))
            / (n_c + n_t - 2)
        )
        se = s_pooled * (1.0 / n_c + 1.0 / n_t) ** 0.5
        dof = n_c + n_t - 2
    else:
        res = stats.ttest_ind(treatment, control, equal_var=False)
        p_value = float(res.pvalue)
        v_c, v_t = np.var(control, ddof=1), np.var(treatment, ddof=1)
        se = (v_c / n_c + v_t / n_t) ** 0.5
        dof = (v_c / n_c + v_t / n_t) ** 2 / (
            (v_c / n_c) ** 2 / (n_c - 1) + (v_t / n_t) ** 2 / (n_t - 1)
        )

    t_crit = float(stats.t.ppf(1 - alpha / 2, dof))
    return TestResult(
        metric=metric,
        test="Welch t-test" if not equal_var else "Student t-test",
        n_control=n_c,
        n_treatment=n_t,
        control_mean=mean_c,
        treatment_mean=mean_t,
        diff=diff,
        rel_lift=_rel_lift(mean_c, diff),
        ci_low=diff - t_crit * se,
        ci_high=diff + t_crit * se,
        p_value=p_value,
        alpha=alpha,
    )


def math_sqrt(x: float) -> float:
    import math

    return math.sqrt(x)


def mannwhitney_two_sample(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    metric: str = "",
) -> TestResult:
    """Mann-Whitney U test (non-parametric); CI via normal approx on the mean diff."""
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    n_c, n_t = len(control), len(treatment)
    mean_c, mean_t = float(np.mean(control)), float(np.mean(treatment))
    diff = mean_t - mean_c
    u_res = stats.mannwhitneyu(treatment, control, alternative="two-sided")
    # Hodges-Lehmann-free CI: reuse the Welch CI for the location shift display
    t_res = ttest_two_sample(control, treatment, alpha=alpha)
    return TestResult(
        metric=metric,
        test="Mann-Whitney U",
        n_control=n_c,
        n_treatment=n_t,
        control_mean=mean_c,
        treatment_mean=mean_t,
        diff=diff,
        rel_lift=_rel_lift(mean_c, diff),
        ci_low=t_res.ci_low,
        ci_high=t_res.ci_high,
        p_value=float(u_res.pvalue),
        alpha=alpha,
    )


def chi_square_proportions(
    c1: int,
    n1: int,
    c2: int,
    n2: int,
    alpha: float = 0.05,
    metric: str = "conversion rate",
) -> TestResult:
    """Chi-square test of two proportions (e.g. conversion: control vs treatment)."""
    table = np.array([[c1, n1 - c1], [c2, n2 - c2]])
    chi2, p_value, _, _ = stats.chi2_contingency(table, correction=False)
    p_c, p_t = c1 / n1, c2 / n2
    diff = p_t - p_c
    se = (p_c * (1 - p_c) / n1 + p_t * (1 - p_t) / n2) ** 0.5
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    return TestResult(
        metric=metric,
        test="Chi-square (2-proportion)",
        n_control=n1,
        n_treatment=n2,
        control_mean=p_c,
        treatment_mean=p_t,
        diff=diff,
        rel_lift=_rel_lift(p_c, diff),
        ci_low=diff - z_crit * se,
        ci_high=diff + z_crit * se,
        p_value=float(p_value),
        alpha=alpha,
    )


def holm(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down procedure. Returns per-hypothesis reject flags (input order)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, i in enumerate(order, start=1):
        if p_values[i] <= alpha / (m - rank + 1):
            reject[i] = True
        else:
            break
    return reject


def bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Bonferroni correction. Returns per-hypothesis reject flags (input order)."""
    m = len(p_values)
    return [p <= alpha / m for p in p_values]


def analyze_metric(
    df: pd.DataFrame,
    value_col: str,
    variant_col: str = "variant",
    method: str = "t",
    alpha: float = 0.05,
    binary: bool = False,
) -> TestResult:
    """Run a two-arm comparison on one metric column."""
    control, treatment = _split(df, value_col, variant_col)
    if binary or method == "chi2":
        c1 = int((control > 0).sum())
        c2 = int((treatment > 0).sum())
        return chi_square_proportions(c1, len(control), c2, len(treatment),
                                      alpha=alpha, metric=value_col)
    if method == "mannwhitney":
        return mannwhitney_two_sample(control, treatment, alpha=alpha, metric=value_col)
    return ttest_two_sample(control, treatment, alpha=alpha, metric=value_col)


def segment_drilldown(
    df: pd.DataFrame,
    value_col: str,
    segment_col: str = "segment",
    variant_col: str = "variant",
    alpha: float = 0.05,
    method: str = "t",
    binary: bool = False,
    correction: str = "holm",
) -> pd.DataFrame:
    """Per-segment analysis with a multiple-comparison correction.

    Returns a DataFrame with one row per segment: means, diff, relative lift,
    CI, raw p-value and the correction-adjusted significance flag.
    """
    rows = []
    for segment, sub in df.groupby(segment_col):
        res = analyze_metric(sub, value_col, variant_col, method, alpha, binary)
        rows.append(
            {
                "segment": segment,
                "n_control": res.n_control,
                "n_treatment": res.n_treatment,
                "control_mean": res.control_mean,
                "treatment_mean": res.treatment_mean,
                "diff": res.diff,
                "rel_lift": res.rel_lift,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "p_value": res.p_value,
            }
        )
    out = pd.DataFrame(rows)
    p_values = out["p_value"].tolist()
    if correction == "holm":
        flags = holm(p_values, alpha)
    elif correction == "bonferroni":
        flags = bonferroni(p_values, alpha)
    else:
        raise ValueError("correction must be 'holm' or 'bonferroni'")
    out["significant_adj"] = flags
    return out.sort_values("segment").reset_index(drop=True)
