"""Experiment guardrails: SRM checks and guardrail-metric monitoring.

* SRM (sample-ratio mismatch): chi-square goodness-of-fit of the observed
  arm sizes against the intended allocation. A failure means randomization
  or logging is broken — analyze nothing until it is explained.
* Guardrail metrics: business/health metrics (bounce, pageviews, ...) that
  should NOT move. Compared across arms with a Holm correction so a
  dashboard full of metrics does not manufacture false alarms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import bonferroni, chi_square_proportions, holm, ttest_two_sample


@dataclass
class SrmResult:
    n_control: int
    n_treatment: int
    expected_treatment_ratio: float
    chi2: float
    p_value: float
    alpha: float
    passed: bool


def srm_check(
    df: pd.DataFrame | None = None,
    n_control: int | None = None,
    n_treatment: int | None = None,
    variant_col: str = "variant",
    expected_treatment_ratio: float = 0.5,
    alpha: float = 0.01,
) -> SrmResult:
    """Chi-square SRM check against the intended allocation ratio.

    Pass a DataFrame (arm sizes counted from ``variant_col``) or the two
    counts directly.
    """
    if df is not None:
        counts = df[variant_col].value_counts()
        n_control = int(counts.get("control", 0))
        n_treatment = int(counts.get("treatment", 0))
    if n_control is None or n_treatment is None:
        raise ValueError("supply df or both n_control and n_treatment")
    total = n_control + n_treatment
    if total == 0:
        raise ValueError("no observations")

    expected = np.array(
        [total * (1 - expected_treatment_ratio), total * expected_treatment_ratio]
    )
    observed = np.array([n_control, n_treatment], dtype=float)
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    p_value = float(stats.chi2.sf(chi2, df=1))
    return SrmResult(
        n_control=n_control,
        n_treatment=n_treatment,
        expected_treatment_ratio=expected_treatment_ratio,
        chi2=chi2,
        p_value=p_value,
        alpha=alpha,
        passed=bool(p_value >= alpha),
    )


def guardrail_checks(
    df: pd.DataFrame,
    metrics: dict[str, str] | None = None,
    variant_col: str = "variant",
    alpha: float = 0.05,
    correction: str = "holm",
) -> pd.DataFrame:
    """Compare guardrail metrics across arms with multiple-testing correction.

    Args:
        metrics: mapping of column -> "binary" | "continuous".
            Defaults to bounce (binary) and pageviews (continuous).

    Returns a DataFrame with one row per metric: means, diff, relative lift,
    p-value and the correction-adjusted significance flag. A True flag means
    the guardrail moved — investigate before shipping.
    """
    if metrics is None:
        metrics = {"bounce": "binary", "pageviews": "continuous"}

    rows = []
    for col, kind in metrics.items():
        if col not in df.columns:
            continue
        control = df.loc[df[variant_col] == "control", col].to_numpy(dtype=float)
        treatment = df.loc[df[variant_col] == "treatment", col].to_numpy(dtype=float)
        if kind == "binary":
            res = chi_square_proportions(
                int((control > 0).sum()), len(control),
                int((treatment > 0).sum()), len(treatment),
                alpha=alpha, metric=col,
            )
        else:
            res = ttest_two_sample(control, treatment, alpha=alpha, metric=col)
        rows.append(
            {
                "metric": col,
                "kind": kind,
                "control_mean": res.control_mean,
                "treatment_mean": res.treatment_mean,
                "diff": res.diff,
                "rel_lift": res.rel_lift,
                "p_value": res.p_value,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    p_values = out["p_value"].tolist()
    if correction == "holm":
        flags = holm(p_values, alpha)
    elif correction == "bonferroni":
        flags = bonferroni(p_values, alpha)
    else:
        raise ValueError("correction must be 'holm' or 'bonferroni'")
    out["significant_adj"] = flags
    return out.reset_index(drop=True)
