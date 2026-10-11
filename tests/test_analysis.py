"""Tests for the A/B test analysis toolkit.

Uses the deterministic demo generator (fixed seed) as ground truth:
+3.5% revenue lift planted on returning users, null elsewhere.
"""

import numpy as np
import pandas as pd
import pytest

from abtoolkit import analysis, design, generator, guardrails, variance
from abtoolkit.generator import TREATMENT_LIFT

SEED = 42
N = 20000


@pytest.fixture(scope="module")
def demo_df() -> pd.DataFrame:
    return generator.generate_experiment(n_users=N, seed=SEED)


@pytest.fixture(scope="module")
def tampered_df() -> pd.DataFrame:
    return generator.generate_experiment(n_users=N, seed=SEED, tamper_srm=True)


# --- CUPED -----------------------------------------------------------------
def test_cuped_shrinks_ci_width_vs_naive_ttest(demo_df):
    naive = analysis.analyze_metric(demo_df, "revenue")
    cuped_res, diag = variance.cuped_ttest(demo_df, "pre_spend", "revenue")
    assert diag.variance_reduction > 0.10, "expected meaningful variance reduction"
    assert cuped_res.ci_width < naive.ci_width, (
        f"CUPED CI width {cuped_res.ci_width:.4f} not smaller than "
        f"naive {naive.ci_width:.4f}"
    )


def test_cuped_detects_planted_lift_on_returning(demo_df):
    returning = demo_df[demo_df["segment"] == "returning"]
    cuped_res, _ = variance.cuped_ttest(returning, "pre_spend", "revenue")
    assert cuped_res.significant, "planted +3.5% lift should be detected on returning"
    assert abs(cuped_res.rel_lift - TREATMENT_LIFT) < 0.025, (
        f"estimated lift {cuped_res.rel_lift:.2%} far from planted {TREATMENT_LIFT:.2%}"
    )


def test_null_segments_show_no_lift(demo_df):
    for seg in ("new", "mobile"):
        sub = demo_df[demo_df["segment"] == seg]
        cuped_res, _ = variance.cuped_ttest(sub, "pre_spend", "revenue")
        assert not cuped_res.significant, f"unexpected significance on {seg} segment"


# --- SRM -------------------------------------------------------------------
def test_srm_flags_tampered_data(tampered_df):
    res = guardrails.srm_check(tampered_df)
    assert not res.passed
    assert res.p_value < 0.01


def test_srm_passes_clean_data(demo_df):
    res = guardrails.srm_check(demo_df)
    assert res.passed
    assert res.p_value >= 0.01


def test_guardrail_metrics_quiet_on_clean_data(demo_df):
    g = guardrails.guardrail_checks(demo_df)
    assert not g["significant_adj"].any(), "guardrails should not move on clean data"


# --- design ----------------------------------------------------------------
def test_sample_size_decreases_with_larger_mde():
    small_mde = design.sample_size_proportions(0.10, 0.02)
    large_mde = design.sample_size_proportions(0.10, 0.10)
    assert small_mde.total_n > large_mde.total_n > 0


def test_sample_size_means_matches_statsmodels():
    mine = design.sample_size_means(sd=20.0, mde_relative=0.05, baseline_mean=8.0)
    sm = design.sample_size_means_statsmodels(sd=20.0, delta=8.0 * 0.05)
    assert abs(mine.total_n - 2 * sm) / (2 * sm) < 0.05


def test_runtime_days_sanity():
    assert design.runtime_days(10_000, 1_000) == 10
    assert design.runtime_days(10_001, 1_000) == 11
    with pytest.raises(ValueError):
        design.runtime_days(100, 0)


# --- corrections ------------------------------------------------------------
def test_holm_rejects_only_strong_signal():
    flags = analysis.holm([0.001, 0.04, 0.30], alpha=0.05)
    assert flags == [True, False, False]


def test_segment_drilldown_flags_returning_only(demo_df):
    drill = analysis.segment_drilldown(demo_df, "converted", binary=True)
    assert set(drill["segment"]) == {"new", "returning", "mobile"}
    # conversion has no planted lift anywhere -> nothing significant after Holm
    assert not drill["significant_adj"].any()
