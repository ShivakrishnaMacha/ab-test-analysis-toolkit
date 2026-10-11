"""Streamlit demo for the A/B test analysis toolkit.

Three sections (sidebar navigation):
  Design     — size the experiment: N per arm + runtime from MDE/power/alpha
  Analyze    — naive t-test vs CUPED-adjusted analysis, segment drilldown
  Guardrails — SRM check + guardrail metrics with Holm correction
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from abtoolkit import analysis, design, generator, guardrails, variance  # noqa: E402

st.set_page_config(page_title="A/B Test Analysis Toolkit", layout="wide")

DEMO_CSV = Path(__file__).resolve().parent / "data" / "demo_experiment.csv"


@st.cache_data
def load_demo() -> pd.DataFrame:
    if DEMO_CSV.exists():
        return pd.read_csv(DEMO_CSV)
    # fall back to in-memory generation (same seed => same data)
    return generator.generate_experiment()


@st.cache_data
def load_tampered() -> pd.DataFrame:
    return generator.generate_experiment(tamper_srm=True)


def result_cards(res: analysis.TestResult, title: str) -> None:
    st.subheader(title)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Relative lift", f"{res.rel_lift:+.2%}")
    c2.metric("Absolute diff", f"{res.diff:+.4f}")
    c3.metric("95% CI", f"[{res.ci_low:+.4f}, {res.ci_high:+.4f}]")
    c4.metric("p-value", f"{res.p_value:.4g}")
    badge = "Significant" if res.significant else "Not significant"
    st.markdown(f"**Verdict:** `{res.test}` → {badge} (alpha={res.alpha})")


def ci_bar_chart(drilldown: pd.DataFrame, metric_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=drilldown["segment"],
            y=drilldown["treatment_mean"],
            name="Treatment",
            error_y=dict(
                type="data",
                symmetric=False,
                array=drilldown["ci_high"] - drilldown["diff"],
                arrayminus=drilldown["diff"] - drilldown["ci_low"],
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=drilldown["segment"],
            y=drilldown["control_mean"],
            name="Control",
        )
    )
    fig.update_layout(
        title=f"{metric_label} by segment (treatment CI = diff CI)",
        barmode="group",
        xaxis_title="Segment",
        yaxis_title=metric_label,
    )
    return fig


# ---------------------------------------------------------------- sidebar ----
st.sidebar.title("A/B Test Toolkit")
page = st.sidebar.radio("Navigate", ["Design", "Analyze", "Guardrails"])
st.sidebar.markdown("---")
st.sidebar.caption(
    "Synthetic demo data: 20k users, +3.5% revenue lift planted on "
    "returning users only. SRM-clean unless tampered."
)

# ------------------------------------------------------------------ DESIGN ---
if page == "Design":
    st.header("Experiment design — sample size & runtime")
    st.markdown(
        "Size the experiment **before** it runs: pick the metric, the minimum "
        "detectable effect (MDE), power and allocation, and get the required "
        "sample plus a calendar runtime estimate."
    )

    metric_type = st.selectbox("Metric type", ["Conversion rate", "Revenue per user"])
    col_a, col_b = st.columns(2)
    with col_a:
        power = st.slider("Power (1 - beta)", 0.5, 0.99, 0.8, 0.01)
        alpha = st.selectbox("Significance level (alpha)", [0.10, 0.05, 0.01], index=1)
    with col_b:
        allocation = st.slider("Allocation (treatment / control)", 0.25, 4.0, 1.0, 0.25)
        daily_traffic = st.number_input("Daily traffic (users/day)", 100, 1_000_000, 5000, 100)

    if metric_type == "Conversion rate":
        baseline = st.number_input("Baseline conversion rate", 0.001, 0.99, 0.12, 0.005,
                                   format="%.3f")
        mde = st.slider("MDE (relative lift)", 0.01, 0.50, 0.05, 0.01)
        if st.button("Calculate sample size"):
            res = design.sample_size_proportions(
                baseline, mde, alpha=alpha, power=power,
                allocation=allocation, daily_traffic=int(daily_traffic),
            )
            st.success("Done — run this long before peeking at significance.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Control N", f"{res.n_control:,}")
            m2.metric("Treatment N", f"{res.n_treatment:,}")
            m3.metric("Total N", f"{res.total_n:,}")
            m4.metric("Runtime", f"{res.runtime_days} days")
    else:
        baseline_mean = st.number_input("Baseline mean revenue/user ($)", 0.1, 1000.0, 8.0, 0.5)
        sd = st.number_input("Std dev of revenue/user ($)", 0.1, 500.0, 20.0, 0.5)
        mde = st.slider("MDE (relative lift)", 0.01, 0.50, 0.05, 0.01)
        if st.button("Calculate sample size"):
            res = design.sample_size_means(
                sd, mde, baseline_mean, alpha=alpha, power=power,
                allocation=allocation, daily_traffic=int(daily_traffic),
            )
            st.success("Done — CUPED variance reduction would shrink this further.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Control N", f"{res.n_control:,}")
            m2.metric("Treatment N", f"{res.n_treatment:,}")
            m3.metric("Total N", f"{res.total_n:,}")
            m4.metric("Runtime", f"{res.runtime_days} days")

# ----------------------------------------------------------------- ANALYZE ---
elif page == "Analyze":
    st.header("Analyze — naive test vs CUPED")
    st.markdown(
        "Compare the plain two-sample test against CUPED variance reduction. "
        "The demo data has a **+3.5% revenue lift planted on returning users only**."
    )

    src = st.radio("Data source", ["Demo dataset", "Upload CSV"], horizontal=True)
    if src == "Upload CSV":
        up = st.file_uploader("Experiment CSV", type=["csv"])
        if up is None:
            st.info("Upload a CSV with variant/segment/pre_spend/revenue/converted columns.")
            st.stop()
        df = pd.read_csv(up)
    else:
        df = load_demo()
        st.caption(f"Loaded demo data: {len(df):,} users from `{DEMO_CSV.name}`")

    metric = st.selectbox("Metric", ["revenue", "converted"])
    segment = st.selectbox("Segment", ["all", "new", "returning", "mobile"])
    method = st.selectbox("Test", ["t-test", "Mann-Whitney", "Chi-square (binary)"])

    work = df if segment == "all" else df[df["segment"] == segment]
    st.caption(f"Analyzing {len(work):,} users ({segment} segment)")

    binary = metric == "converted" or method.startswith("Chi-square")
    method_key = {"t-test": "t", "Mann-Whitney": "mannwhitney",
                  "Chi-square (binary)": "chi2"}[method]
    naive = analysis.analyze_metric(work, metric, method=method_key, binary=binary)

    col1, col2 = st.columns(2)
    with col1:
        result_cards(naive, "Naive analysis")
    with col2:
        if metric == "revenue":
            cuped_res, diag = variance.cuped_ttest(work, "pre_spend", "revenue")
            result_cards(cuped_res, "CUPED-adjusted analysis")
            st.markdown(
                f"CUPED diagnostics: theta={diag.theta:.3f}, "
                f"pre/post correlation={diag.correlation:.3f}, "
                f"variance reduction={diag.variance_reduction:.1%}"
            )
            shrink = 1 - cuped_res.ci_width / naive.ci_width
            st.markdown(f"CI width shrank by **{shrink:.1%}** vs the naive test.")
        else:
            st.subheader("CUPED-adjusted analysis")
            st.info("CUPED needs a continuous metric + pre-period covariate — "
                    "switch the metric to revenue.")

    st.subheader("Segment drilldown (Holm-adjusted)")
    drill = analysis.segment_drilldown(df, metric, binary=binary)
    st.dataframe(drill.style.format({
        "control_mean": "{:.4f}", "treatment_mean": "{:.4f}",
        "diff": "{:+.4f}", "rel_lift": "{:+.2%}",
        "ci_low": "{:+.4f}", "ci_high": "{:+.4f}", "p_value": "{:.4g}",
    }))
    st.plotly_chart(ci_bar_chart(drill, metric), use_container_width=True)

# --------------------------------------------------------------- GUARDRAILS ---
else:
    st.header("Guardrails — SRM & health metrics")
    st.markdown(
        "Trust checks that run **before** any result is read: is randomization "
        "intact (SRM), and did any guardrail metric move?"
    )

    src = st.radio("Data source", ["Demo (clean)", "Demo (tampered SRM)", "Upload CSV"],
                   horizontal=True)
    if src == "Demo (tampered SRM)":
        df = load_tampered()
        st.caption("Tampered dataset: variant assignment skewed to ~56/44.")
    elif src == "Upload CSV":
        up = st.file_uploader("Experiment CSV", type=["csv"])
        if up is None:
            st.info("Upload a CSV with a variant column.")
            st.stop()
        df = pd.read_csv(up)
    else:
        df = load_demo()

    st.subheader("1. Sample-ratio mismatch (SRM) check")
    srm = guardrails.srm_check(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Control N", f"{srm.n_control:,}")
    c2.metric("Treatment N", f"{srm.n_treatment:,}")
    c3.metric("Chi-square", f"{srm.chi2:.2f}")
    c4.metric("p-value", f"{srm.p_value:.4g}")
    if srm.passed:
        st.success("SRM check passed — observed allocation matches the intended 50/50.")
    else:
        st.error(
            "SRM CHECK FAILED — the observed split deviates from the intended "
            "allocation. Randomization or logging is broken: do not trust any "
            "result until this is explained."
        )

    st.subheader("2. Guardrail metrics (Holm-adjusted)")
    g = guardrails.guardrail_checks(df)
    st.dataframe(g.style.format({
        "control_mean": "{:.4f}", "treatment_mean": "{:.4f}",
        "diff": "{:+.4f}", "rel_lift": "{:+.2%}", "p_value": "{:.4g}",
    }))
    moved = g[g["significant_adj"]]
    if not moved.empty:
        st.warning(f"Guardrail(s) moved: {', '.join(moved['metric'])} — investigate.")
    else:
        st.success("No guardrail metric moved significantly.")
