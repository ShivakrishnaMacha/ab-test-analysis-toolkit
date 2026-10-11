"""Synthetic experiment data generator.

Produces a deterministic (seeded) e-commerce experiment dataset with known
planted effects so every analysis in this toolkit can be validated against
ground truth:

* +3.5% revenue lift on **returning** users (treatment vs control)
* null effect on new and mobile segments
* pre-period spend covariate strongly correlated with post-period revenue
  (conversion follows a pre-spend propensity threshold; post spend tracks
  pre spend — this is what makes CUPED variance reduction work)
* SRM-clean 50/50 assignment by default; ``tamper_srm=True`` skews the
  assignment so the guardrail demo has something to flag
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTS = ("new", "returning", "mobile")
_SEGMENT_PROBS = np.array([0.30, 0.50, 0.20])

_BASE_CONVERSION = {"new": 0.08, "returning": 0.22, "mobile": 0.10}
_PRE_SPEND_MU = {"new": 3.2, "returning": 3.9, "mobile": 3.5}  # lognormal mu
_PRE_SPEND_SIGMA = 0.6
_SPEND_MULTIPLIER = 1.25  # post-period spend tracks pre-period habit
_SPEND_NOISE_SIGMA = 0.05  # small multiplicative noise -> high pre/post correlation
_BOUNCE_RATE = {"new": 0.45, "returning": 0.32, "mobile": 0.50}
_PAGEVIEW_LAMBDA = {"new": 4.0, "returning": 7.0, "mobile": 5.0}

TREATMENT_LIFT = 0.035  # +3.5% revenue lift, returning users only
_TAMPERED_TREATMENT_PROB = 0.56  # skewed allocation to inject SRM


def generate_experiment(
    n_users: int = 20000,
    seed: int = 42,
    tamper_srm: bool = False,
) -> pd.DataFrame:
    """Generate a synthetic experiment dataset.

    Args:
        n_users: number of users (rows).
        seed: fixed seed for the numpy Generator (deterministic output).
        tamper_srm: if True, skew variant assignment (~56/44) to inject a
            sample-ratio mismatch for the guardrail demo.

    Returns:
        DataFrame with columns: user_id, variant, segment, pre_spend,
        revenue, converted, bounce, pageviews.
    """
    rng = np.random.default_rng(seed)

    user_id = np.array([f"user_{i:06d}" for i in range(n_users)])

    segment = rng.choice(SEGMENTS, size=n_users, p=_SEGMENT_PROBS)

    pre_spend = np.array(
        [
            rng.lognormal(_PRE_SPEND_MU[s], _PRE_SPEND_SIGMA)
            for s in segment
        ]
    )

    # conversion: propensity threshold on pre-period spend — the highest-spending
    # pre-period users convert (per-segment rate matches _BASE_CONVERSION).
    # This keeps pre/post correlation high so CUPED has signal to exploit.
    converted = np.zeros(n_users, dtype=int)
    for s in SEGMENTS:
        idx = np.where(segment == s)[0]
        threshold = np.quantile(pre_spend[idx], 1.0 - _BASE_CONVERSION[s])
        converted[idx] = (pre_spend[idx] >= threshold).astype(int)

    # variant assignment: SRM-clean 50/50 unless tampered
    treat_prob = _TAMPERED_TREATMENT_PROB if tamper_srm else 0.5
    variant = np.where(rng.random(n_users) < treat_prob, "treatment", "control")

    # post-period revenue: tracks pre-period spend; planted lift for returning
    # users in treatment only
    spend_noise = rng.lognormal(0.0, _SPEND_NOISE_SIGMA, n_users)
    revenue = converted * pre_spend * _SPEND_MULTIPLIER * spend_noise
    lifted = (variant == "treatment") & (segment == "returning")
    revenue = np.where(lifted, revenue * (1.0 + TREATMENT_LIFT), revenue)

    # guardrail metrics: identical distributions across variants (null)
    bounce = (
        rng.random(n_users)
        < np.array([_BOUNCE_RATE[s] for s in segment])
    ).astype(int)
    pageviews = np.array(
        [rng.poisson(_PAGEVIEW_LAMBDA[s]) for s in segment]
    )

    return pd.DataFrame(
        {
            "user_id": user_id,
            "variant": variant,
            "segment": segment,
            "pre_spend": np.round(pre_spend, 2),
            "revenue": np.round(revenue, 2),
            "converted": converted,
            "bounce": bounce,
            "pageviews": pageviews,
        }
    )
