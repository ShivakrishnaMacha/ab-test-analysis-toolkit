"""abtoolkit — industry-grade A/B testing analysis toolkit.

Modules:
    generator   synthetic experiment data with known planted effects
    design      sample-size / power calculations and runtime estimates
    analysis    t-test, chi-square, Mann-Whitney, Holm/Bonferroni corrections
    variance    CUPED variance reduction via pre-period covariates
    guardrails  SRM checks and guardrail-metric monitoring
"""

from . import analysis, design, generator, guardrails, variance

__version__ = "0.1.0"
__all__ = ["analysis", "design", "generator", "guardrails", "variance"]
