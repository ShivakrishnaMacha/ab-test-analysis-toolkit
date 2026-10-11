"""Generate the demo experiment dataset.

Writes data/demo_experiment.csv — ~20k users with known planted effects:
+3.5% revenue lift on returning users, null on the other segments.
SRM-clean by default; pass --tamper to inject a sample-ratio mismatch
for the guardrail demo.

Usage:
    python data/make_demo.py
    python data/make_demo.py --tamper --out data/demo_experiment_tampered.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from abtoolkit.generator import generate_experiment

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo experiment data")
    parser.add_argument("--n", type=int, default=20000, help="number of users")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--tamper",
        action="store_true",
        help="inject sample-ratio mismatch into variant assignment",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/demo_experiment.csv",
        help="output path relative to repo root",
    )
    args = parser.parse_args()

    df = generate_experiment(n_users=args.n, seed=args.seed, tamper_srm=args.tamper)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    treat = int((df["variant"] == "treatment").sum())
    print(f"wrote {out_path} ({len(df):,} rows)")
    print(f"  treatment share: {treat / len(df):.3f} (tampered={args.tamper})")
    for seg, sub in df.groupby("segment"):
        c = sub[sub["variant"] == "control"]["revenue"].mean()
        t = sub[sub["variant"] == "treatment"]["revenue"].mean()
        lift = (t - c) / c if c else float("nan")
        print(f"  segment={seg:<10} n={len(sub):>6,}  rev lift={lift:+.2%}")


if __name__ == "__main__":
    main()
