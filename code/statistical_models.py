"""Fit the manuscript's core building-level and joint policy logit models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


REBUILT = {"rebuilt_equal", "rebuilt_improved", "rebuilt_degraded"}
VALID_TRAJECTORIES = REBUILT | {"empty_lot", "obscured"}

BASELINE = [
    ("log_income", "log tract household income"),
    ("pre_pct_Hispanic", "tract Hispanic share"),
    ("pre_pct_below_poverty", "tract below-poverty share"),
    ("pre_pct_owner_occupied", "tract owner-occupied share"),
    ("pre_pct_BA_or_higher", "tract bachelor-or-higher share"),
    ("log_assessed_value", "log assessed structure value"),
    ("mtbs_severity_code", "MTBS burn severity"),
]

POLICY = [
    ("nonrenew_rate_2020_2023", "cumulative non-renewal rate"),
    ("fair_pct_2022", "FAIR Plan exposure"),
    ("hmda_denial_rate", "HMDA denial rate"),
    ("insurance_proxy", "ACS insurance proxy"),
    ("log_fema_ihp", "FEMA IHP funding"),
    ("j40_disadvantaged", "CEJST disadvantage status"),
    ("log_income", "log tract income"),
    ("mtbs_severity_code", "MTBS burn severity"),
]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fire_id",
        "state_fips",
        "trajectory",
        "pre_tract_med_HH_income",
        "pre_pct_Hispanic",
        "pre_pct_below_poverty",
        "pre_pct_owner_occupied",
        "pre_pct_BA_or_higher",
        "assessed_value_usd",
        "mtbs_severity_code",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    frame = frame.copy()
    frame["state_fips"] = (
        frame["state_fips"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )
    frame["trajectory"] = (
        frame["trajectory"].astype("string").str.strip().str.lower()
    )
    frame["trajectory"] = frame["trajectory"].replace({"uncertain": "obscured"})
    unknown = sorted(
        str(value)
        for value in frame["trajectory"].dropna().unique()
        if value not in VALID_TRAJECTORIES
    )
    if unknown:
        raise ValueError("Unexpected trajectory labels: " + ", ".join(unknown))
    if frame["trajectory"].isna().any():
        raise ValueError("trajectory contains missing values")
    if frame["fire_id"].isna().any() or frame["state_fips"].isna().any():
        raise ValueError("fire_id and state_fips cannot contain missing values")
    frame["rebuilt"] = frame["trajectory"].isin(REBUILT).astype(int)
    frame["upgraded"] = frame["trajectory"].eq("rebuilt_improved").astype(int)
    income = pd.to_numeric(frame["pre_tract_med_HH_income"], errors="coerce")
    assessed_value = pd.to_numeric(frame["assessed_value_usd"], errors="coerce")
    if income.lt(0).any() or assessed_value.lt(0).any():
        raise ValueError("Income and assessed structure value cannot be negative")
    frame["log_income"] = np.log1p(income)
    frame["log_assessed_value"] = np.log1p(assessed_value)
    if "fema_ihp_total_usd" in frame:
        fema_ihp = pd.to_numeric(frame["fema_ihp_total_usd"], errors="coerce")
        if fema_ihp.lt(0).any():
            raise ValueError("FEMA IHP funding cannot be negative")
        frame["log_fema_ihp"] = np.log1p(fema_ihp.fillna(0))
    return frame.replace([np.inf, -np.inf], np.nan)


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    standard_deviation = numeric.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        raise ValueError(f"Cannot standardize {series.name}")
    return (numeric - numeric.mean()) / standard_deviation


def fit_logit(
    frame: pd.DataFrame,
    outcome: str,
    variables: list[tuple[str, str]],
    *,
    cluster: str | None = None,
    standardize_before_deletion: bool = False,
) -> pd.DataFrame:
    names = [name for name, _ in variables]
    needed = names + [outcome] + ([cluster] if cluster else [])
    working = frame[needed].copy()
    if standardize_before_deletion:
        for name in names:
            working[name] = zscore(working[name])
        sample = working.dropna()
        design = sample[names]
    else:
        sample = working.dropna()
        design = sample[names].apply(zscore)
    if sample.empty:
        raise ValueError(f"No complete cases remain for outcome {outcome}")
    if sample[outcome].nunique() != 2:
        raise ValueError(f"Outcome {outcome} must contain both 0 and 1")
    design = sm.add_constant(design.astype(float), has_constant="add")
    model = sm.Logit(sample[outcome].astype(int), design)
    if cluster:
        result = model.fit(
            disp=False,
            method="bfgs",
            maxiter=500,
            cov_type="cluster",
            cov_kwds={"groups": sample[cluster]},
        )
    else:
        result = model.fit(disp=False, maxiter=500)

    confidence = result.conf_int()
    labels = dict(variables)
    rows = []
    for name in names:
        rows.append(
            {
                "variable": labels[name],
                "coefficient": result.params[name],
                "standard_error": result.bse[name],
                "coefficient_ci_low": confidence.loc[name, 0],
                "coefficient_ci_high": confidence.loc[name, 1],
                "odds_ratio": np.exp(result.params[name]),
                "odds_ratio_ci_low": np.exp(confidence.loc[name, 0]),
                "odds_ratio_ci_high": np.exp(confidence.loc[name, 1]),
                "p_value": result.pvalues[name],
                "N": int(result.nobs),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame = prepare(read_table(args.data))
    args.output.mkdir(parents=True, exist_ok=True)

    fit_logit(frame, "rebuilt", BASELINE).to_csv(
        args.output / "baseline_rebuilding_model.csv", index=False
    )
    fit_logit(frame.loc[frame["rebuilt"].eq(1)], "upgraded", BASELINE).to_csv(
        args.output / "baseline_upgrading_model.csv", index=False
    )

    policy_fields = {name for name, _ in POLICY}
    missing_policy = sorted(policy_fields - set(frame.columns))
    if missing_policy:
        raise ValueError("Missing joint-policy fields: " + ", ".join(missing_policy))
    california = frame.loc[frame["state_fips"].eq("06")]
    fit_logit(
        california,
        "rebuilt",
        POLICY,
        cluster="fire_id",
        standardize_before_deletion=True,
    ).to_csv(args.output / "joint_policy_model.csv", index=False)
    print(f"Wrote model tables to {args.output}")


if __name__ == "__main__":
    main()
