"""
02_baseline.py

Establish baseline patterns for workers' compensation claims and providers.

The script:
    1. Loads simulated claims and related entities.
    2. Builds claim-level and provider-level features.
    3. Establishes normal-behavior reference statistics using legitimate claims.
    4. Identifies statistically unusual claims and providers.
    5. Evaluates baseline indicators against known simulated fraud labels.
    6. Saves baseline statistics and reports for subsequent graph analysis.

Expected input files:
    data/processed/claims.csv
    data/processed/claimants.csv
    data/processed/employers.csv
    data/processed/providers.csv
    data/processed/treatments.csv
    data/processed/provider_referrals.csv

If processed files are unavailable, the script falls back to data/raw/.

Run from the repository root:
    python src/02_baseline.py

Dependencies:
    pandas
    numpy
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = PROCESSED_DIR / "baseline"

FRAUD_COLUMN = "is_fraud"
CLAIM_ID_COLUMN = "claim_id"
PROVIDER_ID_COLUMN = "provider_id"
CLAIMANT_ID_COLUMN = "claimant_id"
EMPLOYER_ID_COLUMN = "employer_id"
CLAIM_AMOUNT_COLUMN = "claim_amount"

# Require enough observations to calculate meaningful provider statistics.
MIN_PROVIDER_CLAIMS = 5

# A robust deviation threshold for flagging unusual observations.
# This is a screening threshold, not a fraud determination.
ROBUST_Z_THRESHOLD = 3.5


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_table(table_name: str) -> pd.DataFrame:
    """Load a processed CSV, falling back to the raw CSV."""
    processed_path = PROCESSED_DIR / f"{table_name}.csv"
    raw_path = RAW_DIR / f"{table_name}.csv"

    if processed_path.exists():
        path = processed_path
    elif raw_path.exists():
        path = raw_path
    else:
        raise FileNotFoundError(
            f"Could not find {table_name}.csv in either:\n"
            f"  {processed_path}\n"
            f"  {raw_path}"
        )

    df = pd.read_csv(path)
    print(f"Loaded {table_name}: {len(df):,} rows from {path}")
    return df


def load_data() -> dict[str, pd.DataFrame]:
    """Load the available entity and relationship tables."""
    tables = {}

    for name in [
        "claims",
        "claimants",
        "employers",
        "providers",
        "treatments",
        "provider_referrals",
    ]:
        try:
            tables[name] = load_table(name)
        except FileNotFoundError:
            if name == "claims":
                raise

            # Some analyses can proceed without optional entity tables.
            print(f"Optional table unavailable: {name}.csv")

    return tables


# ---------------------------------------------------------------------
# Validation and utility functions
# ---------------------------------------------------------------------

def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    table_name: str,
) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"{table_name}.csv is missing required columns: {missing}"
        )


def normalize_boolean(series: pd.Series) -> pd.Series:
    """Convert common boolean representations to nullable booleans."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map(mapping).astype("boolean")


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column, or NaN when the column is unavailable."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    return pd.to_numeric(df[column], errors="coerce")


def robust_z_score(series: pd.Series) -> pd.Series:
    """
    Calculate a median/MAD-based robust z-score.

    Unlike the ordinary z-score, this is less sensitive to extreme values.

    Returns NaN if the median absolute deviation is zero or unavailable.
    """
    values = pd.to_numeric(series, errors="coerce")

    median = values.median()
    mad = (values - median).abs().median()

    if pd.isna(mad) or mad == 0:
        return pd.Series(np.nan, index=series.index)

    return 0.6745 * (values - median) / mad


def write_report(df: pd.DataFrame, filename: str) -> None:
    """Write a CSV report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)

    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------
# Claim-level baseline
# ---------------------------------------------------------------------

def build_claim_features(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build one row per claim with available claim-level features.

    Features include:
        - Claim amount
        - Claims associated with the same claimant
        - Claims associated with the same employer
        - Claims associated with the same provider
        - Treatment counts, when treatment records can be linked to claims
    """
    claims = tables["claims"].copy()

    require_columns(
        claims,
        [CLAIM_ID_COLUMN],
        "claims",
    )

    if claims[CLAIM_ID_COLUMN].duplicated().any():
        raise ValueError(
            "claims.csv contains duplicate claim_id values. "
            "Resolve duplicate claims before establishing a baseline."
        )

    claims[CLAIM_AMOUNT_COLUMN] = numeric_column(
        claims, CLAIM_AMOUNT_COLUMN
    )

    # Count claims associated with each claimant.
    if CLAIMANT_ID_COLUMN in claims.columns:
        claimant_counts = (
            claims.groupby(CLAIMANT_ID_COLUMN, dropna=False)
            .size()
            .rename("claimant_claim_count")
        )

        claims = claims.join(
            claimant_counts,
            on=CLAIMANT_ID_COLUMN,
        )

    # Count claims associated with each employer.
    if EMPLOYER_ID_COLUMN in claims.columns:
        employer_counts = (
            claims.groupby(EMPLOYER_ID_COLUMN, dropna=False)
            .size()
            .rename("employer_claim_count")
        )

        claims = claims.join(
            employer_counts,
            on=EMPLOYER_ID_COLUMN,
        )

    # Count claims associated with each provider.
    if PROVIDER_ID_COLUMN in claims.columns:
        provider_counts = (
            claims.groupby(PROVIDER_ID_COLUMN, dropna=False)
            .size()
            .rename("provider_claim_count")
        )

        claims = claims.join(
            provider_counts,
            on=PROVIDER_ID_COLUMN,
        )

    # Incorporate treatment counts if treatments are recorded per claim.
    treatments = tables.get("treatments")

    if treatments is not None and CLAIM_ID_COLUMN in treatments.columns:
        treatment_counts = (
            treatments.dropna(subset=[CLAIM_ID_COLUMN])
            .groupby(CLAIM_ID_COLUMN)
            .size()
            .rename("treatment_record_count")
        )

        claims = claims.join(
            treatment_counts,
            on=CLAIM_ID_COLUMN,
        )

        claims["treatment_record_count"] = (
            claims["treatment_record_count"].fillna(0).astype(int)
        )

    return claims


def summarize_claims(
    claims: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate descriptive statistics for claim-level features."""
    feature_columns = [
        column
        for column in [
            CLAIM_AMOUNT_COLUMN,
            "claimant_claim_count",
            "employer_claim_count",
            "provider_claim_count",
            "treatment_record_count",
        ]
        if column in claims.columns
    ]

    records = []

    for column in feature_columns:
        values = pd.to_numeric(claims[column], errors="coerce").dropna()

        if values.empty:
            continue

        records.append(
            {
                "feature": column,
                "count": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "median": values.median(),
                "p90": values.quantile(0.90),
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
                "max": values.max(),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Provider-level baseline
# ---------------------------------------------------------------------

def build_provider_features(
    tables: dict[str, pd.DataFrame],
    claims: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per provider.

    Measures claim volume, payout concentration, claim amounts, and
    provider-to-provider referral connections when available.
    """
    provider_columns = []

    if PROVIDER_ID_COLUMN in claims.columns:
        provider_claims = claims.dropna(
            subset=[PROVIDER_ID_COLUMN]
        ).copy()

        provider_summary = (
            provider_claims.groupby(PROVIDER_ID_COLUMN)
            .agg(
                claim_count=(CLAIM_ID_COLUMN, "nunique"),
                total_claim_amount=(CLAIM_AMOUNT_COLUMN, "sum"),
                mean_claim_amount=(CLAIM_AMOUNT_COLUMN, "mean"),
                median_claim_amount=(CLAIM_AMOUNT_COLUMN, "median"),
                max_claim_amount=(CLAIM_AMOUNT_COLUMN, "max"),
            )
            .reset_index()
        )

        provider_columns.append(provider_summary)

    else:
        provider_summary = pd.DataFrame(
            columns=[
                PROVIDER_ID_COLUMN,
                "claim_count",
                "total_claim_amount",
                "mean_claim_amount",
                "median_claim_amount",
                "max_claim_amount",
            ]
        )

    providers = tables.get("providers")

    if providers is not None and PROVIDER_ID_COLUMN in providers.columns:
        if providers[PROVIDER_ID_COLUMN].duplicated().any():
            raise ValueError(
                "providers.csv contains duplicate provider_id values."
            )

        # Preserve providers with no observed claims.
        provider_summary = providers.merge(
            provider_summary,
            on=PROVIDER_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )

    # A missing claim count means that the provider has no observed claims.
    if "claim_count" in provider_summary.columns:
        provider_summary["claim_count"] = (
            provider_summary["claim_count"].fillna(0).astype(int)
        )

    # Summarize referral-network connectivity.
    referrals = tables.get("provider_referrals")

    if referrals is not None:
        source_column = PROVIDER_ID_COLUMN
        target_column = "referred_provider_id"

        if {
            source_column,
            target_column,
        }.issubset(referrals.columns):

            # Count outgoing referral records for each provider.
            outgoing = (
                referrals.dropna(subset=[source_column])
                .groupby(source_column)
                .size()
                .rename("outgoing_referral_count")
            )

            provider_summary = provider_summary.join(
                outgoing,
                on=PROVIDER_ID_COLUMN,
            )

            # Count unique destinations, which differs from raw referral count.
            unique_outgoing = (
                referrals.dropna(
                    subset=[source_column, target_column]
                )
                .groupby(source_column)[target_column]
                .nunique()
                .rename("unique_referred_providers")
            )

            provider_summary = provider_summary.join(
                unique_outgoing,
                on=PROVIDER_ID_COLUMN,
            )

            # Count distinct incoming referral sources.
            incoming = (
                referrals.dropna(
                    subset=[source_column, target_column]
                )
                .groupby(target_column)[source_column]
                .nunique()
                .rename("unique_referring_providers")
            )

            provider_summary = provider_summary.join(
                incoming,
                on=PROVIDER_ID_COLUMN,
            )

    return provider_summary


def summarize_providers(
    providers: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize provider activity and network characteristics."""
    metrics = [
        "claim_count",
        "total_claim_amount",
        "mean_claim_amount",
        "median_claim_amount",
        "max_claim_amount",
        "outgoing_referral_count",
        "unique_referred_providers",
        "unique_referring_providers",
    ]

    records = []

    for column in metrics:
        if column not in providers.columns:
            continue

        values = pd.to_numeric(
            providers[column], errors="coerce"
        ).dropna()

        if values.empty:
            continue

        records.append(
            {
                "feature": column,
                "count": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "median": values.median(),
                "p90": values.quantile(0.90),
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
                "max": values.max(),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Baseline construction and anomaly screening
# ---------------------------------------------------------------------

def add_baseline_scores(
    claims: pd.DataFrame,
    providers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate anomaly indicators relative to legitimate observations.

    Fraud labels are excluded when calculating reference medians and MADs.
    Providers associated with known fraudulent claims are also excluded
    from the provider reference population when the labels are available.
    """
    claims = claims.copy()
    providers = providers.copy()

    # Use only claims with an explicitly observed legitimate label.
    if FRAUD_COLUMN in claims.columns:
        fraud_labels = normalize_boolean(claims[FRAUD_COLUMN])
        legitimate_mask = fraud_labels.eq(False).fillna(False)
        legitimate_claims = claims.loc[legitimate_mask].copy()
    else:
        legitimate_claims = claims.copy()
        print(
            "WARNING: No fraud labels found. "
            "All claims will be used to construct the reference baseline."
        )

    claim_metrics = [
        CLAIM_AMOUNT_COLUMN,
        "claimant_claim_count",
        "employer_claim_count",
        "provider_claim_count",
        "treatment_record_count",
    ]

    claim_metrics = [
        column
        for column in claim_metrics
        if column in claims.columns
    ]

    for column in claim_metrics:
        reference = pd.to_numeric(
            legitimate_claims[column], errors="coerce"
        ).dropna()

        if reference.empty:
            continue

        median = reference.median()
        mad = (reference - median).abs().median()

        score_column = f"{column}_robust_z"

        if pd.isna(mad) or mad == 0:
            # A zero MAD makes a robust z-score undefined.
            # Use the reference quantile for a transparent fallback.
            upper_limit = reference.quantile(0.99)

            claims[score_column] = np.nan

            if pd.notna(upper_limit):
                claims[score_column] = (
                    pd.to_numeric(claims[column], errors="coerce")
                    .gt(upper_limit)
                    .astype(float)
                )
        else:
            claims[score_column] = (
                0.6745
                * (
                    pd.to_numeric(
                        claims[column], errors="coerce"
                    )
                    - median
                )
                / mad
            )

    # Construct provider reference population.
    if FRAUD_COLUMN in claims.columns and PROVIDER_ID_COLUMN in claims.columns:
        fraud_labels = normalize_boolean(claims[FRAUD_COLUMN])

        fraudulent_provider_ids = set(
            claims.loc[
                fraud_labels.eq(True).fillna(False),
                PROVIDER_ID_COLUMN,
            ].dropna()
        )

        legitimate_provider_ids = set(
            claims.loc[
                fraud_labels.eq(False).fillna(False),
                PROVIDER_ID_COLUMN,
            ].dropna()
        )

        # If a provider has any known fraudulent claim, exclude it from
        # the provider reference population.
        legitimate_provider_ids -= fraudulent_provider_ids

        reference_providers = providers[
            providers[PROVIDER_ID_COLUMN].isin(legitimate_provider_ids)
        ].copy()
    else:
        reference_providers = providers.copy()

    provider_metrics = [
        "claim_count",
        "total_claim_amount",
        "mean_claim_amount",
        "outgoing_referral_count",
        "unique_referred_providers",
        "unique_referring_providers",
    ]

    provider_metrics = [
        column
        for column in provider_metrics
        if column in providers.columns
    ]

    for column in provider_metrics:
        reference = pd.to_numeric(
            reference_providers[column], errors="coerce"
        ).dropna()

        if reference.empty:
            continue

        median = reference.median()
        mad = (reference - median).abs().median()

        score_column = f"{column}_robust_z"

        if pd.isna(mad) or mad == 0:
            # Preserve an explicit fallback when the reference metric
            # is too concentrated for a robust z-score.
            upper_limit = reference.quantile(0.99)

            providers[score_column] = np.nan

            if pd.notna(upper_limit):
                providers[score_column] = (
                    pd.to_numeric(providers[column], errors="coerce")
                    .gt(upper_limit)
                    .astype(float)
                )
        else:
            providers[score_column] = (
                0.6745
                * (
                    pd.to_numeric(
                        providers[column], errors="coerce"
                    )
                    - median
                )
                / mad
            )

    return claims, providers


def flag_anomalies(
    claims: pd.DataFrame,
    providers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag observations with unusually high robust deviation scores."""
    claims = claims.copy()
    providers = providers.copy()

    claim_score_columns = [
        column
        for column in claims.columns
        if column.endswith("_robust_z")
    ]

    provider_score_columns = [
        column
        for column in providers.columns
        if column.endswith("_robust_z")
    ]

    if claim_score_columns:
        claims["anomaly_indicator_count"] = (
            claims[claim_score_columns]
            .gt(ROBUST_Z_THRESHOLD)
            .sum(axis=1)
        )

        claims["baseline_anomaly_flag"] = (
            claims["anomaly_indicator_count"] > 0
        )

    if provider_score_columns:
        # Ignore providers without enough claims for a meaningful
        # claim-volume-based comparison.
        eligible = (
            providers["claim_count"].ge(MIN_PROVIDER_CLAIMS)
            if "claim_count" in providers.columns
            else pd.Series(True, index=providers.index)
        )

        providers["anomaly_indicator_count"] = (
            providers[provider_score_columns]
            .gt(ROBUST_Z_THRESHOLD)
            .sum(axis=1)
        )

        providers["baseline_anomaly_flag"] = (
            eligible
            & providers["anomaly_indicator_count"].gt(0)
        )

    return claims, providers


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_claim_indicators(
    claims: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate anomaly flags against known simulated fraud labels.

    This is an evaluation of the baseline's screening capability, not
    a claim that the indicators establish actual fraud.
    """
    if FRAUD_COLUMN not in claims.columns:
        print(
            "Skipping fraud evaluation: claims.csv has no is_fraud column."
        )
        return pd.DataFrame()

    labels = normalize_boolean(claims[FRAUD_COLUMN])
    valid = labels.notna()

    if not valid.any():
        print("Skipping fraud evaluation: fraud labels are unavailable.")
        return pd.DataFrame()

    actual = labels.loc[valid].astype(bool)

    records = []

    for prediction_column in [
        "baseline_anomaly_flag",
    ]:
        if prediction_column not in claims.columns:
            continue

        predicted = (
            claims.loc[valid, prediction_column]
            .fillna(False)
            .astype(bool)
        )

        true_positive = int((predicted & actual).sum())
        false_positive = int((predicted & ~actual).sum())
        false_negative = int((~predicted & actual).sum())
        true_negative = int((~predicted & ~actual).sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else np.nan
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else np.nan
        )

        specificity = (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive > 0
            else np.nan
        )

        records.append(
            {
                "indicator": prediction_column,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "fraud_prevalence": actual.mean(),
                "flagged_rate": predicted.mean(),
            }
        )

    return pd.DataFrame(records)


def evaluate_feature_distributions(
    claims: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare legitimate and fraudulent claim distributions.

    This report is descriptive: it helps identify which features may
    be useful for subsequent graph-based investigation.
    """
    if FRAUD_COLUMN not in claims.columns:
        return pd.DataFrame()

    labels = normalize_boolean(claims[FRAUD_COLUMN])

    feature_columns = [
        column
        for column in [
            CLAIM_AMOUNT_COLUMN,
            "claimant_claim_count",
            "employer_claim_count",
            "provider_claim_count",
            "treatment_record_count",
        ]
        if column in claims.columns
    ]

    records = []

    for column in feature_columns:
        values = pd.to_numeric(claims[column], errors="coerce")

        legitimate = values.loc[labels.eq(False).fillna(False)].dropna()
        fraudulent = values.loc[labels.eq(True).fillna(False)].dropna()

        if legitimate.empty or fraudulent.empty:
            continue

        legitimate_median = legitimate.median()
        fraudulent_median = fraudulent.median()

        records.append(
            {
                "feature": column,
                "legitimate_count": len(legitimate),
                "fraudulent_count": len(fraudulent),
                "legitimate_mean": legitimate.mean(),
                "fraudulent_mean": fraudulent.mean(),
                "legitimate_median": legitimate_median,
                "fraudulent_median": fraudulent_median,
                "median_difference": (
                    fraudulent_median - legitimate_median
                ),
                "fraudulent_to_legitimate_median_ratio": (
                    fraudulent_median / legitimate_median
                    if legitimate_median != 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Run the baseline analysis and save all reports."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("WORKERS' COMPENSATION FRAUD DETECTION")
    print("BASELINE ANALYSIS")
    print("=" * 70 + "\n")

    tables = load_data()

    claims = build_claim_features(tables)

    print("\nBuilding claim-level baseline...")
    claim_summary = summarize_claims(claims)
    write_report(claim_summary, "claim_baseline_summary.csv")

    print("\nBuilding provider-level baseline...")
    providers = build_provider_features(tables, claims)
    provider_summary = summarize_providers(providers)
    write_report(provider_summary, "provider_baseline_summary.csv")

    print("\nCalculating anomaly indicators...")
    claims, providers = add_baseline_scores(claims, providers)
    claims, providers = flag_anomalies(claims, providers)

    write_report(claims, "claim_baseline_features.csv")
    write_report(providers, "provider_baseline_features.csv")

    print("\nEvaluating baseline indicators...")
    evaluation = evaluate_claim_indicators(claims)
    if not evaluation.empty:
        write_report(evaluation, "baseline_evaluation.csv")
        print(evaluation.to_string(index=False))

    feature_comparison = evaluate_feature_distributions(claims)
    if not feature_comparison.empty:
        write_report(
            feature_comparison,
            "fraud_feature_comparison.csv",
        )

    print("\n" + "=" * 70)
    print("BASELINE ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nReports saved to: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(
        "\nNext step: use these reference patterns to develop "
        "graph-based features and investigate suspicious provider networks."
    )


if __name__ == "__main__":
    main()
