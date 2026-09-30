"""
00_simulate_data.py

Simulate workers' compensation claims data for fraud detection.

The goal is to create a small, realistic dataset that can eventually
be loaded into Neo4j and used to evaluate graph-based fraud detection.

We simulate two types of behavior:

1. Normal behavior
   - Claimants generally see a small number of providers.
   - Providers treat many different claimants.
   - Claims are connected to employers and treatments.
   - Providers may refer patients to other providers.

2. Fraudulent behavior
   - A small group of providers repeatedly appears together.
   - The same providers treat unusually many claimants together.
   - Fraudulent providers form a dense network.

The fraudulent behavior is intentionally planted so that we have
"ground truth" when we later evaluate our graph algorithms.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

N_CLAIMANTS = 2_000
N_CLAIMS = 2_500
N_PROVIDERS = 200
N_EMPLOYERS = 300
N_TREATMENTS = 20

# Number of providers deliberately involved in the simulated fraud ring.
N_FRAUD_PROVIDERS = 5

OUTPUT_DIR = Path("data/raw")


# ---------------------------------------------------------------------------
# Random number generator
# ---------------------------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def random_choice(values, size=None, probabilities=None):
    """
    Select random values from an array.

    Keeping this in a small helper makes the simulation code below
    easier to read.
    """
    return rng.choice(
        values,
        size=size,
        p=probabilities,
    )


# ---------------------------------------------------------------------------
# Simulate claimants
# ---------------------------------------------------------------------------

def simulate_claimants():
    """
    Create claimant records.

    Each claimant represents a person who has submitted at least one
    workers' compensation claim.
    """

    claimant_ids = [
        f"C{index:05d}"
        for index in range(1, N_CLAIMANTS + 1)
    ]

    claimants = pd.DataFrame(
        {
            "claimant_id": claimant_ids,
        }
    )

    return claimants


# ---------------------------------------------------------------------------
# Simulate employers
# ---------------------------------------------------------------------------

def simulate_employers():
    """
    Create employer records.
    """

    employer_ids = [
        f"E{index:04d}"
        for index in range(1, N_EMPLOYERS + 1)
    ]

    employers = pd.DataFrame(
        {
            "employer_id": employer_ids,
        }
    )

    return employers


# ---------------------------------------------------------------------------
# Simulate providers
# ---------------------------------------------------------------------------

def simulate_providers():
    """
    Create provider records.

    A small number of providers are explicitly marked as fraudulent.

    The fraud flag is NOT something we would have in real production
    data. It exists only so we can evaluate whether our algorithms
    successfully identify the planted fraud later.
    """

    provider_ids = [
        f"P{index:04d}"
        for index in range(1, N_PROVIDERS + 1)
    ]

    providers = pd.DataFrame(
        {
            "provider_id": provider_ids,
        }
    )

    # ------------------------------------------------------------------
    # Provider types
    # ------------------------------------------------------------------
    #
    # Most providers are normal.
    # A small group is part of the simulated fraud ring.
    #
    # We keep the fraud flag because this is simulated data.
    # It gives us ground truth for model evaluation.

    providers["is_fraud"] = False

    fraud_provider_ids = provider_ids[:N_FRAUD_PROVIDERS]

    providers.loc[
        providers["provider_id"].isin(fraud_provider_ids),
        "is_fraud",
    ] = True

    return providers


# ---------------------------------------------------------------------------
# Simulate treatments
# ---------------------------------------------------------------------------

def simulate_treatments():
    """
    Create a small dictionary of treatment types.

    In a real dataset these could come from structured claims data
    or extracted from unstructured medical records.
    """

    treatments = [
        "Physical Therapy",
        "Chiropractic",
        "MRI",
        "X-Ray",
        "Orthopedic Consultation",
        "Pain Management",
        "Occupational Therapy",
        "Surgery",
        "Follow-up Visit",
        "Diagnostic Testing",
        "Medication Management",
        "Neurology",
        "Primary Care",
        "Emergency Room",
        "Imaging",
        "Rehabilitation",
        "Injections",
        "Specialist Consultation",
        "Evaluation",
        "Home Health",
    ]

    treatment_ids = [
        f"T{index:03d}"
        for index in range(1, N_TREATMENTS + 1)
    ]

    return pd.DataFrame(
        {
            "treatment_id": treatment_ids,
            "treatment_type": treatments,
        }
    )


# ---------------------------------------------------------------------------
# Simulate normal claims
# ---------------------------------------------------------------------------

def simulate_claims(claimants, employers, providers, treatments):
    """
    Create claims and the relationships between claims and other entities.

    Each claim has:

        Claimant
        Employer
        Provider
        Treatment

    Normal claims randomly select providers.

    Fraudulent claims are added separately below so that we can
    deliberately create a recognizable provider network.
    """

    claimant_ids = claimants["claimant_id"].to_numpy()
    employer_ids = employers["employer_id"].to_numpy()

    normal_provider_ids = providers.loc[
        ~providers["is_fraud"],
        "provider_id",
    ].to_numpy()

    treatment_ids = treatments["treatment_id"].to_numpy()

    claims = []

    for index in range(1, N_CLAIMS + 1):

        claim_id = f"CL{index:06d}"

        claimant_id = random_choice(claimant_ids)

        employer_id = random_choice(employer_ids)

        # Most normal claims involve one provider.
        provider_id = random_choice(normal_provider_ids)

        treatment_id = random_choice(treatment_ids)

        # Claim amount is deliberately simple for now.
        #
        # Later we can make claim severity depend on treatment,
        # provider, claimant characteristics, etc.
        claim_amount = round(
            rng.lognormal(
                mean=8.0,
                sigma=0.8,
            ),
            2,
        )

        claims.append(
            {
                "claim_id": claim_id,
                "claimant_id": claimant_id,
                "employer_id": employer_id,
                "provider_id": provider_id,
                "treatment_id": treatment_id,
                "claim_amount": claim_amount,
                "is_fraud": False,
            }
        )

    return pd.DataFrame(claims)


# ---------------------------------------------------------------------------
# Add fraudulent claims
# ---------------------------------------------------------------------------

def add_fraudulent_claims(
    claims,
    claimants,
    employers,
    providers,
    treatments,
):
    """
    Add claims associated with the simulated fraud ring.

    The important characteristic is NOT simply that these providers
    have high claim volume.

    Instead, fraudulent providers repeatedly appear together.

    This creates a network structure that we can later detect with
    Neo4j.

    Example:

        Claimant A
             |
             v
           Claim
             |
             v
         Provider 1
             |
             v
         Provider 2
             |
             v
         Provider 3

    Repeating this pattern across many claimants creates a dense
    provider network.
    """

    claimant_ids = claimants["claimant_id"].to_numpy()
    employer_ids = employers["employer_id"].to_numpy()

    fraud_provider_ids = providers.loc[
        providers["is_fraud"],
        "provider_id",
    ].to_numpy()

    treatment_ids = treatments["treatment_id"].to_numpy()

    fraudulent_claims = []

    starting_claim_number = len(claims) + 1

    # Create 100 fraudulent claims.
    #
    # We intentionally keep this relatively small compared with
    # the normal population.
    n_fraud_claims = 100

    for offset in range(n_fraud_claims):

        claim_id = (
            f"CL{starting_claim_number + offset:06d}"
        )

        claimant_id = random_choice(claimant_ids)

        employer_id = random_choice(employer_ids)

        # IMPORTANT:
        #
        # Instead of choosing providers independently, we select
        # providers from the same small group.
        #
        # This creates the suspicious network structure.
        provider_id = random_choice(fraud_provider_ids)

        treatment_id = random_choice(treatment_ids)

        claim_amount = round(
            rng.lognormal(
                mean=8.5,
                sigma=0.7,
            ),
            2,
        )

        fraudulent_claims.append(
            {
                "claim_id": claim_id,
                "claimant_id": claimant_id,
                "employer_id": employer_id,
                "provider_id": provider_id,
                "treatment_id": treatment_id,
                "claim_amount": claim_amount,
                "is_fraud": True,
            }
        )

    fraudulent_claims = pd.DataFrame(fraudulent_claims)

    return pd.concat(
        [claims, fraudulent_claims],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Simulate provider referrals
# ---------------------------------------------------------------------------

def simulate_provider_referrals(providers):
    """
    Create provider-to-provider referral relationships.

    Normal providers have relatively random referral relationships.

    Fraudulent providers are much more likely to refer to other
    providers in the same fraud ring.

    This is particularly important for the graph model because it
    gives us an explicit Provider -> Provider network.
    """

    provider_ids = providers["provider_id"].to_numpy()

    fraud_provider_ids = providers.loc[
        providers["is_fraud"],
        "provider_id",
    ].to_numpy()

    referrals = []

    for provider_id in provider_ids:

        # Number of referrals made by this provider.
        n_referrals = rng.integers(1, 5)

        for _ in range(n_referrals):

            is_fraud_provider = provider_id in fraud_provider_ids

            if is_fraud_provider:

                # Fraudulent providers preferentially refer
                # to other fraudulent providers.
                possible_targets = fraud_provider_ids[
                    fraud_provider_ids != provider_id
                ]

            else:

                # Normal providers refer randomly.
                possible_targets = provider_ids[
                    provider_ids != provider_id
                ]

            if len(possible_targets) == 0:
                continue

            referred_provider_id = random_choice(
                possible_targets
            )

            referrals.append(
                {
                    "provider_id": provider_id,
                    "referred_provider_id": referred_provider_id,
                }
            )

    return pd.DataFrame(referrals).drop_duplicates()


# ---------------------------------------------------------------------------
# Save data
# ---------------------------------------------------------------------------

def save_data(
    claimants,
    employers,
    providers,
    treatments,
    claims,
    referrals,
):
    """
    Save each entity/relationship table as a CSV.

    Keeping these as separate files makes the eventual Neo4j loading
    process straightforward.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    claimants.to_csv(
        OUTPUT_DIR / "claimants.csv",
        index=False,
    )

    employers.to_csv(
        OUTPUT_DIR / "employers.csv",
        index=False,
    )

    providers.to_csv(
        OUTPUT_DIR / "providers.csv",
        index=False,
    )

    treatments.to_csv(
        OUTPUT_DIR / "treatments.csv",
        index=False,
    )

    claims.to_csv(
        OUTPUT_DIR / "claims.csv",
        index=False,
    )

    referrals.to_csv(
        OUTPUT_DIR / "provider_referrals.csv",
        index=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("Simulating workers' compensation data...")

    # ---------------------------------------------------------------
    # Create entities
    # ---------------------------------------------------------------

    claimants = simulate_claimants()

    employers = simulate_employers()

    providers = simulate_providers()

    treatments = simulate_treatments()

    # ---------------------------------------------------------------
    # Create claims
    # ---------------------------------------------------------------

    claims = simulate_claims(
        claimants,
        employers,
        providers,
        treatments,
    )

    # ---------------------------------------------------------------
    # Add known fraudulent claims
    # ---------------------------------------------------------------

    claims = add_fraudulent_claims(
        claims,
        claimants,
        employers,
        providers,
        treatments,
    )

    # ---------------------------------------------------------------
    # Create provider referral network
    # ---------------------------------------------------------------

    referrals = simulate_provider_referrals(
        providers
    )

    # ---------------------------------------------------------------
    # Save everything
    # ---------------------------------------------------------------

    save_data(
        claimants,
        employers,
        providers,
        treatments,
        claims,
        referrals,
    )

    # ---------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------

    print()
    print("Simulation complete.")
    print()

    print(f"Claimants:   {len(claimants):,}")
    print(f"Employers:   {len(employers):,}")
    print(f"Providers:   {len(providers):,}")
    print(f"Treatments:  {len(treatments):,}")
    print(f"Claims:      {len(claims):,}")
    print(f"Referrals:   {len(referrals):,}")

    print()

    print(
        "Fraudulent providers:",
        providers["is_fraud"].sum(),
    )

    print(
        "Fraudulent claims:",
        claims["is_fraud"].sum(),
    )

    print()

    print(f"Data written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()