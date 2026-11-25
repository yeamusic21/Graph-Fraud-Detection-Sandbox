"""
01_load_graph.py

Load prepared workers' compensation data into Neo4j.

Responsibilities:
    1. Load Claimant, Employer, Provider, Treatment, and Claim nodes.
    2. Establish relationships between claims and their associated entities.
    3. Establish provider referral relationships.
    4. Use MERGE to support repeatable, idempotent loading.
    5. Batch database writes for efficiency.

Input:
    data/processed/claimants.csv
    data/processed/employers.csv
    data/processed/providers.csv
    data/processed/treatments.csv
    data/processed/claims.csv
    data/processed/provider_referrals.csv

Prerequisites:
    - Neo4j Aura database
    - Constraints created by 00_create_constraints.py
    - NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD
      environment variables

Expected graph structure:
    After running the script, the graph will contain the following structure.

    (:Claimant)
        |
    FILED
        |
    (:Claim)
        |
        +---- FILED_AGAINST ----> (:Employer)
        |
        +---- TREATED_BY -------> (:Provider)
        |
        +---- RECEIVED_TREATMENT -> (:Treatment)


    (:Provider)
        |
    REFERRED
        |
    (:Provider)

A couple of implementation notes:
    1. Idempotency: Running this script multiple times won't create duplicate nodes or relationships, 
        assuming the uniqueness constraints are in place.
    2. Batching: Records are submitted in batches of 1,000, avoiding individual database writes.
    3. Missing relationships: If a referenced entity doesn't exist, the corresponding relationship is 
        simply not created. This is preferable to creating incomplete graph structures.
    4. Relationship properties: We can introduce treatment dates, referral dates, and sequence 
        information as the project evolves.
"""

import csv
import os
from pathlib import Path

from neo4j import GraphDatabase
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

BATCH_SIZE = 1000

load_dotenv(PROJECT_ROOT / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


# ---------------------------------------------------------------------
# CSV Utilities
# ---------------------------------------------------------------------

def read_csv(filename):
    """Read a CSV file and return its records."""

    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    with open(path, mode="r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_value(value):
    """Convert CSV values into appropriate Python types."""

    if value is None or value == "":
        return None

    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def prepare_records(records):
    """Convert CSV string values into Python values."""

    prepared = []

    for record in records:
        prepared.append({
            key: parse_value(value)
            for key, value in record.items()
        })

    return prepared


def batches(records, batch_size=BATCH_SIZE):
    """Yield records in batches."""

    for start in range(0, len(records), batch_size):
        yield records[start:start + batch_size]


# ---------------------------------------------------------------------
# Neo4j Utilities
# ---------------------------------------------------------------------

def execute_batch(session, query, records):
    """Execute a Cypher query against records in batches."""

    total = 0

    for batch in batches(records):
        result = session.run(query, records=batch)
        summary = result.consume()

        total += summary.counters.nodes_created
        total += summary.counters.relationships_created

    return total


# ---------------------------------------------------------------------
# Node Loading
# ---------------------------------------------------------------------

def load_nodes(session, filename, label, id_field):
    """
    Load nodes from a CSV file.

    Nodes are identified by their corresponding ID field.
    MERGE ensures that existing nodes are updated rather than duplicated.
    """

    records = prepare_records(read_csv(filename))

    query = f"""
    UNWIND $records AS row

    MERGE (n:{label} {{{id_field}: row.{id_field}}})

    SET n += row
    """

    execute_batch(session, query, records)

    print(f"Loaded {len(records):,} {label} nodes.")


def load_all_nodes(session):
    """Load all entity nodes."""

    load_nodes(
        session,
        "claimants.csv",
        "Claimant",
        "claimant_id"
    )

    load_nodes(
        session,
        "employers.csv",
        "Employer",
        "employer_id"
    )

    load_nodes(
        session,
        "providers.csv",
        "Provider",
        "provider_id"
    )

    load_nodes(
        session,
        "treatments.csv",
        "Treatment",
        "treatment_id"
    )

    load_nodes(
        session,
        "claims.csv",
        "Claim",
        "claim_id"
    )


# ---------------------------------------------------------------------
# Claim Relationships
# ---------------------------------------------------------------------

def load_claim_relationships(session):
    """Establish relationships between claims and associated entities."""

    records = prepare_records(read_csv("claims.csv"))

    relationships = [
    (
        """
        UNWIND $records AS row
        MATCH (claimant:Claimant {claimant_id: row.claimant_id})
        MATCH (claim:Claim {claim_id: row.claim_id})
        MERGE (claimant)-[:FILED]->(claim)
        """,
        "Claimant -> Claim"
    ),
    (
        """
        UNWIND $records AS row
        MATCH (claim:Claim {claim_id: row.claim_id})
        MATCH (employer:Employer {employer_id: row.employer_id})
        MERGE (claim)-[:FILED_AGAINST]->(employer)
        """,
        "Claim -> Employer"
    ),
    (
        """
        UNWIND $records AS row
        MATCH (claim:Claim {claim_id: row.claim_id})
        MATCH (provider:Provider {provider_id: row.provider_id})
        MERGE (claim)-[:TREATED_BY]->(provider)
        """,
        "Claim -> Provider"
    ),
    (
        """
        UNWIND $records AS row
        MATCH (claim:Claim {claim_id: row.claim_id})
        MATCH (treatment:Treatment {treatment_id: row.treatment_id})
        MERGE (claim)-[:RECEIVED_TREATMENT]->(treatment)
        """,
        "Claim -> Treatment"
    )
]

    for query, description in relationships:
        execute_batch(session, query, records)

        print(f"Loaded relationships: {description}")


# ---------------------------------------------------------------------
# Provider Referral Relationships
# ---------------------------------------------------------------------

def load_provider_referrals(session):
    """Establish referral relationships between providers."""

    records = prepare_records(read_csv("provider_referrals.csv"))

    query = """
    UNWIND $records AS row

    MATCH (source:Provider {provider_id: row.provider_id})
    MATCH (target:Provider {
        provider_id: row.referred_provider_id
    })

    MERGE (source)-[:REFERRED]->(target)
    """

    execute_batch(session, query, records)

    print(f"Loaded {len(records):,} provider referral records.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        raise ValueError(
            "Missing Neo4j configuration. "
            "Check NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
        )

    print("Connecting to Neo4j Aura...")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    try:
        driver.verify_connectivity()

        print("Connected successfully.\n")

        with driver.session(database=NEO4J_DATABASE) as session:

            print("Loading graph nodes...")
            load_all_nodes(session)

            print("\nLoading claim relationships...")
            load_claim_relationships(session)

            print("\nLoading provider referrals...")
            load_provider_referrals(session)

        print("\nGraph loading completed successfully.")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
