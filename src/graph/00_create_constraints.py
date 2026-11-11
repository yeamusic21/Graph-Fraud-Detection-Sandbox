"""
00_create_constraints.py

Create Neo4j constraints and indexes for the workers' compensation
fraud detection graph.

Node labels:
    Claimant
    Claim
    Provider
    Employer
    Procedure
    Diagnosis
    Injury

The uniqueness constraints ensure that each entity has a unique
identifier, preventing duplicate nodes during graph ingestion.
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError(
        "Missing Neo4j configuration. "
        "Check NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
    )


# ---------------------------------------------------------------------
# Schema Constraints
# ---------------------------------------------------------------------

CONSTRAINTS = [
    """
    CREATE CONSTRAINT claimant_id_unique IF NOT EXISTS
    FOR (n:Claimant)
    REQUIRE n.claimant_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT claim_id_unique IF NOT EXISTS
    FOR (n:Claim)
    REQUIRE n.claim_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT provider_id_unique IF NOT EXISTS
    FOR (n:Provider)
    REQUIRE n.provider_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT employer_id_unique IF NOT EXISTS
    FOR (n:Employer)
    REQUIRE n.employer_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT procedure_id_unique IF NOT EXISTS
    FOR (n:Procedure)
    REQUIRE n.procedure_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT diagnosis_id_unique IF NOT EXISTS
    FOR (n:Diagnosis)
    REQUIRE n.diagnosis_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT injury_id_unique IF NOT EXISTS
    FOR (n:Injury)
    REQUIRE n.injury_id IS UNIQUE
    """,
]


# ---------------------------------------------------------------------
# Create Constraints
# ---------------------------------------------------------------------

def create_constraints(driver):
    """Create all graph uniqueness constraints."""

    with driver.session() as session:

        for constraint in CONSTRAINTS:
            session.run(constraint)

    print(f"Created or verified {len(CONSTRAINTS)} constraints.")


# ---------------------------------------------------------------------
# Verify Constraints
# ---------------------------------------------------------------------

def verify_constraints(driver):
    """Display all constraints currently defined in Neo4j."""

    with driver.session() as session:

        result = session.run("SHOW CONSTRAINTS")

        constraints = list(result)

    print("\nCurrent Neo4j Constraints:")
    print("-" * 60)

    for constraint in constraints:
        print(
            f"{constraint['name']}: "
            f"{constraint['type']}"
        )

    print("-" * 60)
    print(f"Total constraints: {len(constraints)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("Connecting to Neo4j Aura...")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    try:
        driver.verify_connectivity()

        print("Successfully connected to Neo4j.")

        create_constraints(driver)

        verify_constraints(driver)

        print("\nSchema initialization complete.")

    finally:
        driver.close()


if __name__ == "__main__":
    main()