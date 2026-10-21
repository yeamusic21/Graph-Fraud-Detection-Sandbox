# Graph-Fraud-Detection-Sandbox
Sandbox for learning to detect potential fraud using graph DB

# Pipeline
The pipeline will be:
1. 00_simulate_data.py — Generate synthetic workers' compensation data, including a simulated provider ring.
2. 01_prepare_data.py — Validate relationships, check data quality, and prepare the CSV files.
3. 02_baseline_analysis.py — Analyze normal recovery patterns and establish baseline behavior.
4. src/graph/ — Load the data into Neo4j Aura, explore the graph, and calculate graph features.
5. src/fraud/ — Detect anomalies and evaluate fraud-detection performance.

# Notes

01_prepare_data.py creates claims_enriched.csv for analysis.
- This dataset combines claims with their providers' fraud labels. It will be useful for comparing claim characteristics and establishing an initial baseline before we introduce graph-based features.