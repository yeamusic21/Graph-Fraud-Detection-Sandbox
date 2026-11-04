# Evaluating the Baseline Fraud Indicators

Let's break down what we're seeing.

## 1. The baseline is performing poorly

Our dataset contains 2,600 claims, of which 100 are fraudulent (3.85%).


  |Metric                |       Result |Interpretation
  |--------------------- |------------- |---------------------
  |Precision             |        8.15% |Only 8 out of 100 flagged claims are actually fraudulent.
  |Recall                |       11.00% |We identify only 11 of the 100 fraudulent claims.
  |Specificity           |       95.04% |We correctly leave 95% of legitimate claims unflagged.
  |Fraud prevalence      |        3.85% |Fraud represents a relatively small portion of the dataset.
  |Flagged rate          |        5.19% |We flag approximately 5% of all claims.

The particularly interesting metric is precision.

We're flagging 135 claims, but only 11 are fraudulent. That's 124 false
positives.

In an actual workers' compensation environment, that would create a
significant investigation workload for relatively few discoveries.

## 2. What does this tell us about our approach?

I think we've established an important distinction:

**Knowing which claims are fraudulent doesn't necessarily mean that
simple aggregate statistics can distinguish fraudulent behavior from
legitimate behavior.**

For example, the features we're using include:

-   Number of claims associated with a claimant.
-   Number of claimants associated with a provider.
-   Number of providers associated with a claim.
-   Number of treatments associated with a claim.
-   Potentially, historical payout amounts.

These features provide useful information about the structure and volume
of activity, but they don't necessarily tell us whether that activity is
suspicious.

A legitimate provider might treat hundreds of claimants. A fraudulent
provider might treat hundreds of claimants.

The difference might be in *which* claimants they treat, *how* they are
connected, and *what patterns* emerge from those relationships.

## 3. This is where the graph becomes interesting

I would actually consider this a justification for continuing with
Neo4j.

Instead of simply counting relationships, we want to investigate the
relationships themselves.

For example:

  -----------------------------------------------------------------------
  Indicator               Aggregate approach      Graph approach
  ----------------------- ----------------------- -----------------------
  Provider activity       Number of claimants     Unusual
                                                  claimant-provider
                                                  relationships

  Claimant activity       Number of claims        Connections to
                                                  suspicious providers

  Treatment activity      Number of treatments    Unusual treatment
                                                  sequences

  Provider networks       Number of connected     Clusters of providers
                          providers               sharing claimants

  Claimant networks       Number of shared        Unusual overlap between
                          providers               claimant groups
  -----------------------------------------------------------------------

Consider a hypothetical scenario:

A provider treats 150 claimants. That's not inherently suspicious.

However, suppose:

-   40 of those claimants also visited the same three providers.
-   Those three providers share an unusually high proportion of
    claimants.
-   The claimants exhibit remarkably similar treatment patterns.
-   The providers have unusually similar payout distributions.

Individually, these observations might not raise any flags.

Collectively, they could indicate an organized fraud network.
