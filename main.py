import pandas as pd

df = pd.read_csv("data/ai_usage.csv")

# Capture
df["total_tokens"] = df["input_tokens"] + df["output_tokens"]

# Allocate
showback = df.groupby(["team", "application"], as_index=False).agg(
    total_cost=("cost_usd", "sum"),
    total_tokens=("total_tokens", "sum"),
    total_requests=("requests", "sum"),
    business_outcome=("business_outcome", "sum")
)

# Analyze
showback["cost_per_request"] = showback["total_cost"] / showback["total_requests"]
showback["cost_per_outcome"] = showback["total_cost"] / showback["business_outcome"]

# Optimize logic
def recommendation(row):
    if row["cost_per_request"] > 0.05:
        return "Review model choice, prompt size, caching, and routing policy"
    elif row["cost_per_outcome"] > 0.10:
        return "Validate business value and reduce unnecessary AI calls"
    else:
        return "Cost profile acceptable; continue monitoring"

showback["recommendation"] = showback.apply(recommendation, axis=1)

showback.to_csv("outputs/showback_report.csv", index=False)
# Model Tradeoff Analysis

tradeoff = df.groupby(
    ["provider", "model", "model_tier"],
    as_index=False
).agg(
    total_cost=("cost_usd", "sum"),
    total_tokens=("total_tokens", "sum"),
    total_requests=("requests", "sum"),
    avg_quality=("quality_score", "mean"),
    avg_latency_ms=("latency_ms", "mean"),
    business_outcome=("business_outcome", "sum")
)

tradeoff["cost_per_1k_tokens"] = (
    tradeoff["total_cost"] /
    (tradeoff["total_tokens"] / 1000)
)

tradeoff["cost_per_request"] = (
    tradeoff["total_cost"] /
    tradeoff["total_requests"]
)

tradeoff["cost_per_outcome"] = (
    tradeoff["total_cost"] /
    tradeoff["business_outcome"]
)

tradeoff["efficiency_score"] = (
    tradeoff["avg_quality"] /
    tradeoff["cost_per_request"]
)

def tradeoff_recommendation(row):
    if row["cost_per_request"] < 0.02 and row["avg_quality"] >= 85:
        return "Strong candidate for cost-optimized routing"
    elif row["avg_quality"] >= 92:
        return "Use for high-value workflows"
    else:
        return "Monitor cost-quality tradeoff"

tradeoff["recommendation"] = tradeoff.apply(
    tradeoff_recommendation,
    axis=1
)

tradeoff.to_csv(
    "outputs/model_tradeoff_report.csv",
    index=False
)
summary = f"""
# AI FinOps Operating Model Executive Summary

This model applies FinOps operating principles to AI consumption.

## Operating Flow

Capture → Allocate → Analyze → Optimize → Govern → Measure Value

## Total AI Cost
${df["cost_usd"].sum():,.2f}

## Total Tokens
{df["total_tokens"].sum():,.0f}

## Total Requests
{df["requests"].sum():,.0f}

## What the Model Shows

This model captures AI usage by provider, model, team, application, workflow, tokens, cost, and business outcome.

It then allocates cost to teams and applications, calculates unit economics, and generates optimization recommendations.

## Key Metrics

- Cost per request
- Cost per business outcome
- Token consumption by team
- AI cost by application
- Optimization recommendation

## FinOps Translation

Traditional FinOps manages cloud consumption.

AI FinOps manages AI consumption using the same operating principles:

- Visibility
- Allocation
- Accountability
- Optimization
- Forecasting
- Governance
- Business value measurement
"""
# Model Tradeoff Analysis
tradeoff = df.groupby(["provider", "model", "model_tier"], as_index=False).agg(
    total_cost=("cost_usd", "sum"),
    total_tokens=("total_tokens", "sum"),
    total_requests=("requests", "sum"),
    avg_quality=("quality_score", "mean"),
    avg_latency_ms=("latency_ms", "mean"),
    business_outcome=("business_outcome", "sum")
)

tradeoff["cost_per_1k_tokens"] = tradeoff["total_cost"] / (tradeoff["total_tokens"] / 1000)
tradeoff["cost_per_request"] = tradeoff["total_cost"] / tradeoff["total_requests"]
tradeoff["cost_per_outcome"] = tradeoff["total_cost"] / tradeoff["business_outcome"]

tradeoff["efficiency_score"] = (
    tradeoff["avg_quality"] / tradeoff["cost_per_request"]
)

def tradeoff_recommendation(row):
    if row["cost_per_request"] < 0.02 and row["avg_quality"] >= 85:
        return "Strong candidate for cost-optimized routing"
    elif row["avg_quality"] >= 92 and row["cost_per_request"] > 0.05:
        return "Use for high-value or quality-sensitive workflows"
    else:
        return "Use selectively; monitor quality, latency, and cost"

tradeoff["recommendation"] = tradeoff.apply(tradeoff_recommendation, axis=1)

tradeoff.to_csv("outputs/model_tradeoff_report.csv", index=False)

with open("outputs/executive_summary.md", "w") as f:
    f.write(summary)

print(showback)
print("\nReports created:")
print("- outputs/showback_report.csv")
print("- outputs/executive_summary.md")
print("- outputs/model_tradeoff_report.csv")
