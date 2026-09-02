
# Factory Reallocation & Shipping Optimization Recommendation System

### Nassau Candy Distributor | Unified Mentor Data Science Internship

## Overview

Nassau Candy currently assigns products to factories using static rules that don't account for shipping distance or delivery performance. This project analyzes order data to predict shipping lead time, identify underperforming factory-region routes, and simulate factory reassignment scenarios — recommending configurations that reduce delivery time without hurting profitability.

## Problem Statement

- Products are assigned to factories using fixed rules, not shipping performance data
- Some factory-region routes are consistently slower than others
- There was no system to simulate "what if this product shipped from a different factory?"

## Data

The dataset (`Nassau_Candy_Distributor.csv`) contains 10,194 order records with order details, customer location, product/division, and financial fields. Factory coordinates and a product-to-factory mapping table were also provided.

**Data quality note:** The dataset's Order Date and Ship Date fields did not reflect real shipping durations (see the research paper, Section 2, for full investigation). A synthetic lead-time variable was constructed from factory-to-customer distance (via geocoding + Haversine distance) and shipping mode instead.

## Methodology

1. **Data preparation** — geocoded customer locations, calculated factory-to-customer distance (Haversine formula), constructed synthetic lead time
2. **EDA** — analyzed lead time patterns by factory, division, and region; correlation analysis
3. **Predictive modeling** — trained and compared Linear Regression, Random Forest, and Gradient Boosting to predict lead time
4. **Route clustering** — K-Means clustering to identify high-priority (slow + high-volume) routes
5. **Scenario simulation** — simulated reassigning each real order scenario to all 5 factories, ranked by predicted improvement

## Key Results

| Model                       | RMSE  | MAE   | R²             |
| --------------------------- | ----- | ----- | --------------- |
| **Linear Regression** | 0.574 | 0.452 | **0.894** |
| Gradient Boosting           | 0.578 | 0.458 | 0.892           |
| Random Forest               | 0.622 | 0.482 | 0.875           |

- **77.8%** of shipping scenarios (63 of 81) show improvement through factory reassignment
- **33.7%** average lead-time reduction among scenarios that benefit
- **~0** correlation between distance and profit — reassignment is profit-neutral
- **100%** of real orders covered by simulated scenarios

## Dashboard

The Streamlit dashboard includes:

- **Overview & KPIs** — summary metrics and factory locations
- **Factory Optimization Simulator** — live prediction for any product/region/ship mode
- **What-If Scenario Analysis** — compare current vs. recommended assignments
- **Recommendation Dashboard** — ranked reassignment opportunities
- **Risk & Impact Panel** — profit safety check and confidence-flagged recommendations

## Tech Stack

Python, pandas, scikit-learn, geopy, Plotly, Streamlit

## Author

Kaustabhjyoti Baishya — Unified Mentor Data Science Internship
