# Zomato Kitchen Prep Time (KPT) Optimization

## Problem
KPT prediction is inaccurate because current systems rely on merchant-marked Food Ready (FOR) signals, which are noisy and biased.

This leads to:
- Incorrect ETA
- Rider waiting time
- Higher cancellations

## Our Approach
We redesigned the signal layer instead of only improving the ML model.

We introduced:
- Order complexity signals
- Kitchen Load Index (KLI)
- Merchant reliability modeling
- Rush hour adjustment

## Kitchen Load Index Formula

KLI =
0.4 × orders_last_10min +
0.3 × order_complexity +
0.2 × external_load +
0.1 × merchant_speed_inverse

## Results

Baseline MAE: 25.16  
Improved MAE: 2.23  

~90% improvement in prediction accuracy.

## How To Run

1. Install dependencies:
pip install -r requirements.txt

2. Run:
python run_pipeline.py

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

## Conclusion
Improving signal quality instead of just model complexity significantly improves KPT prediction accuracy and operational efficiency.
