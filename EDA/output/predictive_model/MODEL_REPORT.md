# Predictive Modeling Report

## Dataset
- Rows used: 46589
- Train rows: 37274
- Test rows: 9315
- Split cutoff: 2023-10-21 12:00:00+00:00

## Feature Blocks
- Meteo features: 14
- Panchangam features: 18
- Scientific features: 190

## Regression Summary
- Best model: meteo_only
- Best RMSE: 4.1642
- Best MAE: 2.5396
- Full table: `model_comparison.csv`

## Classification Target
- RI threshold selected for next-step delta: >= 10.0
- NOTE: This is a NEXT-STEP wind delta, NOT a 24-hour delta.
  It differs from 06_hypothesis_testing.py which uses 30 kt/24h rolling.
  Cross-script RI comparisons are NOT valid.
- Metrics table: `classification_metrics.csv`

## Spline Feature Leakage Warning
- spline_bearing_deg, spline_curvature, spline_displacement, spline_length_deg
  are included in the METEO block. These encode forecast track geometry and
  may carry implicit information about the future trajectory (leakage risk).
  See discover_blocks() in 07_predictive_model_stack.py for full comment.

## Group Contribution
- Group-level RMSE lift table: `group_importance.csv`
- Feature ranking: `top_features.csv`