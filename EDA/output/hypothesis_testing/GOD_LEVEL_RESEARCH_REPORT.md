# God-Level Research Report

## Scope
This report consolidates hypothesis testing on meteorological, astronomical, Vedic, and Samvatsaram-linked signals in storm evolution.

## Hypothesis 1: Vedic Predictive Lift
- Baseline RMSE: 4.0430
- Vedic RMSE: 4.6207
- Relative improvement (%): -14.29
- **RESULT: Vedic features DEGRADED prediction by 14.3%. Null hypothesis NOT rejected.**
- Adding Vedic features increased RMSE compared to the meteorological baseline.
- This indicates Vedic features add noise rather than signal for predicting wind change.

## Hypothesis 2: Lunar Phase vs RI
- Chi-square: 19.6558
- p-value: 0.006363
- Degrees of freedom: 7
- Bonferroni-corrected alpha (across 7 hypothesis groups): 0.0071
- **RESULT: Significant after Bonferroni correction.**
- RI threshold used: 30 kt/24h (NOTE: WMO standard is 35 kt)

## Hypothesis 3: Planetary Speed vs Trajectory
- Status: ok
- Tested feature pairs: 60
- Top associations are written to `hypothesis3_spearman_matrix.csv`.
- Retrograde/direct ANOVA table is written to `hypothesis3_retrograde_anova.csv`.

## Samvatsaram Analysis
- Status: ok
- Samvatsara classes: 11
- Wind ANOVA p-value: 0.000000
- Pressure ANOVA p-value: 0.000000
- RI vs Samvatsara chi-square p-value: 0.000208
- **CAUTION**: Each Samvatsara maps to roughly one calendar year. Significant ANOVA
  results reflect interannual storm intensity variability, NOT a Hindu calendar effect.
  This is equivalent to testing 'do storms vary in intensity across different years?'
  — which is trivially true due to ENSO, PDO, and decadal climate cycles.

## Mars + Rahu + Ketu Hypothesis
- Mars feature count tested: 16
- Rahu feature count tested: 12
- Ketu feature count tested: 12
- Wind-change RMSE baseline: 4.0340
- Wind-change RMSE with Mars+Rahu+Ketu: 4.0785
- Relative improvement (%): -1.10
- RI split mode: stratified_fallback
- RI AUC baseline: nan
- RI AUC with Mars+Rahu+Ketu: nan
- Mars features in top-20 importance: 11
- Best Mars feature rank: 6
- Rahu features in top-20 importance: 1
- Best Rahu feature rank: 19
- Ketu features in top-20 importance: 1
- Best Ketu feature rank: 20
- Altitude-band vs RI chi-square p-value: 0.531932
- Retrograde vs direct wind-change p-value: 0.915742
- Rahu retrograde vs direct wind-change p-value: 0.039763
- Rahu degree vs wind-change Spearman p-value: 0.006045
- Ketu retrograde vs direct wind-change p-value: 0.039763
- Ketu degree vs wind-change Spearman p-value: 0.082170

## Mars+Rahu Event-Window (24h Pre/Post RI)
- Status: ok
- RI event windows: 91
- non-RI control windows: 87
- Tested astro features: 13
- RI threshold for event windows: 15 kt (relaxed from 30 kt to generate sufficient events)
- Detailed comparison: `mars_rahu_event_window_comparison.csv`.

## Files Produced
- hypothesis1_results.txt
- hypothesis2_results.txt
- hypothesis3_results.txt
- hypothesis3_spearman_matrix.csv
- hypothesis3_retrograde_anova.csv
- samvatsara_summary.csv
- samvatsara_results.txt
- mars_rahu_ketu_hypothesis_results.txt
- mars_rahu_feature_importance.csv
- mars_altitude_ri_crosstab.csv
- mars_rahu_event_window_results.txt
- mars_rahu_event_window_comparison.csv
- mars_rahu_event_windows.csv

## Research Interpretation
- H1 indicates whether Vedic signals add out-of-sample lift over meteorology baseline.
- H2 quantifies non-random lunar-phase concentration of rapid intensification.
- H3 tests mechanistic coupling between planetary speed dynamics and trajectory geometry.
- Samvatsaram block evaluates inter-year cyclic grouping effects on intensity and RI rates.