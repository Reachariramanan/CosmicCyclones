# Output Analysis Report
Date: 2026-03-15
Workspace: `drik_parser`
Analyzed path: `EDA/output/`

## 1) What `/output/` contains (high-level)
`EDA/output/` is the analytics artifact hub produced by the EDA pipeline:

1. `01_data_transform.py` -> creates unified datasets + metadata
2. `02_basic_eda.py` -> baseline descriptive and correlation outputs
3. `03_advanced_eda.py` -> PCA, clustering, feature-importance, MI outputs
4. `04_shap_lime.py` -> model explanation artifacts (SHAP/LIME)
5. `05_comprehensive_eda.py` -> deeper statistical/comparative analyses + text summary

Top-level files in `EDA/output/`:
- `storm_data_all.parquet` (main analysis dataset)
- `storm_data_all.csv`
- `storm_data_all.xlsx`
- `column_metadata.csv` (column completeness/type profile)
- Subfolders: `basic_eda/`, `advanced_eda/`, `shap_lime/`, `comprehensive/`

Observed dataset snapshot (from generated summaries):
- Rows: **36,536**
- Columns: **281**
- Storms: **735**
- Date range: **2016-07-15 to 2026-01-24 (UTC)**
- Numeric features: **208**

---

## 2) Subfolder-by-subfolder map (`what is what`)

### A. `basic_eda/` (25 files: 18 PNG, 7 CSV)
Purpose: first-pass data understanding and sanity checks.

Key outputs:
- Data quality: `missing_values.csv`, `missing_heatmap.png`
- Descriptive stats: `descriptive_stats.csv`, `core_meteo_stats.csv`
- Distribution views: `distributions_core.png`, `wind_by_storm_type.png`, `wind_by_storm.png`, `pressure_by_storm.png`
- Relationship views: `correlation_with_wind.csv`, `correlations_with_wind.png`, `correlation_matrix_core.png`
- Robust category checks: `quartile_wind_by_category.png`, `kruskal_wallis_wind.csv`, `wind_quartile_summary.csv`
- Timeline/geography: `time_series.png`, `storm_tracks.png`, `storm_tracks_individual.png`

Interpretation:
- This folder answers: *Is data usable? What are primary distributions? Which variables linearly co-move with wind?*

### B. `advanced_eda/` (19 files: 11 PNG, 7 CSV, 1 TXT)
Purpose: higher-order structure and predictive feature ranking.

Key outputs:
- PCA: `pca_astronomical.png`, `pca_loadings.csv`, `pca_pc1_loadings.png`
- Model importances: `feature_importance_all.csv`, `feature_importance_top30.png`, `variable_importance_summary.txt`
- Nonlinear dependence: `mutual_information.csv`, `mutual_information_top30.png`
- Clustering/redundancy: `clustering_kmeans.png`, `cluster_profiles.csv`, `dendrogram_storms.png`, `highly_correlated_pairs.csv`, `feature_clustermap.png`
- Vedic category significance: `vedic_kruskal_wallis.csv`, `vedic_significant_boxplots.png`
- Dynamics + aspects: `intensity_change.png`, `aspect_correlations.csv`, `aspect_correlations.png`

Interpretation:
- This folder answers: *Which features matter under tree models? Are there nonlinear signals? Are there clusters/redundancies?*

### C. `shap_lime/` (21 files: 14 PNG, 4 CSV, 3 TXT)
Purpose: explain model predictions globally and locally.

Key outputs:
- SHAP global: `shap_summary_beeswarm.png`, `shap_summary_bar.png`, `shap_importance.csv`
- SHAP local/interactions: `shap_dependence_top6.png`, `shap_top_interactions.csv`, `shap_interactions_top20.png`, `shap_instance_explanations.txt`
- LIME local+aggregate: `lime_instance_*.png`, `lime_importance_aggregated.csv`, `lime_explanations.txt`
- Agreement layer: `shap_vs_lime_comparison.csv`, `shap_vs_lime_scatter.png`, `shap_vs_lime_top20.png`, `consensus_important_features.txt`

Interpretation:
- This folder answers: *Why did the model predict this wind value? Which features are consistently important across explainers?*

### D. `comprehensive/` (26 files: 16 PNG, 9 CSV, 1 TXT)
Purpose: domain/statistical depth checks and final consolidated summary.

Key outputs:
- Lifecycle & temporal: `lifecycle_*`, `autocorrelation_wind.png`, `crosscorr_wind_sun.png`, `spectral_analysis.png`
- Distributional tests: `normality_tests.csv`, `qq_plots.png`
- Category interactions: `tithi_nakshatra_heatmap*.png`, `paksha_moonsign_pivot.csv`, `nakshatra_intensity_*`
- Robustness/risk: `anomaly_detection.png`, `anomaly_by_storm.csv`, `vif_analysis.csv`, `vif_barplot.png`
- Astro/geo effects: `retrograde_*`, `vedic_degree_*`, `basin_*`
- Final text summary: `comprehensive_summary.txt`

Interpretation:
- This folder answers: *Are effects robust, interpretable, and statistically coherent?*

---

## 3) Parameters that are playing a role (grouped)

## 3.1 Data construction parameters (`01_data_transform.py`)
These decide **what enters the dataset**:
- Input scope: all `Compiled_data/*.json` files found.
- Join key: storm id + `scientific_data.datetime_utc` to match `all_storm_data.json` track rows.
- Skyfield body set: `sun, moon, mercury, venus, mars, jupiter, saturn, uranus, neptune`.
- Skyfield per-body fields: RA/Dec, ecliptic lon/lat, altitude/azimuth, sin/cos transforms.
- Vedic numeric fields: `full_degree`, `padam`, `speed_deg_per_day`, `right_ascension`, `declination`.
- Derived temporal features: `hour_utc`, `day_of_week`, `day_of_year`, `month`.
- Derived intensity/motion features: `wind_change`, `pressure_change`, `displacement`, `hours_elapsed`, `movement_speed`.
- Spline derived features: point count, length, endpoint, displacement, bearing, curvature.
- Cone derived features: point count, area, bounds, spreads.

## 3.2 Basic EDA parameters (`02_basic_eda.py`)
These control **summary behavior and visualization slices**:
- Top-storm slices: top 40 by peak wind / deepest pressure.
- Missing heatmap cap: first 40 partially-missing columns.
- Category robustness checks:
  - Quartiles (Q1/Q3), median, skew checks
  - Kruskal-Wallis for grouped wind distributions
  - Minimum grouped sample for KW filtering (`>=5` points)
- Correlation views target wind; includes selected core + astro columns.

## 3.3 Advanced modeling parameters (`03_advanced_eda.py`)
These control **feature ranking and structure discovery**:
- Feature eligibility: numeric columns with >50% non-null, with leakage exclusions.
- Target: `wind`.
- RF: `n_estimators=200`, `max_depth=10`, `random_state=42`, CV=5.
- XGBoost: `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `random_state=42`, CV=5.
- LightGBM: `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `random_state=42`, CV=5.
- MI: `n_neighbors=5`, `random_state=42`.
- KMeans elbow: k=2..10; chosen cluster plot uses k=4.
- High-correlation cutoff for redundancy report: `|r| > 0.95`.
- Intensity event thresholds: rapid intensification/weakening uses ±30 kt change.
- Vedic category significance threshold: Kruskal-Wallis `p < 0.05`.

## 3.4 Explainability parameters (`04_shap_lime.py`)
These control **explanation reliability and granularity**:
- Feature eligibility stricter than advanced EDA: >80% non-null.
- Train-test split: 80/20 with `random_state=42`.
- SHAP model: XGBoost (`n_estimators=200`, `max_depth=6`, `learning_rate=0.1`).
- SHAP interactions computed on sample size `min(50, test_size)`.
- LIME:
  - quartile-based instance selection (10/25/50/75/90 percentiles of wind)
  - `num_samples=1000` per explanation.
- SHAP/LIME comparison normalized to [0,1], with consensus defined as overlap in top-30 lists.

## 3.5 Comprehensive statistics parameters (`05_comprehensive_eda.py`)
These control **deep diagnostic conclusions**:
- Lifecycle bins: `[0-20%], [20-50%], [50-70%], [70-100%]`.
- Isolation Forest contamination: `0.05` (expected anomaly rate).
- Day/night split: `sun_altitude_deg > 0`.
- Day vs night test: Mann-Whitney U (non-parametric).
- Basin assignment by hard longitude-latitude rules.
- Normality tests: Shapiro (sample max 5000) + D’Agostino.
- VIF interpretation lines: 5 (moderate), 10 (high).

---

## 4) What currently looks most influential (from outputs)

From generated artifacts (not assumptions):

- `advanced_eda/variable_importance_summary.txt` top features include:
  - `longitude`, `latitude`, `spline_end_lon`, `spline_bearing_deg`,
  - `vedic_chandra_speed_deg_per_day`, `dinamana_minutes`, `sun_dec_deg`, `elevation`.

- `shap_lime/consensus_important_features.txt` (SHAP & LIME agreement, 14 features):
  - `latitude`, `longitude`, `spline_end_lon`, `spline_bearing_deg`, `spline_curvature`,
  - `dinamana_minutes`, `elevation`, and several Vedic/planetary terms.

- `comprehensive/vif_analysis.csv` shows low multicollinearity in selected check set (all VIF near ~1–2; none near danger levels 5/10).

- `comprehensive/comprehensive_summary.txt` indicates notable missingness concentration in:
  - `storm_active` (~99.6%)
  - cone features (~98.8%)
  - spline features (~15.7–15.9%)

### Important caution
`advanced_eda/variable_importance_summary.txt` reports negative CV R² across RF/XGB/LGBM in this setup, meaning feature importance rankings are informative for exploration but **not yet strong predictive proof** under current preprocessing/split strategy.

---

## 5) Practical reading order (fast understanding)
If you want to understand “what matters” quickly:

1. `column_metadata.csv` -> completeness + data types
2. `basic_eda/correlation_with_wind.csv` -> linear screening
3. `advanced_eda/feature_importance_all.csv` + `variable_importance_summary.txt` -> model-driven ranking
4. `shap_lime/consensus_important_features.txt` -> robust overlap across explainers
5. `comprehensive/comprehensive_summary.txt` + `vif_analysis.csv` -> final diagnostics and caveats

---

## 6) Bottom line
The key parameter families influencing analysis outcomes are:
- **Data inclusion/completeness filters** (especially non-null thresholds, join success)
- **Feature-engineering choices** (spline/cone/temporal derivatives)
- **Model hyperparameters and split/CV design**
- **Statistical thresholds** (p-value cutoffs, RI thresholds, anomaly contamination)
- **Category cleaning rules** (e.g., Panchang text normalization)

In the current run, consistent influence signals concentrate on geospatial (`latitude/longitude`), motion-shape (`spline_*`), and selected astronomical/Vedic timing features (`dinamana_minutes`, planetary degrees/speeds), with predictive strength still needing improvement.

---

## 7) Known Issues, Bugs Fixed & Interpretation Caveats
*(Added 2026-04-10 during pipeline audit)*

### CRITICAL BUGS FIXED

**[06_hypothesis_testing.py] Rahu = Ketu retrograde results were identical**
- Root cause: Rahu (North Lunar Node) and Ketu (South Lunar Node) always move together
  with identical speed magnitudes by astronomical definition — they are two ends of the
  same nodal axis. This is NOT a copy-paste error; it is an inherent property of the data.
- Fix: Added explicit documentation note in the Ketu retrograde result dict explaining
  this astronomical constraint. The two tests do NOT provide independent evidence.

**[05_comprehensive_eda.py] Vedic degrees correlated with wind using linear Pearson r**
- Root cause: Planetary degrees (0–360°) are cyclical/angular data. Naive Pearson r
  treats degree 359 as far from degree 1, which is mathematically incorrect.
- Fix: Replaced with circular-linear correlation (sin/cos decomposition). The updated
  `vedic_degree_correlations.csv` now reports `r_sin`, `r_cos`, and `r_circular_max`.
  Results from any previous version of this file are invalid for angular features.

**[03_advanced_eda.py] CV leakage: same storm in train and test folds**
- Root cause: Standard 5-fold CV splits randomly, allowing observations from the same
  storm to appear in both train and test, inflating R² via within-storm autocorrelation.
- Fix: Switched to `GroupKFold(n_splits=5)` grouped by `storm_id`. The updated
  `variable_importance_summary.txt` uses the corrected Group CV R² values.

### KEY FINDINGS FROM HYPOTHESIS TESTING (06_hypothesis_testing.py)

**H1: Vedic features DEGRADED prediction (−11.5% worse RMSE)**
- Adding all Vedic/panchang features to the meteorological baseline increased RMSE from
  4.043 to 4.509. Vedic features add noise, not signal, for predicting wind_change.
- The null hypothesis (Vedic features have no predictive lift) is NOT rejected.

**H2: Lunar phase vs RI (p=0.006)**
- This is nominally significant at α=0.05 but FAILS Bonferroni correction across the
  7 main hypothesis groups (adjusted α ≈ 0.007). Interpret with caution.

**Samvatsara ANOVA (p<10⁻¹³²) — spurious significance**
- Each Samvatsara maps to roughly one calendar year. The highly significant ANOVA simply
  confirms that storm intensity varies by year (known interannual climate variability —
  ENSO, PDO, etc.), NOT that Hindu calendar cycles influence storm intensity.

**Rahu/Ketu retrograde vs wind (p=0.040)**
- Barely below α=0.05 and FAILS Bonferroni correction (adjusted α = 0.05/7 ≈ 0.007).
  Not a reliable finding.

### PREDICTIVE MODELING (07_predictive_model_stack.py)

**Best model: `meteo_panchang` — astronomical features did NOT help**
- Adding the full scientific (astronomical/Vedic planetary) block to the meteo+panchang
  model worsened RMSE. The 193 astronomical features added noise.

**Spline feature leakage risk**
- `spline_bearing_deg`, `spline_curvature`, `spline_displacement`, `spline_length_deg`
  encode the smoothed forecast track geometry. Using these to predict next-step wind
  may introduce implicit leakage (knowing future trajectory implies future position).
  These are in the meteo block with a warning comment; the scientific block now
  explicitly excludes `spline_` prefix features.

**RI threshold inconsistency across scripts**
- `06_hypothesis_testing.py` uses 30 kt/24h rolling window (≈WMO standard of 35 kt/24h).
- `07_predictive_model_stack.py` uses adaptive next-step delta threshold (≥10 kt selected).
- These are fundamentally different definitions. Cross-script RI comparisons are invalid.

### R² CONTRADICTION EXPLAINED (03 vs 04)
- `03_advanced_eda.py` GroupKFold CV R²: −0.09 to −0.11 (worse than mean baseline).
- `04_shap_lime.py` XGBoost test R²: 0.76.
- The difference is: (1) different feature subsets (>80% vs >50% completeness),
  (2) random 80/20 split in 04 allows within-storm autocorrelation inflation,
  (3) GroupKFold in 03 is the correct out-of-storm generalization measure.
- The GroupKFold R² is the more honest estimate. The 0.76 figure is inflated.

### STATISTICAL BEST PRACTICES APPLIED
- `02_basic_eda.py`: Benjamini-Hochberg FDR correction added to `kruskal_wallis_wind.csv`
  (new column: `p_adjusted_bh`, `significant_after_bh_correction`). Use the corrected
  column for drawing conclusions, not the raw `significant_p05` column.
- `05_comprehensive_eda.py`: Retrograde analysis now includes Mann-Whitney U p-values
  and sample sizes in `retrograde_analysis.csv`.
- `06_hypothesis_testing.py`: Bonferroni-corrected α reported for all multi-test blocks.
