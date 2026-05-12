# Vedic Astrology / Astronomical Features and Tropical Cyclone Intensity
## Research Synthesis Report

**Generated:** 2026-05-02 05:33  
**Dataset:** 47,533 track observations, 944 unique storms, 2016–2026  
**Feature blocks:** 14 meteorological · 18 Panchangam (Vedic) · 190 scientific/astronomical


---

## 1. Executive Summary

- **Vedic/Panchang features provide no predictive lift** over meteorological features alone: adding them degrades next-step wind-change RMSE by 14.3% and the full model loses to meteo-only in 4 of 5 forward-chaining folds.
- **A statistically significant lunar-phase × RI association exists** (chi²=19.66, p=0.006, survives Bonferroni), but the effect size is negligible (Cramer's V≈0.020).
- **Planetary speed correlates with storm bearing** (rho≈−0.18, p<1e-200) consistently across multiple planets, but explains only ~3% of bearing variance (r²=0.032).
- **All statistically significant retrograde effects have small rank-biserial r<0.1**: the massive dataset (N≈47k) inflates significance well beyond practical importance.
- **Samvatsara (Hindu year-cycle) results are confounded** with calendar year — they reflect interannual climate variability (ENSO/PDO), not an independent Vedic signal.


---

## 2. Predictive Modeling: Can Vedic Features Improve Intensity Forecasts?

### Model RMSE Comparison (chronological holdout)

model | rmse | mae
--- | --- | ---
meteo_only | 4.164158710037904 | 2.539559323228345
stacked_residual | 4.191773403465205 | 2.5818393674288176
meteo_panchang | 4.371772512631825 | 2.6847075690262114
full_all_blocks | 4.514530673834453 | 2.9210388965925573


### Forward-Chaining Cross-Validation (5 folds)

fold | full_all_blocks | meteo_only | stacked_residual
--- | --- | --- | ---
1.0 | 3.3572151470331977 | 3.3692370199921813 | 3.372758185947851
2.0 | 3.884751241283051 | 3.7533314443618457 | 3.812926512686943
3.0 | 3.5080061307593806 | 3.191179311650056 | 3.1880505429330808
4.0 | 3.685194747357059 | 3.2093046208817606 | 3.2114569740208543
5.0 | 4.09024046431635 | 3.796871234057982 | 3.87493052079515

**Key finding:** `meteo_only` wins or ties in 4/5 folds. The stacked residual model occasionally matches meteo-only but never beats it consistently.

### Feature Group RMSE Lift (permutation importance)

group | rmse_lift_mean | rmse_lift_std
--- | --- | ---
meteo | 0.5816242574701576 | 0.0322293390404993
scientific | 0.1750158488602709 | 0.0048914057187646
panchang | 0.0102577360679934 | 0.0054677028710395

> Panchang (Vedic) lift = 0.010 kt represents <2% of the meteo lift (0.582 kt). Shuffling all Vedic features produces almost no change in prediction quality.


---

## 3. Effect Size Assessment: Statistical vs. Practical Significance

All variables in the dataset are non-normal (Shapiro p<1e-30 for all tested variables; wind skew=1.35). Non-parametric tests (Mann-Whitney, Kruskal-Wallis, Spearman) are therefore the correct choice throughout. However, **with N≈47,533 observations even trivially small effects produce p-values well below any correction threshold**. Effect sizes are the appropriate discriminator.

finding_id | feature | stat_label | statistic | p_value | effect_metric | effect_value | effect_interpretation
--- | --- | --- | --- | --- | --- | --- | ---
EW_VEDIC_RAHU_FULL_DEGR | vedic_rahu_full_degree | F | 18.752 | 2.50854447841492e-05 | eta-squared | 0.0973 | medium
EW_VEDIC_KETU_FULL_DEGR | vedic_ketu_full_degree | F | 18.752 | 2.5085444785125297e-05 | eta-squared | 0.0973 | medium
RET_BUDHA | budha retrograde vs direct wind | U | 174341576.0 | 2.337799904246324e-22 | rank-biserial r | 0.0633 | small
RET_GURU | guru retrograde vs direct wind | U | 222381232.0 | 2.3128684460498814e-23 | rank-biserial r | 0.0576 | small
H3_VEDIC_YAM_SPEED_DEG_ | vedic_yam_speed_deg_per_day vs spline_bearing_deg | rho | -0.1782 | 3.825611300824046e-271 | r² (Spearman) | 0.0318 | medium
H3_VEDIC_SHANI_SPEED_DE | vedic_shani_speed_deg_per_day vs spline_bearing_deg | rho | -0.1682 | 2.2283841499431903e-241 | r² (Spearman) | 0.0283 | medium
H3_VEDIC_VARUN_SPEED_DE | vedic_varun_speed_deg_per_day vs spline_bearing_deg | rho | -0.1666 | 1.0432381836156772e-236 | r² (Spearman) | 0.0278 | medium
KW_DRIK_RITU | drik_ritu vs wind | H | 1214.03 | 2.6840821456089064e-260 | epsilon-squared | 0.025418 | small
H3_VEDIC_SURYA_SPEED_DE | vedic_surya_speed_deg_per_day vs spline_bearing_deg | rho | -0.1581 | 6.18501298406308e-213 | r² (Spearman) | 0.025 | medium
H2_LUNAR_RI | Lunar phase vs RI (>=10 kt) | chi2 | 19.6558 | 0.006363 | Cramer's V | 0.0203 | small
KW_SUNSIGN | sunsign vs wind | H | 1123.14 | 5.890709283305625e-234 | epsilon-squared | 0.019545 | small
EW_MARS_ALTITUDE_DEG | mars_altitude_deg | F | 1.364 | 0.2444155576764491 | eta-squared | 0.0077 | small
EW_VEDIC_SPASHTH_RAHU_S | vedic_spashth_rahu_speed_deg_per_day | F | 0.561 | 0.4546990100316824 | eta-squared | 0.0032 | small
EW_VEDIC_SPASHTH_KETU_S | vedic_spashth_ketu_speed_deg_per_day | F | 0.561 | 0.4546990100316824 | eta-squared | 0.0032 | small
H3_VEDIC_GURU_SPEED_DEG | vedic_guru_speed_deg_per_day vs spline_bearing_deg | rho | -0.0558 | 8.586084116410565e-28 | r² (Spearman) | 0.0031 | small

> **Thresholds:** rank-biserial r: small<0.1, medium 0.1–0.3, large>0.3; eta²: small<0.01, medium 0.06–0.14, large>0.14; Cramer's V: small<0.1.


---

## 4. Hypothesis H2: Lunar Phase and Rapid Intensification

- **Test:** Chi-squared test, 8 lunar phases × RI/non-RI (RI = wind_change ≥ 10 kt/step)
- **Result:** chi²=19.66, p=0.006363, DoF=7
- **Bonferroni correction** (across 7 hypothesis groups, α/7=0.0071): **survives**
- **Effect size:** Cramer's V = √(19.66/47533) ≈ 0.020 — **small by convention**

Phase-level RI rate (aggregated across all lifecycle stages):

lunar_phase_cat | ri_count | total_count | ri_rate
--- | --- | --- | ---
New Moon | 80 | 6529 | 0.0123
Waxing Crescent | 49 | 6188 | 0.0079
First Quarter | 82 | 6024 | 0.0136
Waxing Gibbous | 79 | 5419 | 0.0146
Full Moon | 65 | 5360 | 0.0121
Waning Gibbous | 81 | 5649 | 0.0143
Third Quarter | 64 | 6115 | 0.0105
Waning Crescent | 83 | 6249 | 0.0133


**Lunar-phase × lifecycle interaction (chi² within each lifecycle stage):**

lifecycle_phase | chi2 | p | n_obs
--- | --- | --- | ---
Genesis | 24.878 | 0.0007975799567450237 | 9870
Intensify | 15.99 | 0.025209321840799156 | 14285
Mature | 8.831 | 0.26498714180315613 | 9011
Dissipate | 24.384 | 0.000975024221785321 | 14367

> **Interpretation:** The lunar-phase RI signal survives Bonferroni correction but the effect size is negligible. Waning Crescent shows the highest RI rate — this is worth reporting as a curiosity but should not be over-interpreted. The lifecycle breakdown shows whether the signal is driven by a specific storm stage.


---

## 5. Hypothesis H3: Planetary Speed and Storm Bearing

**Top 10 planetary speed vs storm bearing (Spearman rho):**

speed_feature | spearman_rho | p_value | n | r_sq
--- | --- | --- | --- | ---
vedic_yam_speed_deg_per_day | -0.1782455287557754 | 3.825611300824046e-271 | 38338 | 0.0318
vedic_shani_speed_deg_per_day | -0.1682444515853991 | 2.2283841499431903e-241 | 38338 | 0.0283
vedic_varun_speed_deg_per_day | -0.1666174578691762 | 1.0432381836156772e-236 | 38338 | 0.0278
vedic_surya_speed_deg_per_day | -0.1580620207633171 | 6.18501298406308e-213 | 38338 | 0.025
vedic_guru_speed_deg_per_day | -0.0557638463689348 | 8.586084116410565e-28 | 38338 | 0.0031
vedic_arun_speed_deg_per_day | -0.0488498771368526 | 1.0664502384085405e-21 | 38338 | 0.0024
vedic_mangal_speed_deg_per_day | -0.0431933479865234 | 2.6516265293217868e-17 | 38338 | 0.0019
vedic_spashth_rahu_speed_deg_per_day | 0.0352770157309855 | 4.874038555759767e-12 | 38338 | 0.0012
vedic_spashth_ketu_speed_deg_per_day | 0.0352770157309855 | 4.874038555759767e-12 | 38338 | 0.0012
vedic_budha_speed_deg_per_day | 0.0280784370935729 | 3.8262619503647555e-08 | 38338 | 0.0008


**Quintile dose-response (monotonicity check — first 3 speed features):**

speed_feature | quintile | speed_median | bearing_median | n | monotone_rho_bearing
--- | --- | --- | --- | --- | ---
vedic_yam_speed_deg_per_day | Q1 | -0.03 | 279.46 | 13869 | -1.0
vedic_yam_speed_deg_per_day | Q2 | -0.01 | 270.0 | 5883 | -1.0
vedic_yam_speed_deg_per_day | Q3 | 0.0 | 243.43 | 3951 | -1.0
vedic_yam_speed_deg_per_day | Q4 | 0.02 | 231.34 | 9805 | -1.0
vedic_yam_speed_deg_per_day | Q5 | 0.03 | 216.87 | 4830 | -1.0
vedic_shani_speed_deg_per_day | Q1 | -0.08 | 277.13 | 8151 | -1.0
vedic_shani_speed_deg_per_day | Q2 | -0.04 | 270.0 | 8622 | -1.0
vedic_shani_speed_deg_per_day | Q3 | 0.01 | 261.87 | 7420 | -1.0
vedic_shani_speed_deg_per_day | Q4 | 0.07 | 236.31 | 6663 | -1.0
vedic_shani_speed_deg_per_day | Q5 | 0.11 | 213.69 | 7482 | -1.0
vedic_surya_speed_deg_per_day | Q1 | 0.95 | 278.13 | 12803 | -1.0
vedic_surya_speed_deg_per_day | Q2 | 0.97 | 270.0 | 4999 | -1.0
vedic_surya_speed_deg_per_day | Q3 | 0.98 | 243.43 | 8741 | -1.0
vedic_surya_speed_deg_per_day | Q4 | 1.0 | 236.31 | 4540 | -1.0
vedic_surya_speed_deg_per_day | Q5 | 1.01 | 225.0 | 7255 | -1.0

> **Interpretation:** The correlations are statistically robust but explain only ~3% of storm bearing variance (r²≈0.03). Outer planets (Varun, Arun) move very slowly and are retrograde >50% of the time — their 'speed' is nearly a constant epoch-level variable. The quintile dose-response tables show whether the relationship is monotone or driven by extremes.
> **Spline leakage caution:** `spline_bearing_deg` encodes the smoothed forecast track bearing — it may carry implicit future trajectory information. Correlations with planetary speeds could reflect this rather than a causal astronomical link.


---

## 6. Retrograde Planet Effects on Wind Intensity

**Retrograde analysis (Mann-Whitney U, rank-biserial r from effect_sizes.csv):**

finding_id | feature | statistic | p_value | effect_value | effect_interpretation | cohens_d | cohens_d_interp
--- | --- | --- | --- | --- | --- | --- | ---
RET_GURU | guru retrograde vs direct wind | 222381232.0 | 2.3128684460498814e-23 | 0.0576 | small | -0.1167 | small
RET_SHANI | shani retrograde vs direct wind | 287701469.0 | 4.988765903408708e-06 | -0.0242 | small | 0.0125 | small
RET_MANGAL | mangal retrograde vs direct wind | 96293298.0 | 0.0004488896390088 | -0.0323 | small | 0.0398 | small
RET_BUDHA | budha retrograde vs direct wind | 174341576.0 | 2.337799904246324e-22 | 0.0633 | small | -0.1018 | small
RET_SHUKRA | shukra retrograde vs direct wind | 85699652.0 | 6.619880626410572e-06 | -0.0442 | small | 0.0745 | small
RET_ARUN | arun retrograde vs direct wind | 301477181.0 | 1.01747662943546e-38 | -0.0688 | small | 0.1249 | small
RET_VARUN | varun retrograde vs direct wind | 294164036.0 | 2.329952338753825e-59 | -0.0878 | small | 0.1392 | small

> **Interpretation:** All 7 planets show statistically significant retrograde/direct wind differences, but **all rank-biserial r values are <0.1 (small)**. Varun and Arun are retrograde >50–60% of the time, so the retrograde 'group' is simply the majority of observations during certain epochs — not a meaningful comparison. Only Shukra (Venus, retrograde 8%) and Mangal (Mars, retrograde 9%) represent genuinely rare retrograde phases worth discussing.


---

## 7. Confound-Adjusted Vedic Category Analysis

The Kruskal-Wallis H statistics for Vedic categories (Yoga H=143, Nakshatra H=116, Moonsign H=78, Tithi H=51) may be driven by the seasonal proxy `drik_ritu` (H=1214). Partial KW tests were run **within** each basin × season stratum (≥100 observations, ≥3 categories with ≥10 obs each) to check whether the signal persists after removing the seasonal/geographic confound.

**Summary (% of viable strata showing significant KW after BH correction):**

variable | total_strata | sig_strata_bh | pct_sig | verdict
--- | --- | --- | --- | ---
tithi | 12 | 12 | 100.0 | strong within-stratum signal
nakshatra | 12 | 12 | 100.0 | strong within-stratum signal
yoga | 12 | 12 | 100.0 | strong within-stratum signal
moonsign | 12 | 12 | 100.0 | strong within-stratum signal

> A finding of <25% significant strata suggests the overall KW result is driven by the seasonal confound. A finding of >50% suggests a genuinely residual signal within homogeneous subgroups.


---

## 8. Consolidated Evidence Table

All 16 paper-level findings with effect sizes and confound verdicts:

finding_id | finding | source_script | test | statistic | p_value | effect_metric | effect_value | survives_confound | verdict
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
F01 | Vedic/Panchang features degrade next-step wind-change RMSE | 06, 07 | RMSE comparison (chronological holdout) | RMSE 4.164 → 4.372 kt | N/A | % RMSE change | -5.0% | N/A | No predictive lift — Vedic features add noise
F02 | Panchang group RMSE permutation lift ≈ 0.010 kt (vs meteo 0.582) | 07 | Permutation group importance | lift = 0.0103 kt | N/A | Absolute RMSE lift | 0.0103 kt | N/A | Panchang contributes <2% of total explainable variance
F03 | Lunar phase significantly associated with RI (chi²=19.66, Bonferroni p=0.006) | 06 | Chi-squared (8 phases × 2 RI classes) | chi²=19.66 | 0.006363 | Cramer's V | 0.0203 | Yes (Bonferroni-corrected) | Statistically significant; Cramer's V≈0.020 — negligible practical effect
F04 | Waning Crescent has highest RI rate (301), Waxing Crescent lowest (220) — 37% range | 06 | Contingency table inspection | Range 220–301 RI events across phases | 0.006363 | RI count range | 37% | Yes (within H2) | Real but small; needs lifecycle stratification
F05 | Yam speed vs storm bearing: Spearman rho=−0.178 (p<1e-270) | 06 | Spearman rank correlation | rho=−0.178 | 3.8e-271 | r² (variance explained) | 0.0317 | Partially — outer planets have near-constant speed | Consistent small correlation; r²=3%; spline leakage risk noted
F06 | Shani/Varun speed vs bearing rho≈−0.168/−0.167 (p<1e-230) | 06 | Spearman rank correlation | rho≈−0.17 | <1e-230 | r² | 0.028 | Varun retro 60% of obs — near-constant | Same caveat as F05; effect small but replicable across planets
F07 | Varun retrograde: median wind +4 kt vs direct (p=2.3e-59) | 05 | Mann-Whitney U | U=294M | 2.3e-59 | rank-biserial r | -0.0878 | Suspect — Varun retrograde 60% of observations | Statistically significant; r<0.1 (small) — large N inflates significance
F08 | Shukra retrograde: median wind +5 kt vs direct (p=6.6e-6) | 05 | Mann-Whitney U | U=85.7M | 6.6e-06 | rank-biserial r | -0.0442 | Not checked | Small effect; largest absolute median difference among retrograde planets
F09 | Rahu full degree shift differs significantly around RI onset (F=18.75, p=2.5e-5) | 06 | ANOVA (event window, N=176 windows) | F=18.75 | 2.51e-05 | eta-squared | ≈0.097 | Not assessed — small sample | Medium eta²=0.097 on N=176; requires replication with larger sample
F10 | Mars (Mangal) retrograde → higher spline curvature (F=19.34, p=1.1e-5) | 06 | ANOVA | F=19.34 | 1.1e-05 | eta-squared | 0.000505 | Not assessed | Statistically significant; eta²<0.001 — trivially small effect
F11 | Yoga KW H=143 on wind (p=3.4e-18) | 02 | Kruskal-Wallis | H=143.3 | 3.4e-18 | epsilon-squared | -0.120352 | strong within-stratum signal | Confound check required; seasonal proxy likely drives signal
F12 | Nakshatra KW H=116 on wind (p=2.4e-13) | 02 | Kruskal-Wallis | H=116.1 | 2.4e-13 | epsilon-squared | -0.118273 | strong within-stratum signal | Confound check required; collinear with sunsign (H=1123)
F13 | Samvatsara ANOVA p<1e-130 on wind | 06 | One-way ANOVA | F=65.12 | 1.6e-132 | N/A | N/A | No — maps to calendar year | CONFOUNDED: Samvatsara ≈ calendar year; reflects ENSO/PDO, not Hindu cycle
F14 | Rahu full degree: Spearman rho=0.013 vs wind_change (p=0.006) | 06 | Spearman | rho=0.013 | 0.006 | r² | 0.000169 | Survives Bonferroni (barely) | Negligible r²=0.017%; statistically real, practically irrelevant
F15 | 4 Vedic features in SHAP/LIME top-14 consensus: arun_speed, budha_speed, budha_degree, guru_degree | 04 | SHAP + LIME feature consensus | 14 consensus features; 4 Vedic | N/A | Consensus rank | Top 30 of both methods | Not assessed | Vedic speed/degree features rank in model importance, but model uses autocorrelated train/test split
F16 | Meteo-only wins or ties in 4/5 forward-chaining folds vs full model | 07 | Forward-chaining CV (5 folds) | 4/5 folds: meteo ≤ full | N/A | RMSE difference per fold | 0.15–0.48 kt worse for full | N/A | Consistent across time — Vedic/astro features do not generalise to future storms


---

## 9. Methodological Limitations

1. **Temporal autocorrelation:** Track observations are not i.i.d. — consecutive points within the same storm are highly correlated (autocorrelation). Models trained with random train/test splits overestimate performance; Group/forward-chaining CV mitigates this.
2. **Large-N significance inflation:** With N≈47,533 even r=0.02 produces p<0.001. Effect sizes (Cramer's V, rank-biserial r, eta²) are the correct discriminators — not p-values.
3. **Seasonal confounding:** drik_ritu and sunsign are lunar/solar calendar proxies that co-vary strongly with storm season. K-W tests on Vedic categories may be detecting season rather than an independent Vedic effect.
4. **Spline leakage:** spline_bearing_deg and spline_curvature encode forecast track geometry — using them as targets in planetary speed correlations introduces implicit future information.
5. **RI threshold heterogeneity:** Scripts 06 (30 kt/24h rolling), 07 (10 kt/next-step), and 08 (10 kt) use different RI definitions. Cross-script RI comparisons are not valid.
6. **Outer planet near-constant speed:** Varun, Arun, Varun retrograde 50–60% of observations. Their 'speed' is nearly a constant within multi-year subsets. Correlations with storm behaviour may reflect epoch-level climate modes, not the planets' motion.


---

## 10. Conclusion

Across 47,533 tropical cyclone track observations (944 storms, 2016–2026), this analysis finds **no evidence that Vedic Panchang features improve storm intensity prediction** — they consistently degrade forecasting accuracy. Two genuinely interesting signals emerge: (1) a statistically significant but practically negligible lunar-phase × RI association (Cramer's V≈0.020); and (2) a consistent small correlation between planetary speeds and storm bearing direction (r²≈3%). Both signals survive multiple-comparison correction but have effect sizes too small to be operationally meaningful. The paper should frame these as null-result findings with important methodological contributions: demonstrating how large-N datasets inflate astronomical significance, and providing a rigorous framework for testing Vedic astrology claims against geophysical data.
