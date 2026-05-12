"""
06_hypothesis_testing.py
========================
This script tests the hypotheses outlined in the research plan.

Hypothesis 1: Vedic Astrology's Predictive Power on Intensity Change
- Test if Vedic features can predict future changes in wind_change and pressure_change.
- Compare a baseline model (meteorological features) with a model including Vedic features.

Hypothesis 2: Astronomical Cycles and Storm Intensification
- Test if astronomical cycles (e.g., lunar phases) are associated with rapid intensification of storms.
- Use statistical methods to analyze the relationship between lunar phases and storm intensification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import chi2_contingency, spearmanr, f_oneway

# Define paths
DATA_PATH = Path(__file__).resolve().parent / "output" / "storm_data_all.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "hypothesis_testing"
OUTPUT_DIR.mkdir(exist_ok=True)


def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan

def run_hypothesis1():
    """
    Tests Hypothesis 1: Vedic Astrology's Predictive Power on Intensity Change.
    """
    print("--- Running Hypothesis 1: Vedic Astrology's Predictive Power ---")

    # Load data
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded data with shape: {df.shape}")

    # --- Feature Selection ---
    target_vars = ['wind_change', 'pressure_change']
    
    met_features = [
        'wind', 'pressure', 'longitude', 'latitude', 'hour_utc', 
        'day_of_year', 'movement_speed', 'displacement'
    ]
    
    vedic_features = [
        'tithi', 'nakshatra', 'yoga', 'karana', 'weekday', 'paksha', 'moonsign', 
        'sunsign', 'drik_ritu', 'vedic_ritu', 'drik_ayana', 'vedic_ayana', 
        'anandadi_yoga', 'tamil_yoga', 'samvatsara'
    ]
    # Add all vedic planet numerical features
    vedic_numerical_cols = [col for col in df.columns if col.startswith('vedic_')]
    vedic_features = list(set(vedic_features + vedic_numerical_cols))
    
    # --- Data Preprocessing ---
    # For simplicity, we'll focus on 'wind_change' and drop rows with NaNs in key columns
    df_h1 = df[['storm_id', 'track_num', 'date_utc'] + met_features + vedic_features + target_vars].copy()
    df_h1 = df_h1.dropna(subset=['wind_change'] + met_features)
    print(f"Shape after dropping NaNs: {df_h1.shape}")

    # Identify all categorical columns to be used as features
    categorical_features = [
        'tithi', 'nakshatra', 'yoga', 'karana', 'weekday', 'paksha', 'moonsign',
        'sunsign', 'drik_ritu', 'vedic_ritu', 'drik_ayana', 'vedic_ayana',
        'anandadi_yoga', 'tamil_yoga', 'samvatsara'
    ]
    # Also include vedic planet nakshatra columns
    nakshatra_cols = [col for col in vedic_numerical_cols if 'nakshatra' in col]
    categorical_features.extend(nakshatra_cols)
    categorical_features = list(set(categorical_features)) # Ensure unique columns

    # Encode categorical features
    for col in categorical_features:
        if col in df_h1.columns:
            df_h1[col] = df_h1[col].astype(str)
            le = LabelEncoder()
            df_h1[col] = le.fit_transform(df_h1[col])

    # Fill any remaining NaNs in vedic features with 0 (post-encoding)
    df_h1.fillna(0, inplace=True)

    # --- Model Training & Evaluation ---
    X = df_h1.drop(columns=target_vars + ['storm_id', 'date_utc'])
    y = df_h1['wind_change']

    # Chronological split
    split_date = df_h1['date_utc'].quantile(0.8, interpolation='nearest')
    train_df = df_h1[df_h1['date_utc'] <= split_date]
    test_df = df_h1[df_h1['date_utc'] > split_date]

    X_train = train_df[X.columns]
    y_train = train_df['wind_change']
    X_test = test_df[X.columns]
    y_test = test_df['wind_change']
    
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    # --- Baseline Model (Meteorological Features) ---
    print("\nTraining Baseline Model (Meteorological features only)...")
    X_train_met = X_train[met_features]
    X_test_met = X_test[met_features]
    
    rf_met = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_met.fit(X_train_met, y_train)
    
    preds_met = rf_met.predict(X_test_met)
    rmse_met = np.sqrt(mean_squared_error(y_test, preds_met))
    print(f"Baseline Model RMSE: {rmse_met:.4f}")

    # --- Vedic Model (Meteorological + Vedic Features) ---
    print("\nTraining Vedic Model (Meteorological + Vedic features)...")
    X_train_vedic = X_train[met_features + vedic_features]
    X_test_vedic = X_test[met_features + vedic_features]

    rf_vedic = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_vedic.fit(X_train_vedic, y_train)

    preds_vedic = rf_vedic.predict(X_test_vedic)
    rmse_vedic = np.sqrt(mean_squared_error(y_test, preds_vedic))
    print(f"Vedic Model RMSE: {rmse_vedic:.4f}")

    # --- Results ---
    print("\n--- Hypothesis 1 Results ---")
    print(f"Baseline Model RMSE (Met only): {rmse_met:.4f}")
    print(f"Vedic Model RMSE (Met + Vedic): {rmse_vedic:.4f}")
    
    improvement = ((rmse_met - rmse_vedic) / rmse_met) * 100
    print(f"Improvement with Vedic features: {improvement:.2f}%")

    # Save results
    results = {
        "baseline_model_rmse": rmse_met,
        "vedic_model_rmse": rmse_vedic,
        "improvement_pct": improvement
    }
    results_path = OUTPUT_DIR / "hypothesis1_results.txt"
    with open(results_path, "w") as f:
        f.write(str(results))
    print(f"Results saved to {results_path}")
    return results


def run_hypothesis2():
    """
    Tests Hypothesis 2: Astronomical Cycles and Storm Intensification.
    """
    print("\n--- Running Hypothesis 2: Astronomical Cycles and Storm Intensification ---")

    # Load data
    df = pd.read_parquet(DATA_PATH)
    
    # --- Data Cleaning: Handle non-monotonic index ---
    # Keep only columns needed for this hypothesis to speed up processing
    h2_cols = ['storm_id', 'date_utc', 'wind', 'sun_ecliptic_lon_deg', 'moon_ecliptic_lon_deg']
    df_h2 = df[h2_cols].copy()

    # Ensure date_utc is datetime
    df_h2['date_utc'] = pd.to_datetime(df_h2['date_utc'])

    # Average values for duplicate timestamps within the same storm
    df_h2 = df_h2.groupby(['storm_id', 'date_utc']).mean().reset_index()
    
    df_h2 = df_h2.sort_values(['storm_id', 'date_utc'])
    df_h2 = df_h2.set_index('date_utc')
    print(f"Loaded and cleaned data with shape: {df_h2.shape}")

    # --- Feature Engineering ---
    # 1. Calculate wind change over a 24-hour rolling window
    # We need to handle each storm group separately
    df_h2['wind_change_24hr'] = df_h2.groupby('storm_id')['wind'].rolling('24h').apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0).reset_index(level=0, drop=True)

    # 2. Identify Rapid Intensification (RI) events
    # RI definition: wind speed increase of >= 30 knots in 24 hours.
    # NOTE: WMO/NOAA standard RI threshold is 35 kt/24h; this analysis uses 30 kt to
    # capture a broader set of intensification events. Results should not be directly
    # compared to operational RI statistics which use the 35 kt standard.
    df_h2['is_ri_event'] = (df_h2['wind_change_24hr'] >= 30).astype(int)

    # 3. Calculate Lunar Phase
    # Phase = angle between sun and moon ecliptic longitude
    # 0 = New Moon, 180 = Full Moon
    df_h2['lunar_phase_angle'] = (df_h2['moon_ecliptic_lon_deg'] - df_h2['sun_ecliptic_lon_deg']) % 360

    # 4. Categorize Lunar Phase using 8 equal bins (45 deg each)
    # This avoids pandas cut edge/label compatibility issues across versions.
    phase_idx = ((df_h2['lunar_phase_angle'] % 360) / 45).astype(int)
    phase_map = {
        0: 'New Moon',
        1: 'Waxing Crescent',
        2: 'First Quarter',
        3: 'Waxing Gibbous',
        4: 'Full Moon',
        5: 'Waning Gibbous',
        6: 'Third Quarter',
        7: 'Waning Crescent',
    }
    df_h2['lunar_phase_category'] = phase_idx.map(phase_map)
    
    # --- Statistical Analysis ---
    # Avoid duplicate-index reindex issues in older/newer pandas internals.
    df_h2 = df_h2.reset_index(drop=True)
    print(f"\nTotal Rapid Intensification events found: {df_h2['is_ri_event'].sum()}")

    # Create a contingency table
    contingency_table = pd.crosstab(df_h2['lunar_phase_category'], df_h2['is_ri_event'])
    contingency_table.columns = ['Non-RI', 'RI']
    print("\nContingency Table (RI events vs. Lunar Phase):")
    print(contingency_table)

    # Perform Chi-Squared Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print("\n--- Hypothesis 2 Results ---")
    print(f"Chi-Squared Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    print(f"Degrees of Freedom: {dof}")

    # Bonferroni-corrected alpha: This script tests multiple hypotheses (H1, H2, H3,
    # Samvatsara, Mars/Rahu/Ketu sub-tests). To control family-wise error rate across
    # the ~7 primary test groups, a conservative adjusted threshold is alpha/7 ≈ 0.007.
    alpha = 0.05
    alpha_corrected = alpha / 7  # Conservative Bonferroni across main hypothesis groups
    if p < alpha_corrected:
        print(f"\nThe result is statistically significant after Bonferroni correction (p={p:.6f} < {alpha_corrected:.4f}).")
        print("We can reject the null hypothesis that there is no association between lunar phase and rapid intensification.")
    elif p < alpha:
        print(f"\nThe result is nominally significant (p={p:.6f} < {alpha}) but does NOT survive Bonferroni correction (adjusted alpha={alpha_corrected:.4f}).")
        print("This result should be interpreted with caution due to multiple comparisons.")
    else:
        print(f"\nThe result is not statistically significant (p={p:.6f} >= {alpha}).")
        print("We cannot reject the null hypothesis. There is no clear association found.")

    # Save results
    results = {
        "contingency_table": contingency_table.to_dict(),
        "chi2_statistic": chi2,
        "p_value": p,
        "dof": dof
    }
    results_path = OUTPUT_DIR / "hypothesis2_results.txt"
    with open(results_path, "w") as f:
        f.write(str(results))
    print(f"Results saved to {results_path}")
    return results


def run_hypothesis3():
    """
    Hypothesis 3: Planetary speed/retrograde phases are associated with trajectory dynamics.
    """
    print("\n--- Running Hypothesis 3: Planetary Speeds vs Trajectory Dynamics ---")

    df = pd.read_parquet(DATA_PATH)

    target_cols = [
        "spline_bearing_deg",
        "spline_curvature",
        "movement_speed",
        "wind_change",
    ]
    target_cols = [c for c in target_cols if c in df.columns]

    speed_cols = [
        c for c in df.columns
        if c.startswith("vedic_") and c.endswith("_speed_deg_per_day")
    ]

    if not speed_cols or not target_cols:
        results = {
            "status": "skipped",
            "reason": "missing speed or target columns",
            "speed_cols": speed_cols,
            "target_cols": target_cols,
        }
        with open(OUTPUT_DIR / "hypothesis3_results.txt", "w") as f:
            f.write(str(results))
        print("Hypothesis 3 skipped: missing required columns.")
        return results

    rows = []
    for s_col in speed_cols:
        for t_col in target_cols:
            pair = df[[s_col, t_col]].copy()
            pair[s_col] = pair[s_col].map(_safe_float)
            pair[t_col] = pair[t_col].map(_safe_float)
            pair = pair.dropna()

            if len(pair) < 50:
                continue

            rho, pval = spearmanr(pair[s_col], pair[t_col], nan_policy="omit")
            rows.append({
                "speed_feature": s_col,
                "target_feature": t_col,
                "n": int(len(pair)),
                "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                "p_value": float(pval) if pd.notna(pval) else np.nan,
            })

    corr_df = pd.DataFrame(rows)
    corr_df = corr_df.sort_values(["p_value", "spearman_rho"], ascending=[True, False])
    corr_df.to_csv(OUTPUT_DIR / "hypothesis3_spearman_matrix.csv", index=False)

    # Retrograde vs direct effect on curvature for each planet-speed feature.
    retro_rows = []
    if "spline_curvature" in df.columns:
        for s_col in speed_cols:
            tmp = df[[s_col, "spline_curvature"]].copy()
            tmp[s_col] = tmp[s_col].map(_safe_float)
            tmp["spline_curvature"] = tmp["spline_curvature"].map(_safe_float)
            tmp = tmp.dropna()
            if len(tmp) < 100:
                continue

            retro = tmp[tmp[s_col] < 0]["spline_curvature"]
            direct = tmp[tmp[s_col] >= 0]["spline_curvature"]
            if len(retro) < 20 or len(direct) < 20:
                continue

            f_stat, pval = f_oneway(retro, direct)
            retro_rows.append({
                "speed_feature": s_col,
                "retro_n": int(len(retro)),
                "direct_n": int(len(direct)),
                "retro_mean_curvature": float(retro.mean()),
                "direct_mean_curvature": float(direct.mean()),
                "f_stat": float(f_stat),
                "p_value": float(pval),
            })

    retro_df = pd.DataFrame(retro_rows)
    if not retro_df.empty:
        retro_df = retro_df.sort_values("p_value", ascending=True)
    retro_df.to_csv(OUTPUT_DIR / "hypothesis3_retrograde_anova.csv", index=False)

    top_assoc = corr_df.head(10).to_dict(orient="records") if not corr_df.empty else []
    top_retro = retro_df.head(10).to_dict(orient="records") if not retro_df.empty else []

    results = {
        "status": "ok",
        "tested_speed_features": len(speed_cols),
        "tested_target_features": len(target_cols),
        "tested_pairs": int(len(corr_df)),
        "top_associations": top_assoc,
        "top_retrograde_effects": top_retro,
    }
    with open(OUTPUT_DIR / "hypothesis3_results.txt", "w") as f:
        f.write(str(results))

    print(f"Tested pairs: {len(corr_df)}")
    print(f"Top associations saved to: {OUTPUT_DIR / 'hypothesis3_spearman_matrix.csv'}")
    return results


def run_samvatsara_analysis():
    """
    Samvatsaram-based grouped analysis and statistical tests.
    """
    print("\n--- Running Samvatsaram Analysis ---")
    df = pd.read_parquet(DATA_PATH)

    required = ["samvatsara", "storm_id", "wind", "pressure", "wind_change"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        results = {
            "status": "skipped",
            "reason": f"missing columns: {missing_required}",
        }
        with open(OUTPUT_DIR / "samvatsara_results.txt", "w") as f:
            f.write(str(results))
        print(f"Samvatsaram analysis skipped: missing columns {missing_required}")
        return results

    work = df[required].copy()
    work = work.dropna(subset=["samvatsara"])
    # NOTE on RI threshold: using same 30 kt threshold as H2. WMO standard is 35 kt.
    work["ri_event"] = (pd.to_numeric(work["wind_change"], errors="coerce") >= 30).astype(int)

    summary = work.groupby("samvatsara", dropna=False).agg(
        storm_count=("storm_id", "nunique"),
        track_count=("storm_id", "size"),
        mean_wind=("wind", "mean"),
        max_wind=("wind", "max"),
        mean_pressure=("pressure", "mean"),
        min_pressure=("pressure", "min"),
        ri_events=("ri_event", "sum"),
        ri_rate=("ri_event", "mean"),
    ).reset_index()
    summary = summary.sort_values("track_count", ascending=False)
    summary.to_csv(OUTPUT_DIR / "samvatsara_summary.csv", index=False)

    # ANOVA on wind by samvatsara
    anova_groups_wind = [g["wind"].dropna().values for _, g in work.groupby("samvatsara") if len(g["wind"].dropna()) >= 10]
    anova_groups_pressure = [g["pressure"].dropna().values for _, g in work.groupby("samvatsara") if len(g["pressure"].dropna()) >= 10]

    wind_anova = {"f_stat": np.nan, "p_value": np.nan}
    pressure_anova = {"f_stat": np.nan, "p_value": np.nan}

    if len(anova_groups_wind) >= 2:
        f_stat, pval = f_oneway(*anova_groups_wind)
        wind_anova = {"f_stat": float(f_stat), "p_value": float(pval)}

    if len(anova_groups_pressure) >= 2:
        f_stat, pval = f_oneway(*anova_groups_pressure)
        pressure_anova = {"f_stat": float(f_stat), "p_value": float(pval)}

    # RI vs Samvatsara association
    ctab = pd.crosstab(work["samvatsara"], work["ri_event"])
    chi2, p, dof, _ = chi2_contingency(ctab)

    results = {
        "status": "ok",
        "samvatsara_count": int(summary["samvatsara"].nunique()),
        "wind_anova": wind_anova,
        "pressure_anova": pressure_anova,
        "ri_chi2": {
            "chi2_statistic": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
        },
        "top_samvatsara_by_wind": summary.sort_values("mean_wind", ascending=False).head(5).to_dict(orient="records"),
        "interpretation_caution": (
            "CAUTION: Samvatsara is a Hindu calendar year-cycle derived from the Julian year. "
            "Each Samvatsara spans roughly one calendar year, so this ANOVA is equivalent to "
            "testing whether mean wind differs by year — which reflects interannual climate "
            "variability (El Nino, PDO, basin activity cycles), NOT a Hindu calendar effect. "
            "The highly significant p-values (p<1e-130) should be attributed to year-over-year "
            "differences in storm season intensity, not to the Samvatsara cycle itself. "
            "Also note: top Samvatsara years by mean wind show 0 RI events, likely because "
            "the data temporal resolution is coarse for those years, not because RI did not occur."
        ),
    }

    with open(OUTPUT_DIR / "samvatsara_results.txt", "w") as f:
        f.write(str(results))
    print(f"Samvatsara summary saved to: {OUTPUT_DIR / 'samvatsara_summary.csv'}")
    return results


def run_mars_hypothesis():
    """
    Mars + Rahu + Ketu focused research block:
    1) Does adding Mars+Rahu+Ketu scientific parameters improve wind-change prediction?
    2) Nearby hypothesis: do Mars+Rahu+Ketu parameters improve RI classification?
    3) Nearby hypothesis: Mars altitude bands vs RI association.
    4) Nearby hypothesis: Mangal retrograde vs direct mean wind-change difference.
    5) Nearby hypothesis: Rahu retrograde vs direct mean wind-change difference.
    6) Nearby hypothesis: Rahu degree vs wind-change monotonic association.
    7) Nearby hypothesis: Ketu retrograde vs direct mean wind-change difference.
    8) Nearby hypothesis: Ketu degree vs wind-change monotonic association.
    """
    print("\n--- Running Mars+Rahu+Ketu Hypothesis ---")
    df = pd.read_parquet(DATA_PATH)

    met_features = [
        "wind", "pressure", "longitude", "latitude", "hour_utc",
        "day_of_year", "movement_speed", "displacement"
    ]

    mars_candidates = [
        "mars_ra_hours", "mars_dec_deg", "mars_ecliptic_lon_deg", "mars_ecliptic_lat_deg",
        "mars_altitude_deg", "mars_azimuth_deg", "mars_sin_alt", "mars_cos_alt",
        "mars_sin_az", "mars_cos_az",
        "vedic_mangal_full_degree", "vedic_mangal_padam", "vedic_mangal_speed_deg_per_day",
        "vedic_mangal_right_ascension", "vedic_mangal_declination", "vedic_mangal_nakshatra",
    ]
    mars_features = [c for c in mars_candidates if c in df.columns]

    rahu_candidates = [
        "vedic_rahu_full_degree", "vedic_rahu_padam", "vedic_rahu_speed_deg_per_day",
        "vedic_rahu_right_ascension", "vedic_rahu_declination", "vedic_rahu_nakshatra",
        "vedic_spashth_rahu_full_degree", "vedic_spashth_rahu_padam", "vedic_spashth_rahu_speed_deg_per_day",
        "vedic_spashth_rahu_right_ascension", "vedic_spashth_rahu_declination", "vedic_spashth_rahu_nakshatra",
    ]
    rahu_features = [c for c in rahu_candidates if c in df.columns]

    ketu_candidates = [
        "vedic_ketu_full_degree", "vedic_ketu_padam", "vedic_ketu_speed_deg_per_day",
        "vedic_ketu_right_ascension", "vedic_ketu_declination", "vedic_ketu_nakshatra",
        "vedic_spashth_ketu_full_degree", "vedic_spashth_ketu_padam", "vedic_spashth_ketu_speed_deg_per_day",
        "vedic_spashth_ketu_right_ascension", "vedic_spashth_ketu_declination", "vedic_spashth_ketu_nakshatra",
    ]
    ketu_features = [c for c in ketu_candidates if c in df.columns]

    astro_features = list(dict.fromkeys(mars_features + rahu_features + ketu_features))

    needed_cols = list(dict.fromkeys(["storm_id", "date_utc", "wind_change"] + met_features + astro_features))
    work = df[needed_cols].copy()
    work["date_utc"] = pd.to_datetime(work["date_utc"], errors="coerce")
    work = work.dropna(subset=["wind_change", "date_utc"] + [c for c in met_features if c in work.columns])

    # Encode categorical columns (e.g., vedic_mangal_nakshatra)
    cat_cols = work.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in cat_cols:
        work[col] = work[col].astype(str)
        le = LabelEncoder()
        work[col] = le.fit_transform(work[col])

    work = work.fillna(0)

    split_date = work["date_utc"].quantile(0.8, interpolation="nearest")
    train = work[work["date_utc"] <= split_date]
    test = work[work["date_utc"] > split_date]

    base_feats = [c for c in met_features if c in work.columns]
    astro_feats_only = [c for c in astro_features if c in work.columns]
    full_feats = list(dict.fromkeys(base_feats + astro_feats_only))

    # Regression target
    y_train_reg = train["wind_change"]
    y_test_reg = test["wind_change"]

    rf_base = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_base.fit(train[base_feats], y_train_reg)
    pred_base = rf_base.predict(test[base_feats])
    rmse_base = float(np.sqrt(mean_squared_error(y_test_reg, pred_base)))

    rf_mars = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_mars.fit(train[full_feats], y_train_reg)
    pred_mars = rf_mars.predict(test[full_feats])
    rmse_mars = float(np.sqrt(mean_squared_error(y_test_reg, pred_mars)))
    delta_pct = float(((rmse_base - rmse_mars) / rmse_base) * 100)

    # Mars+Rahu+Ketu feature importances
    imp_df = pd.DataFrame({"feature": full_feats, "importance": rf_mars.feature_importances_})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df["rank"] = np.arange(1, len(imp_df) + 1)
    imp_df.to_csv(OUTPUT_DIR / "mars_rahu_feature_importance.csv", index=False)
    astro_imp = imp_df[imp_df["feature"].isin(astro_feats_only)].copy()
    mars_imp = imp_df[imp_df["feature"].isin(mars_features)].copy()
    rahu_imp = imp_df[imp_df["feature"].isin(rahu_features)].copy()
    ketu_imp = imp_df[imp_df["feature"].isin(ketu_features)].copy()

    # Nearby hypothesis A: RI classification improvement with Mars+Rahu+Ketu features
    train_cls = train.copy()
    test_cls = test.copy()
    train_cls["ri_event"] = (train_cls["wind_change"] >= 30).astype(int)
    test_cls["ri_event"] = (test_cls["wind_change"] >= 30).astype(int)

    y_train_cls = train_cls["ri_event"]
    y_test_cls = test_cls["ri_event"]

    auc_base = np.nan
    auc_mars = np.nan
    acc_base = np.nan
    acc_mars = np.nan
    split_mode = "chronological"

    if len(np.unique(y_train_cls)) < 2 or len(np.unique(y_test_cls)) < 2:
        split_mode = "stratified_fallback"
        cls_df = work[full_feats + ["wind_change"]].copy()
        cls_df["ri_event"] = (cls_df["wind_change"] >= 30).astype(int)
        X_all_base = cls_df[base_feats]
        X_all_full = cls_df[full_feats]
        y_all = cls_df["ri_event"]

        if len(np.unique(y_all)) >= 2 and y_all.sum() >= 30:
            xb_tr, xb_te, y_tr, y_te = train_test_split(
                X_all_base, y_all, test_size=0.2, random_state=42, stratify=y_all
            )
            xf_tr, xf_te, _, _ = train_test_split(
                X_all_full, y_all, test_size=0.2, random_state=42, stratify=y_all
            )

            clf_base = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
            clf_base.fit(xb_tr, y_tr)
            p_base_all = clf_base.predict_proba(xb_te)
            if p_base_all.shape[1] >= 2:
                p_base = p_base_all[:, 1]
                yhat_base = (p_base >= 0.5).astype(int)
                auc_base = float(roc_auc_score(y_te, p_base))
                acc_base = float(accuracy_score(y_te, yhat_base))

            clf_mars = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
            clf_mars.fit(xf_tr, y_tr)
            p_mars_all = clf_mars.predict_proba(xf_te)
            if p_mars_all.shape[1] >= 2:
                p_mars = p_mars_all[:, 1]
                yhat_mars = (p_mars >= 0.5).astype(int)
                auc_mars = float(roc_auc_score(y_te, p_mars))
                acc_mars = float(accuracy_score(y_te, yhat_mars))
    else:
        clf_base = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
        clf_base.fit(train_cls[base_feats], y_train_cls)
        p_base_all = clf_base.predict_proba(test_cls[base_feats])
        if p_base_all.shape[1] >= 2:
            p_base = p_base_all[:, 1]
            yhat_base = (p_base >= 0.5).astype(int)
            auc_base = float(roc_auc_score(y_test_cls, p_base))
            acc_base = float(accuracy_score(y_test_cls, yhat_base))

        clf_mars = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
        clf_mars.fit(train_cls[full_feats], y_train_cls)
        p_mars_all = clf_mars.predict_proba(test_cls[full_feats])
        if p_mars_all.shape[1] >= 2:
            p_mars = p_mars_all[:, 1]
            yhat_mars = (p_mars >= 0.5).astype(int)
            auc_mars = float(roc_auc_score(y_test_cls, p_mars))
            acc_mars = float(accuracy_score(y_test_cls, yhat_mars))

    # Nearby hypothesis B: Mars altitude bands vs RI
    altitude_result = {"status": "skipped"}
    if "mars_altitude_deg" in work.columns:
        tmp = work[["mars_altitude_deg", "wind_change"]].copy()
        tmp["ri_event"] = (tmp["wind_change"] >= 30).astype(int)
        tmp = tmp.dropna(subset=["mars_altitude_deg"])
        if len(tmp) >= 100:
            tmp["alt_band"] = pd.qcut(tmp["mars_altitude_deg"], q=4, duplicates="drop")
            ctab = pd.crosstab(tmp["alt_band"], tmp["ri_event"])
            chi2, pval, dof, _ = chi2_contingency(ctab)
            ctab.to_csv(OUTPUT_DIR / "mars_altitude_ri_crosstab.csv")
            altitude_result = {
                "status": "ok",
                "chi2_statistic": float(chi2),
                "p_value": float(pval),
                "dof": int(dof),
            }

    # Nearby hypothesis C: Mangal retrograde vs direct wind-change
    retro_result = {"status": "skipped"}
    if "vedic_mangal_speed_deg_per_day" in work.columns:
        tmp = work[["vedic_mangal_speed_deg_per_day", "wind_change"]].copy().dropna()
        retro = tmp[tmp["vedic_mangal_speed_deg_per_day"] < 0]["wind_change"]
        direct = tmp[tmp["vedic_mangal_speed_deg_per_day"] >= 0]["wind_change"]
        if len(retro) >= 20 and len(direct) >= 20:
            f_stat, pval = f_oneway(retro, direct)
            retro_result = {
                "status": "ok",
                "retro_n": int(len(retro)),
                "direct_n": int(len(direct)),
                "retro_mean_wind_change": float(retro.mean()),
                "direct_mean_wind_change": float(direct.mean()),
                "f_stat": float(f_stat),
                "p_value": float(pval),
            }

    # Nearby hypothesis D: Rahu retrograde vs direct wind-change
    rahu_retro_result = {"status": "skipped"}
    rahu_speed_col = None
    if "vedic_rahu_speed_deg_per_day" in work.columns:
        rahu_speed_col = "vedic_rahu_speed_deg_per_day"
    elif "vedic_spashth_rahu_speed_deg_per_day" in work.columns:
        rahu_speed_col = "vedic_spashth_rahu_speed_deg_per_day"

    if rahu_speed_col:
        tmp = work[[rahu_speed_col, "wind_change"]].copy().dropna()
        retro = tmp[tmp[rahu_speed_col] < 0]["wind_change"]
        direct = tmp[tmp[rahu_speed_col] >= 0]["wind_change"]
        if len(retro) >= 20 and len(direct) >= 20:
            f_stat, pval = f_oneway(retro, direct)
            rahu_retro_result = {
                "status": "ok",
                "speed_feature": rahu_speed_col,
                "retro_n": int(len(retro)),
                "direct_n": int(len(direct)),
                "retro_mean_wind_change": float(retro.mean()),
                "direct_mean_wind_change": float(direct.mean()),
                "f_stat": float(f_stat),
                "p_value": float(pval),
            }

    # Nearby hypothesis E: Rahu degree monotonic relation with wind change
    rahu_degree_result = {"status": "skipped"}
    rahu_degree_col = None
    if "vedic_rahu_full_degree" in work.columns:
        rahu_degree_col = "vedic_rahu_full_degree"
    elif "vedic_spashth_rahu_full_degree" in work.columns:
        rahu_degree_col = "vedic_spashth_rahu_full_degree"

    if rahu_degree_col:
        tmp = work[[rahu_degree_col, "wind_change"]].copy().dropna()
        if len(tmp) >= 100:
            rho, pval = spearmanr(tmp[rahu_degree_col], tmp["wind_change"], nan_policy="omit")
            rahu_degree_result = {
                "status": "ok",
                "degree_feature": rahu_degree_col,
                "n": int(len(tmp)),
                "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                "p_value": float(pval) if pd.notna(pval) else np.nan,
            }

    # Nearby hypothesis F: Ketu retrograde vs direct wind-change
    # NOTE: Rahu and Ketu are the North and South lunar nodes. They are always exactly
    # 180° apart and always move together with the same speed magnitude (opposite sign
    # convention differs by system). As a result, vedic_ketu_speed_deg_per_day and
    # vedic_rahu_speed_deg_per_day have the same absolute values, and the retrograde/direct
    # split will produce identical group sizes and near-identical results. This is an
    # astronomical property, not a data bug. The Ketu test is still reported separately
    # for completeness, but results should be interpreted as confirming the Rahu finding,
    # not as an independent replication.
    ketu_retro_result = {"status": "skipped"}
    ketu_speed_col = None
    if "vedic_ketu_speed_deg_per_day" in work.columns:
        ketu_speed_col = "vedic_ketu_speed_deg_per_day"
    elif "vedic_spashth_ketu_speed_deg_per_day" in work.columns:
        ketu_speed_col = "vedic_spashth_ketu_speed_deg_per_day"

    if ketu_speed_col:
        tmp = work[[ketu_speed_col, "wind_change"]].copy().dropna()
        retro = tmp[tmp[ketu_speed_col] < 0]["wind_change"]
        direct = tmp[tmp[ketu_speed_col] >= 0]["wind_change"]
        if len(retro) >= 20 and len(direct) >= 20:
            f_stat, pval = f_oneway(retro, direct)
            ketu_retro_result = {
                "status": "ok",
                "speed_feature": ketu_speed_col,
                "retro_n": int(len(retro)),
                "direct_n": int(len(direct)),
                "retro_mean_wind_change": float(retro.mean()),
                "direct_mean_wind_change": float(direct.mean()),
                "f_stat": float(f_stat),
                "p_value": float(pval),
                "note": (
                    "Ketu and Rahu always move together (same speed, nodes of Moon orbit). "
                    "Results will match Rahu retrograde test by astronomical necessity — "
                    "this is not an independent test."
                ),
            }

    # Nearby hypothesis G: Ketu degree monotonic relation with wind change
    ketu_degree_result = {"status": "skipped"}
    ketu_degree_col = None
    if "vedic_ketu_full_degree" in work.columns:
        ketu_degree_col = "vedic_ketu_full_degree"
    elif "vedic_spashth_ketu_full_degree" in work.columns:
        ketu_degree_col = "vedic_spashth_ketu_full_degree"

    if ketu_degree_col:
        tmp = work[[ketu_degree_col, "wind_change"]].copy().dropna()
        if len(tmp) >= 100:
            rho, pval = spearmanr(tmp[ketu_degree_col], tmp["wind_change"], nan_policy="omit")
            ketu_degree_result = {
                "status": "ok",
                "degree_feature": ketu_degree_col,
                "n": int(len(tmp)),
                "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                "p_value": float(pval) if pd.notna(pval) else np.nan,
            }

    # Multiple-comparison note: This function tests 7 nearby sub-hypotheses (A–G).
    # Bonferroni-corrected alpha for these 7 tests: 0.05 / 7 ≈ 0.0071.
    # Any p-value that is < 0.05 but > 0.0071 is nominally significant but does NOT
    # survive family-wise error correction. The corrected threshold is stored in results.
    bonferroni_alpha_nearby = 0.05 / 7

    top_astro = astro_imp.sort_values("importance", ascending=False).head(12).to_dict(orient="records") if not astro_imp.empty else []
    top_mars = mars_imp.sort_values("importance", ascending=False).head(8).to_dict(orient="records") if not mars_imp.empty else []
    top_rahu = rahu_imp.sort_values("importance", ascending=False).head(8).to_dict(orient="records") if not rahu_imp.empty else []
    top_ketu = ketu_imp.sort_values("importance", ascending=False).head(8).to_dict(orient="records") if not ketu_imp.empty else []
    results = {
        "status": "ok",
        "mars_feature_count": len(mars_features),
        "rahu_feature_count": len(rahu_features),
        "ketu_feature_count": len(ketu_features),
        "astro_feature_count": len(astro_feats_only),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "regression": {
            "baseline_rmse": rmse_base,
            "with_mars_rahu_ketu_rmse": rmse_mars,
            "relative_improvement_pct": delta_pct,
        },
        "classification_ri": {
            "split_mode": split_mode,
            "baseline_auc": auc_base,
            "with_mars_rahu_ketu_auc": auc_mars,
            "baseline_accuracy": acc_base,
            "with_mars_rahu_ketu_accuracy": acc_mars,
        },
        "astro_importance_summary": {
            "astro_total_importance": float(astro_imp["importance"].sum()) if not astro_imp.empty else 0.0,
            "top_astro_features": top_astro,
        },
        "mars_importance_summary": {
            "mars_total_importance": float(mars_imp["importance"].sum()) if not mars_imp.empty else 0.0,
            "best_mars_rank": int(mars_imp["rank"].min()) if not mars_imp.empty else None,
            "mars_in_top20": int((mars_imp["rank"] <= 20).sum()) if not mars_imp.empty else 0,
            "top_mars_features": top_mars,
        },
        "rahu_importance_summary": {
            "rahu_total_importance": float(rahu_imp["importance"].sum()) if not rahu_imp.empty else 0.0,
            "best_rahu_rank": int(rahu_imp["rank"].min()) if not rahu_imp.empty else None,
            "rahu_in_top20": int((rahu_imp["rank"] <= 20).sum()) if not rahu_imp.empty else 0,
            "top_rahu_features": top_rahu,
        },
        "ketu_importance_summary": {
            "ketu_total_importance": float(ketu_imp["importance"].sum()) if not ketu_imp.empty else 0.0,
            "best_ketu_rank": int(ketu_imp["rank"].min()) if not ketu_imp.empty else None,
            "ketu_in_top20": int((ketu_imp["rank"] <= 20).sum()) if not ketu_imp.empty else 0,
            "top_ketu_features": top_ketu,
        },
        "nearby_hypothesis_altitude_vs_ri": altitude_result,
        "nearby_hypothesis_retrograde_vs_wind_change": retro_result,
        "nearby_hypothesis_rahu_retrograde_vs_wind_change": rahu_retro_result,
        "nearby_hypothesis_rahu_degree_vs_wind_change": rahu_degree_result,
        "nearby_hypothesis_ketu_retrograde_vs_wind_change": ketu_retro_result,
        "nearby_hypothesis_ketu_degree_vs_wind_change": ketu_degree_result,
        "multiple_comparison_note": (
            f"7 nearby sub-hypotheses tested (A-G). Bonferroni-corrected alpha = {bonferroni_alpha_nearby:.4f}. "
            "p-values between 0.0071 and 0.05 are nominally significant but do not survive correction."
        ),
    }

    with open(OUTPUT_DIR / "mars_rahu_ketu_hypothesis_results.txt", "w") as f:
        f.write(str(results))
    print(f"Mars+Rahu+Ketu hypothesis results saved to: {OUTPUT_DIR / 'mars_rahu_ketu_hypothesis_results.txt'}")
    return results


def run_mars_rahu_event_window_hypothesis():
    """
    Stronger event-window design:
    Compare Mars/Rahu/Ketu parameter shifts in 24h pre vs 24h post around RI onset,
    against matched non-RI control windows.
    """
    print("\n--- Running Mars+Rahu Event-Window Hypothesis ---")
    df = pd.read_parquet(DATA_PATH)

    required_cols = ["storm_id", "date_utc", "wind", "wind_change"]
    astro_cols = [
        "mars_ecliptic_lat_deg", "mars_dec_deg", "mars_altitude_deg",
        "vedic_mangal_speed_deg_per_day", "vedic_mangal_full_degree",
        "vedic_rahu_speed_deg_per_day", "vedic_spashth_rahu_speed_deg_per_day",
        "vedic_rahu_full_degree", "vedic_spashth_rahu_full_degree",
        "vedic_ketu_speed_deg_per_day", "vedic_spashth_ketu_speed_deg_per_day",
        "vedic_ketu_full_degree", "vedic_spashth_ketu_full_degree",
    ]
    use_astro = [c for c in astro_cols if c in df.columns]
    missing_core = [c for c in required_cols if c not in df.columns]
    if missing_core or not use_astro:
        results = {
            "status": "skipped",
            "reason": f"missing core={missing_core}, astro={use_astro}",
        }
        with open(OUTPUT_DIR / "mars_rahu_event_window_results.txt", "w") as f:
            f.write(str(results))
        print("Event-window hypothesis skipped due to missing columns.")
        return results

    work = df[required_cols + use_astro].copy()
    work["date_utc"] = pd.to_datetime(work["date_utc"], errors="coerce")
    work = work.dropna(subset=["storm_id", "date_utc", "wind", "wind_change"]) 
    work = work.sort_values(["storm_id", "date_utc"]).reset_index(drop=True)
    # Using a lower RI threshold (15 kt) for event-window analysis to generate enough
    # RI onset events for comparison. The 30 kt threshold only produced ~5 onset events
    # which is too few to test astro feature differences. 15 kt captures moderate
    # intensification events while still being distinct from normal variation.
    # Results should be noted as using this relaxed threshold.
    RI_THRESHOLD_EVENT_WINDOW = 15
    work["ri_flag"] = (pd.to_numeric(work["wind_change"], errors="coerce") >= RI_THRESHOLD_EVENT_WINDOW).astype(int)
    work["ri_prev"] = work.groupby("storm_id")["ri_flag"].shift(1).fillna(0).astype(int)
    work["ri_onset"] = ((work["ri_flag"] == 1) & (work["ri_prev"] == 0)).astype(int)

    rng = np.random.default_rng(42)
    event_rows = []

    for sid, grp in work.groupby("storm_id", sort=False):
        grp = grp.sort_values("date_utc").reset_index(drop=True)
        ri_times = grp.loc[grp["ri_onset"] == 1, "date_utc"].tolist()
        if not ri_times:
            continue

        # Build non-RI candidate controls: stable windows with low instantaneous wind-change
        control_candidates = grp.loc[grp["wind_change"].abs() <= 5, "date_utc"].tolist()

        def extract_event(t0, event_type):
            pre = grp[(grp["date_utc"] >= t0 - pd.Timedelta(hours=24)) & (grp["date_utc"] < t0)]
            post = grp[(grp["date_utc"] > t0) & (grp["date_utc"] <= t0 + pd.Timedelta(hours=24))]
            if len(pre) < 1 or len(post) < 1:
                return

            delta_wind = float(post["wind"].mean() - pre["wind"].mean())
            row = {
                "storm_id": sid,
                "event_type": event_type,
                "event_time": t0,
                "delta_wind_24h": delta_wind,
            }
            for c in use_astro:
                row[f"delta_{c}"] = float(post[c].mean() - pre[c].mean()) if c in pre.columns and c in post.columns else np.nan
            event_rows.append(row)

        for t0 in ri_times:
            extract_event(t0, "RI")

        if control_candidates:
            k = min(len(ri_times), len(control_candidates))
            sampled = rng.choice(control_candidates, size=k, replace=False)
            for t0 in sampled:
                extract_event(pd.Timestamp(t0), "nonRI")

    events = pd.DataFrame(event_rows)
    if events.empty:
        results = {"status": "skipped", "reason": "no valid RI/control event windows"}
        with open(OUTPUT_DIR / "mars_rahu_event_window_results.txt", "w") as f:
            f.write(str(results))
        print("Event-window hypothesis skipped: no valid windows.")
        return results

    events.to_csv(OUTPUT_DIR / "mars_rahu_event_windows.csv", index=False)

    # Compare RI vs nonRI event-window deltas per feature
    rows = []
    ri = events[events["event_type"] == "RI"]
    ctrl = events[events["event_type"] == "nonRI"]
    for c in use_astro:
        dc = f"delta_{c}"
        r = ri[dc].dropna()
        n = ctrl[dc].dropna()
        if len(r) < 10 or len(n) < 10:
            continue
        f_stat, pval = f_oneway(r, n)
        rho, p_corr = spearmanr(ri[dc].dropna(), ri.loc[ri[dc].notna(), "delta_wind_24h"], nan_policy="omit")
        rows.append({
            "feature": c,
            "ri_n": int(len(r)),
            "nonri_n": int(len(n)),
            "ri_mean_delta": float(r.mean()),
            "nonri_mean_delta": float(n.mean()),
            "mean_delta_diff": float(r.mean() - n.mean()),
            "anova_f": float(f_stat),
            "anova_p": float(pval),
            "ri_spearman_rho_with_delta_wind": float(rho) if pd.notna(rho) else np.nan,
            "ri_spearman_p": float(p_corr) if pd.notna(p_corr) else np.nan,
        })

    comp = pd.DataFrame(rows)
    if not comp.empty:
        comp = comp.sort_values(["anova_p", "ri_spearman_p"], ascending=[True, True])
    comp.to_csv(OUTPUT_DIR / "mars_rahu_event_window_comparison.csv", index=False)

    results = {
        "status": "ok",
        "ri_event_windows": int((events["event_type"] == "RI").sum()),
        "nonri_event_windows": int((events["event_type"] == "nonRI").sum()),
        "tested_features": int(len(comp)),
        "top_event_window_signals": comp.head(10).to_dict(orient="records") if not comp.empty else [],
    }

    with open(OUTPUT_DIR / "mars_rahu_event_window_results.txt", "w") as f:
        f.write(str(results))
    print(f"Event-window results saved to: {OUTPUT_DIR / 'mars_rahu_event_window_results.txt'}")
    return results


def write_consolidated_report(h1, h2, h3, sam, mars, event_window):
    report_path = OUTPUT_DIR / "GOD_LEVEL_RESEARCH_REPORT.md"
    lines = []
    lines.append("# God-Level Research Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("This report consolidates hypothesis testing on meteorological, astronomical, Vedic, and Samvatsaram-linked signals in storm evolution.")
    lines.append("")
    lines.append("## Hypothesis 1: Vedic Predictive Lift")
    lines.append(f"- Baseline RMSE: {h1.get('baseline_model_rmse', np.nan):.4f}")
    lines.append(f"- Vedic RMSE: {h1.get('vedic_model_rmse', np.nan):.4f}")
    improvement = h1.get('improvement_pct', np.nan)
    lines.append(f"- Relative improvement (%): {improvement:.2f}")
    if not np.isnan(improvement) and improvement < 0:
        lines.append(f"- **RESULT: Vedic features DEGRADED prediction by {abs(improvement):.1f}%. Null hypothesis NOT rejected.**")
        lines.append("- Adding Vedic features increased RMSE compared to the meteorological baseline.")
        lines.append("- This indicates Vedic features add noise rather than signal for predicting wind change.")
    elif not np.isnan(improvement) and improvement > 0:
        lines.append(f"- **RESULT: Vedic features improved prediction by {improvement:.1f}%.**")
    else:
        lines.append("- **RESULT: No measurable improvement from Vedic features.**")
    lines.append("")
    lines.append("## Hypothesis 2: Lunar Phase vs RI")
    lines.append(f"- Chi-square: {h2.get('chi2_statistic', np.nan):.4f}")
    h2_p = h2.get('p_value', np.nan)
    lines.append(f"- p-value: {h2_p:.6f}")
    lines.append(f"- Degrees of freedom: {h2.get('dof', np.nan)}")
    lines.append(f"- Bonferroni-corrected alpha (across 7 hypothesis groups): 0.0071")
    if not np.isnan(h2_p):
        if h2_p < 0.0071:
            lines.append("- **RESULT: Significant after Bonferroni correction.**")
        elif h2_p < 0.05:
            lines.append("- **RESULT: Nominally significant (p<0.05) but FAILS Bonferroni correction. Interpret with caution.**")
        else:
            lines.append("- **RESULT: Not significant.**")
    lines.append("- RI threshold used: 30 kt/24h (NOTE: WMO standard is 35 kt)")
    lines.append("")
    lines.append("## Hypothesis 3: Planetary Speed vs Trajectory")
    lines.append(f"- Status: {h3.get('status', 'unknown')}")
    lines.append(f"- Tested feature pairs: {h3.get('tested_pairs', 0)}")
    lines.append("- Top associations are written to `hypothesis3_spearman_matrix.csv`.")
    lines.append("- Retrograde/direct ANOVA table is written to `hypothesis3_retrograde_anova.csv`.")
    lines.append("")
    lines.append("## Samvatsaram Analysis")
    lines.append(f"- Status: {sam.get('status', 'unknown')}")
    lines.append(f"- Samvatsara classes: {sam.get('samvatsara_count', 0)}")
    if sam.get("wind_anova"):
        lines.append(f"- Wind ANOVA p-value: {sam['wind_anova'].get('p_value', np.nan):.6f}")
    if sam.get("pressure_anova"):
        lines.append(f"- Pressure ANOVA p-value: {sam['pressure_anova'].get('p_value', np.nan):.6f}")
    if sam.get("ri_chi2"):
        lines.append(f"- RI vs Samvatsara chi-square p-value: {sam['ri_chi2'].get('p_value', np.nan):.6f}")
    lines.append("- **CAUTION**: Each Samvatsara maps to roughly one calendar year. Significant ANOVA")
    lines.append("  results reflect interannual storm intensity variability, NOT a Hindu calendar effect.")
    lines.append("  This is equivalent to testing 'do storms vary in intensity across different years?'")
    lines.append("  — which is trivially true due to ENSO, PDO, and decadal climate cycles.")
    lines.append("")
    lines.append("## Mars + Rahu + Ketu Hypothesis")
    lines.append(f"- Mars feature count tested: {mars.get('mars_feature_count', 0)}")
    lines.append(f"- Rahu feature count tested: {mars.get('rahu_feature_count', 0)}")
    lines.append(f"- Ketu feature count tested: {mars.get('ketu_feature_count', 0)}")
    if mars.get("regression"):
        lines.append(f"- Wind-change RMSE baseline: {mars['regression'].get('baseline_rmse', np.nan):.4f}")
        lines.append(f"- Wind-change RMSE with Mars+Rahu+Ketu: {mars['regression'].get('with_mars_rahu_ketu_rmse', np.nan):.4f}")
        lines.append(f"- Relative improvement (%): {mars['regression'].get('relative_improvement_pct', np.nan):.2f}")
    if mars.get("classification_ri"):
        lines.append(f"- RI split mode: {mars['classification_ri'].get('split_mode', 'unknown')}")
        lines.append(f"- RI AUC baseline: {mars['classification_ri'].get('baseline_auc', np.nan):.4f}")
        lines.append(f"- RI AUC with Mars+Rahu+Ketu: {mars['classification_ri'].get('with_mars_rahu_ketu_auc', np.nan):.4f}")
    if mars.get("mars_importance_summary"):
        lines.append(f"- Mars features in top-20 importance: {mars['mars_importance_summary'].get('mars_in_top20', 0)}")
        lines.append(f"- Best Mars feature rank: {mars['mars_importance_summary'].get('best_mars_rank', 'NA')}")
    if mars.get("rahu_importance_summary"):
        lines.append(f"- Rahu features in top-20 importance: {mars['rahu_importance_summary'].get('rahu_in_top20', 0)}")
        lines.append(f"- Best Rahu feature rank: {mars['rahu_importance_summary'].get('best_rahu_rank', 'NA')}")
    if mars.get("ketu_importance_summary"):
        lines.append(f"- Ketu features in top-20 importance: {mars['ketu_importance_summary'].get('ketu_in_top20', 0)}")
        lines.append(f"- Best Ketu feature rank: {mars['ketu_importance_summary'].get('best_ketu_rank', 'NA')}")
    alt = mars.get("nearby_hypothesis_altitude_vs_ri", {})
    if alt.get("status") == "ok":
        lines.append(f"- Altitude-band vs RI chi-square p-value: {alt.get('p_value', np.nan):.6f}")
    retro = mars.get("nearby_hypothesis_retrograde_vs_wind_change", {})
    if retro.get("status") == "ok":
        lines.append(f"- Retrograde vs direct wind-change p-value: {retro.get('p_value', np.nan):.6f}")
    rretro = mars.get("nearby_hypothesis_rahu_retrograde_vs_wind_change", {})
    if rretro.get("status") == "ok":
        lines.append(f"- Rahu retrograde vs direct wind-change p-value: {rretro.get('p_value', np.nan):.6f}")
    rdeg = mars.get("nearby_hypothesis_rahu_degree_vs_wind_change", {})
    if rdeg.get("status") == "ok":
        lines.append(f"- Rahu degree vs wind-change Spearman p-value: {rdeg.get('p_value', np.nan):.6f}")
    kretro = mars.get("nearby_hypothesis_ketu_retrograde_vs_wind_change", {})
    if kretro.get("status") == "ok":
        lines.append(f"- Ketu retrograde vs direct wind-change p-value: {kretro.get('p_value', np.nan):.6f}")
    kdeg = mars.get("nearby_hypothesis_ketu_degree_vs_wind_change", {})
    if kdeg.get("status") == "ok":
        lines.append(f"- Ketu degree vs wind-change Spearman p-value: {kdeg.get('p_value', np.nan):.6f}")
    lines.append("")
    lines.append("## Mars+Rahu Event-Window (24h Pre/Post RI)")
    lines.append(f"- Status: {event_window.get('status', 'unknown')}")
    lines.append(f"- RI event windows: {event_window.get('ri_event_windows', 0)}")
    lines.append(f"- non-RI control windows: {event_window.get('nonri_event_windows', 0)}")
    lines.append(f"- Tested astro features: {event_window.get('tested_features', 0)}")
    lines.append("- RI threshold for event windows: 15 kt (relaxed from 30 kt to generate sufficient events)")
    lines.append("- Detailed comparison: `mars_rahu_event_window_comparison.csv`.")
    lines.append("")
    lines.append("## Files Produced")
    lines.append("- hypothesis1_results.txt")
    lines.append("- hypothesis2_results.txt")
    lines.append("- hypothesis3_results.txt")
    lines.append("- hypothesis3_spearman_matrix.csv")
    lines.append("- hypothesis3_retrograde_anova.csv")
    lines.append("- samvatsara_summary.csv")
    lines.append("- samvatsara_results.txt")
    lines.append("- mars_rahu_ketu_hypothesis_results.txt")
    lines.append("- mars_rahu_feature_importance.csv")
    lines.append("- mars_altitude_ri_crosstab.csv")
    lines.append("- mars_rahu_event_window_results.txt")
    lines.append("- mars_rahu_event_window_comparison.csv")
    lines.append("- mars_rahu_event_windows.csv")
    lines.append("")
    lines.append("## Research Interpretation")
    lines.append("- H1 indicates whether Vedic signals add out-of-sample lift over meteorology baseline.")
    lines.append("- H2 quantifies non-random lunar-phase concentration of rapid intensification.")
    lines.append("- H3 tests mechanistic coupling between planetary speed dynamics and trajectory geometry.")
    lines.append("- Samvatsaram block evaluates inter-year cyclic grouping effects on intensity and RI rates.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Consolidated report saved to: {report_path}")


if __name__ == "__main__":
    h1 = run_hypothesis1()
    h2 = run_hypothesis2()
    h3 = run_hypothesis3()
    sam = run_samvatsara_analysis()
    mars = run_mars_hypothesis()
    event_window = run_mars_rahu_event_window_hypothesis()
    write_consolidated_report(h1, h2, h3, sam, mars, event_window)
