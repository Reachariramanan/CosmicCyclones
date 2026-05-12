"""
05_comprehensive_eda.py
=======================
Comprehensive EDA covering all remaining analyses:
- Spatial analysis & geographic patterns
- Vedic cross-tabulation (Tithi × Nakshatra × Intensity)
- Anomaly detection in feature space
- Variance Inflation Factor (multicollinearity)
- Statistical tests (normality, stationarity)
- Feature selection (Boruta-style, recursive)
- Lag analysis & autocorrelation
- Storm lifecycle phase analysis
- Planetary speed / retrograde analysis
- Day/Night cycle analysis
- Hemisphere & basin comparison
All outputs saved to EDA/output/comprehensive/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy import stats
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf

OUT = Path(__file__).resolve().parent / "output" / "comprehensive"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(__file__).resolve().parent / "output" / "storm_data_all.parquet"

sns.set_theme(style="whitegrid", font_scale=1.0)


def load():
    df = pd.read_parquet(DATA)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────
# 1. Storm Lifecycle Phase Analysis
# ─────────────────────────────────────────────────────────────
def lifecycle_analysis(df):
    print("\n[1] Storm Lifecycle Phase Analysis")
    # LIMITATION: Lifecycle phases are approximated using normalized track_num (0→1 scale).
    # This assumes storms follow a monotonic single-peak intensification curve: Genesis →
    # Intensification → Mature → Dissipation. In reality, tropical cyclones can re-intensify
    # after weakening, stall, or undergo rapid short-term fluctuations. For multi-peak storms,
    # track_num-based phases may label re-intensification as "Mature" or "Dissipation" instead
    # of a second "Intensification" phase. A more accurate approach would use elapsed time
    # since genesis or a wind-threshold-based phase classification.

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, sid in enumerate(df["storm_id"].unique()):
        if i >= 8:
            break
        sd = df[df["storm_id"] == sid].sort_values("track_num").copy()
        ax = axes[i]

        # Normalize track number to 0-1 (lifecycle progress)
        sd["lifecycle_pct"] = (sd["track_num"] - sd["track_num"].min()) / max(
            sd["track_num"].max() - sd["track_num"].min(), 1)

        # Classify phase: genesis (<20%), intensification (20-50%), mature (50-70%), dissipation (>70%)
        # NOTE: This binning is approximate; see function docstring above.
        sd["phase"] = pd.cut(sd["lifecycle_pct"], bins=[0, 0.2, 0.5, 0.7, 1.0],
                            labels=["Genesis", "Intensify", "Mature", "Dissipate"],
                            include_lowest=True)

        colors_map = {"Genesis": "green", "Intensify": "orange", "Mature": "red", "Dissipate": "blue"}
        for phase in ["Genesis", "Intensify", "Mature", "Dissipate"]:
            mask = sd["phase"] == phase
            ax.scatter(sd.loc[mask, "lifecycle_pct"], sd.loc[mask, "wind"],
                      c=colors_map.get(phase, "gray"), label=phase, s=20, alpha=0.7)

        ax.plot(sd["lifecycle_pct"], sd["wind"], "-", color="gray", alpha=0.3)
        ax.set_title(sd["storm_name"].iloc[0], fontsize=10)
        ax.set_xlabel("Lifecycle %")
        ax.set_ylabel("Wind (kt)")
        if i == 0:
            ax.legend(fontsize=6)

    fig.suptitle("Storm Lifecycle Phases", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "lifecycle_phases.png", dpi=150)
    plt.close(fig)

    # Aggregate: mean wind by lifecycle phase across all storms
    df_temp = df.copy()
    for sid in df["storm_id"].unique():
        mask = df_temp["storm_id"] == sid
        sd = df_temp.loc[mask].sort_values("track_num")
        lc = (sd["track_num"] - sd["track_num"].min()) / max(sd["track_num"].max() - sd["track_num"].min(), 1)
        df_temp.loc[mask, "lifecycle_pct"] = lc.values

    df_temp["phase"] = pd.cut(df_temp["lifecycle_pct"], bins=[0, 0.2, 0.5, 0.7, 1.0],
                             labels=["Genesis", "Intensify", "Mature", "Dissipate"],
                             include_lowest=True)

    phase_stats = df_temp.groupby("phase", observed=True)["wind"].agg(
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
        count="count"
    )
    phase_stats.to_csv(OUT / "lifecycle_phase_stats.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    yerr_low = phase_stats["median"] - phase_stats["q1"]
    yerr_high = phase_stats["q3"] - phase_stats["median"]
    phase_stats["median"].plot.bar(ax=ax, yerr=[yerr_low, yerr_high], capsize=5,
                                   color=["green", "orange", "red", "blue"])
    ax.set_title("Median Wind by Lifecycle Phase (IQR)")
    ax.set_ylabel("Wind (kt)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(OUT / "lifecycle_median_wind.png", dpi=150)
    plt.close(fig)

    print(f"    Saved lifecycle_phases.png, lifecycle_phase_stats.csv, lifecycle_median_wind.png")


# ─────────────────────────────────────────────────────────────
# 2. Vedic Cross-Tabulation Analysis
# ─────────────────────────────────────────────────────────────
def vedic_crosstab(df):
    print("\n[2] Vedic Cross-Tabulation (Tithi × Nakshatra × Intensity)")

    # Clean categories
    for col in ["tithi", "nakshatra", "moonsign", "paksha"]:
        if col in df.columns:
            df[col + "_c"] = df[col].astype(str).apply(lambda x: x.split(" upto")[0].strip())

    # Tithi × Nakshatra pivot (mean wind)
    if "tithi_c" in df.columns and "nakshatra_c" in df.columns:
        pivot = df.pivot_table(values="wind", index="tithi_c", columns="nakshatra_c",
                              aggfunc="mean")
        pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")

        if pivot.shape[0] > 1 and pivot.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(16, 10))
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                       linewidths=0.5)
            ax.set_title("Mean Wind Speed: Tithi × Nakshatra")
            fig.tight_layout()
            fig.savefig(OUT / "tithi_nakshatra_heatmap.png", dpi=150)
            plt.close(fig)

        # Median heatmap
        pivot_med = df.pivot_table(values="wind", index="tithi_c", columns="nakshatra_c",
                                   aggfunc="median")
        pivot_med = pivot_med.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if pivot_med.shape[0] > 1 and pivot_med.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(16, 10))
            sns.heatmap(pivot_med, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                       linewidths=0.5)
            ax.set_title("Median Wind Speed: Tithi × Nakshatra")
            fig.tight_layout()
            fig.savefig(OUT / "tithi_nakshatra_heatmap_median.png", dpi=150)
            plt.close(fig)

    # Paksha × Moonsign
    if "paksha_c" in df.columns and "moonsign_c" in df.columns:
        pivot2 = df.pivot_table(values="wind", index="paksha_c", columns="moonsign_c",
                               aggfunc=["mean", "count"])
        pivot2.to_csv(OUT / "paksha_moonsign_pivot.csv")

    # Nakshatra frequency in high- vs low-intensity
    median_wind = df["wind"].median()
    df["intensity_class"] = np.where(df["wind"] >= median_wind, "High", "Low")

    if "nakshatra_c" in df.columns:
        ct = pd.crosstab(df["nakshatra_c"], df["intensity_class"])
        ct.to_csv(OUT / "nakshatra_intensity_crosstab.csv")

        # Chi-square test
        chi2, p, dof, expected = stats.chi2_contingency(ct)

        fig, ax = plt.subplots(figsize=(12, 8))
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_pct.plot.barh(ax=ax, stacked=True, color=["steelblue", "coral"])
        ax.set_title(f"Nakshatra vs Intensity (Chi²={chi2:.2f}, p={p:.4f})")
        ax.set_xlabel("Percentage (%)")
        ax.legend(title="Intensity")
        fig.tight_layout()
        fig.savefig(OUT / "nakshatra_intensity_bar.png", dpi=150)
        plt.close(fig)

        print(f"    Nakshatra × Intensity Chi²={chi2:.2f}, p={p:.4f}")

    print(f"    Saved cross-tabulation outputs")


# ─────────────────────────────────────────────────────────────
# 3. Anomaly Detection
# ─────────────────────────────────────────────────────────────
def anomaly_detection(df):
    print("\n[3] Anomaly Detection (Isolation Forest)")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if df[c].notna().sum() > len(df) * 0.9]
    exclude = ["track_num", "storm_max_wind", "storm_forecast"]
    feature_cols = [c for c in feature_cols if c not in exclude]

    df_clean = df[feature_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    labels = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)

    df.loc[df_clean.index, "anomaly_label"] = labels
    df.loc[df_clean.index, "anomaly_score"] = scores

    n_anomalies = (labels == -1).sum()
    print(f"    Detected {n_anomalies} anomalies out of {len(labels)} observations")

    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    normal = labels == 1
    ax.scatter(X_pca[normal, 0], X_pca[normal, 1], c="steelblue", s=10, alpha=0.5, label="Normal")
    ax.scatter(X_pca[~normal, 0], X_pca[~normal, 1], c="red", s=30, alpha=0.8, label="Anomaly",
              edgecolors="black", linewidth=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Anomalies in PCA Space")
    ax.legend()

    ax = axes[1]
    ax.hist(scores, bins=40, color="steelblue", edgecolor="black")
    ax.axvline(np.percentile(scores, 5), color="red", linestyle="--", label="5% threshold")
    ax.set_title("Anomaly Score Distribution")
    ax.set_xlabel("Isolation Forest Score")
    ax.legend()

    fig.suptitle("Anomaly Detection", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "anomaly_detection.png", dpi=150)
    plt.close(fig)

    # What storms have most anomalies?
    anomaly_df = df.loc[df_clean.index][labels == -1]
    anomaly_storms = anomaly_df["storm_name"].value_counts()
    anomaly_storms.to_csv(OUT / "anomaly_by_storm.csv")

    print(f"    Saved anomaly_detection.png, anomaly_by_storm.csv")


# ─────────────────────────────────────────────────────────────
# 4. Autocorrelation & Lag Analysis
# ─────────────────────────────────────────────────────────────
def autocorrelation_analysis(df):
    print("\n[4] Autocorrelation & Lag Analysis")

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, sid in enumerate(df["storm_id"].unique()):
        if i >= 8:
            break
        sd = df[df["storm_id"] == sid].sort_values("track_num")
        ax = axes[i]

        wind_series = sd["wind"].dropna().values
        if len(wind_series) >= 5:
            nlags = min(20, len(wind_series) - 1)
            acf_vals = acf(wind_series, nlags=nlags, fft=True)
            ax.bar(range(len(acf_vals)), acf_vals, color="steelblue", width=0.5)
            ax.axhline(1.96 / np.sqrt(len(wind_series)), color="red", linestyle="--", alpha=0.5)
            ax.axhline(-1.96 / np.sqrt(len(wind_series)), color="red", linestyle="--", alpha=0.5)
        ax.set_title(sd["storm_name"].iloc[0], fontsize=10)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")

    fig.suptitle("Wind Speed Autocorrelation by Storm", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "autocorrelation_wind.png", dpi=150)
    plt.close(fig)

    # Cross-correlation: wind vs sun altitude (lagged)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, sid in enumerate(df["storm_id"].unique()):
        if i >= 8:
            break
        sd = df[df["storm_id"] == sid].sort_values("track_num")
        ax = axes[i]

        if "sun_altitude_deg" in sd.columns:
            w = sd["wind"].values
            s = sd["sun_altitude_deg"].values
            valid = ~(np.isnan(w) | np.isnan(s))
            if valid.sum() >= 5:
                w_c = w[valid] - np.mean(w[valid])
                s_c = s[valid] - np.mean(s[valid])
                xcorr = np.correlate(w_c, s_c, mode="full") / (np.std(w_c) * np.std(s_c) * len(w_c))
                lags = np.arange(-len(w_c) + 1, len(w_c))
                ax.plot(lags, xcorr, color="mediumpurple")
                ax.axhline(0, color="black", linestyle="-", alpha=0.3)
                ax.set_xlim(-10, 10)

        ax.set_title(sd["storm_name"].iloc[0], fontsize=10)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Cross-corr")

    fig.suptitle("Wind × Sun Altitude Cross-Correlation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "crosscorr_wind_sun.png", dpi=150)
    plt.close(fig)

    print(f"    Saved autocorrelation_wind.png, crosscorr_wind_sun.png")


# ─────────────────────────────────────────────────────────────
# 5. Day/Night Cycle Analysis
# ─────────────────────────────────────────────────────────────
def day_night_analysis(df):
    print("\n[5] Day/Night Cycle Analysis")

    if "sun_altitude_deg" not in df.columns:
        print("    No sun altitude data, skipping")
        return

    df["is_day"] = df["sun_altitude_deg"] > 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Day vs Night wind distribution
    ax = axes[0]
    for label, color in [(True, "gold"), (False, "navy")]:
        subset = df[df["is_day"] == label]["wind"].dropna()
        ax.hist(subset, bins=25, alpha=0.6, color=color,
               label=f"{'Day' if label else 'Night'} (n={len(subset)})", edgecolor="black")
    ax.set_xlabel("Wind (kt)")
    ax.set_title("Wind Distribution: Day vs Night")
    ax.legend()

    # Mann-Whitney U test (non-parametric, appropriate for skewed wind data)
    day_wind = df[df["is_day"] == True]["wind"].dropna()
    night_wind = df[df["is_day"] == False]["wind"].dropna()
    u_stat, p_val = stats.mannwhitneyu(day_wind, night_wind, alternative="two-sided")

    ax = axes[1]
    ax.boxplot([day_wind, night_wind], labels=[f"Day\n(med={day_wind.median():.1f})",
                                                f"Night\n(med={night_wind.median():.1f})"])
    ax.set_ylabel("Wind (kt)")
    ax.set_title(f"Day vs Night Wind (U={u_stat:.0f}, p={p_val:.4f})")

    # Hour of day effect (median + IQR)
    ax = axes[2]
    hourly = df.groupby("hour_utc")["wind"].agg(
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75)
    )
    yerr_low = hourly["median"] - hourly["q1"]
    yerr_high = hourly["q3"] - hourly["median"]
    ax.errorbar(hourly.index, hourly["median"], yerr=[yerr_low, yerr_high], fmt="o-", capsize=3)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Wind (kt)")
    ax.set_title("Median Wind by Hour of Day (IQR)")

    fig.suptitle("Diurnal Cycle Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "day_night_analysis.png", dpi=150)
    plt.close(fig)

    print(f"    Day median={day_wind.median():.1f}, Night median={night_wind.median():.1f}, U={u_stat:.0f}, p={p_val:.4f}")
    print(f"    Saved day_night_analysis.png")


# ─────────────────────────────────────────────────────────────
# 6. Planetary Speed & Retrograde Analysis
# ─────────────────────────────────────────────────────────────
def retrograde_analysis(df):
    print("\n[6] Planetary Speed & Retrograde Analysis")

    from scipy.stats import mannwhitneyu as _mannwhitneyu

    vedic_planets = ["guru", "shani", "mangal", "budha", "shukra", "arun", "varun"]
    speed_cols = [f"vedic_{p}_speed_deg_per_day" for p in vedic_planets]
    speed_cols = [c for c in speed_cols if c in df.columns]

    if not speed_cols:
        print("    No Vedic speed columns found")
        return

    # Retrograde identification (negative speed)
    retro_results = {}
    for col in speed_cols:
        planet = col.replace("vedic_", "").replace("_speed_deg_per_day", "")
        df[f"retro_{planet}"] = df[col] < 0
        retro_count = df[f"retro_{planet}"].sum()
        retro_pct = retro_count / len(df) * 100

        if retro_count > 0:
            retro_vals = df[df[f"retro_{planet}"] == True]["wind"].dropna()
            direct_vals = df[df[f"retro_{planet}"] == False]["wind"].dropna()
            retro_wind = retro_vals.median()
            direct_wind = direct_vals.median()
            # Mann-Whitney U test (non-parametric, appropriate for skewed wind distribution)
            mw_stat, mw_p = (np.nan, np.nan)
            if len(retro_vals) >= 10 and len(direct_vals) >= 10:
                mw_stat, mw_p = _mannwhitneyu(retro_vals, direct_vals, alternative="two-sided")
            retro_results[planet] = {
                "retro_count": int(retro_count),
                "direct_count": int(len(direct_vals)),
                "retro_pct": float(retro_pct),
                "retro_median_wind": float(retro_wind),
                "direct_median_wind": float(direct_wind),
                "wind_diff_median": float(retro_wind - direct_wind),
                "mannwhitney_u": float(mw_stat) if pd.notna(mw_stat) else None,
                "mannwhitney_p": float(mw_p) if pd.notna(mw_p) else None,
                "significant_p05": bool(pd.notna(mw_p) and mw_p < 0.05),
            }

    if retro_results:
        retro_df = pd.DataFrame(retro_results).T
        retro_df.to_csv(OUT / "retrograde_analysis.csv")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Retrograde percentage
        ax = axes[0]
        retro_df["retro_pct"].plot.barh(ax=ax, color="mediumpurple")
        ax.set_title("Retrograde Percentage by Planet")
        ax.set_xlabel("% of Observations in Retrograde")

        # Wind difference (median)
        ax = axes[1]
        colors = ["red" if v > 0 else "blue" for v in retro_df["wind_diff_median"]]
        retro_df["wind_diff_median"].plot.barh(ax=ax, color=colors)
        ax.set_title("Median Wind Difference (Retrograde - Direct)")
        ax.set_xlabel("Δ Wind (kt)")
        ax.axvline(0, color="black", linestyle="-")

        fig.suptitle("Planetary Retrograde Effects on Storm Intensity", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / "retrograde_effects.png", dpi=150)
        plt.close(fig)

        print(f"    Analyzed {len(retro_results)} planets with retrograde periods")
        print(f"    Saved retrograde_analysis.csv, retrograde_effects.png")


# ─────────────────────────────────────────────────────────────
# 7. Hemisphere & Basin Analysis
# ─────────────────────────────────────────────────────────────
def basin_analysis(df):
    print("\n[7] Hemisphere & Basin Analysis")

    df["hemisphere"] = np.where(df["latitude"] >= 0, "Northern", "Southern")

    # Basin classification (approximate)
    def classify_basin(row):
        lon, lat = row["longitude"], row["latitude"]
        if pd.isna(lon) or pd.isna(lat):
            return "Unknown"
        # Normalise to -180..180 so dateline-crossing tracks (lon>180) are handled correctly
        lon = ((lon + 180) % 360) - 180
        if lat >= 0:
            if lon < -20:
                return "Atlantic"
            elif 100 <= lon <= 180:
                return "W Pacific"
            elif 30 <= lon < 100:
                return "N Indian"
            else:
                return "Other NH"
        else:
            if 30 <= lon <= 130:
                return "S Indian"
            elif 130 < lon <= 180 or lon < -70:
                return "S Pacific"
            else:
                return "Other SH"

    df["basin_geo"] = df.apply(classify_basin, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Storm count by hemisphere
    ax = axes[0]
    df.groupby("hemisphere")["storm_id"].nunique().plot.bar(ax=ax, color=["steelblue", "coral"])
    ax.set_title("Storms by Hemisphere")
    ax.set_ylabel("Count")

    # Median wind by basin with IQR error bars
    ax = axes[1]
    basin_stats = df.groupby("basin_geo")["wind"].agg(
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    )
    basin_stats.columns = ["median", "q1", "q3"]
    basin_stats = basin_stats.sort_values("median", ascending=False)
    yerr_low = basin_stats["median"] - basin_stats["q1"]
    yerr_high = basin_stats["q3"] - basin_stats["median"]
    ax.barh(range(len(basin_stats)), basin_stats["median"], color="seagreen",
            xerr=[yerr_low, yerr_high], capsize=4)
    ax.set_yticks(range(len(basin_stats)))
    ax.set_yticklabels(basin_stats.index)
    ax.set_title("Median Wind by Basin (IQR)")
    ax.set_xlabel("Wind (kt)")

    # Scatter by basin
    ax = axes[2]
    for basin in df["basin_geo"].unique():
        bd = df[df["basin_geo"] == basin]
        ax.scatter(bd["longitude"], bd["latitude"], label=basin, alpha=0.5, s=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Track Points by Basin")
    ax.legend(fontsize=7)

    fig.suptitle("Geographic Basin Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "basin_analysis.png", dpi=150)
    plt.close(fig)

    basin_summary = df.groupby("basin_geo").agg(
        n_storms=("storm_id", "nunique"),
        n_tracks=("track_num", "count"),
        median_wind=("wind", "median"),
        q1_wind=("wind", lambda x: x.quantile(0.25)),
        q3_wind=("wind", lambda x: x.quantile(0.75)),
        max_wind=("wind", "max"),
        mean_lat=("latitude", "mean"),
    ).round(2)
    basin_summary.to_csv(OUT / "basin_summary.csv")

    print(f"    Basins found: {df['basin_geo'].value_counts().to_dict()}")
    print(f"    Saved basin_analysis.png, basin_summary.csv")


# ─────────────────────────────────────────────────────────────
# 8. Statistical Normality Tests
# ─────────────────────────────────────────────────────────────
def normality_tests(df):
    print("\n[8] Statistical Normality Tests")

    test_cols = ["wind", "pressure", "latitude", "longitude",
                 "sun_altitude_deg", "moon_altitude_deg",
                 "sun_ecliptic_lon_deg", "moon_ecliptic_lon_deg"]
    test_cols = [c for c in test_cols if c in df.columns]

    results = {}
    for col in test_cols:
        data = df[col].dropna()
        if len(data) >= 8:
            # Shapiro-Wilk (use subsample if > 5000)
            sample = data.sample(min(5000, len(data)), random_state=42)
            sw_stat, sw_p = stats.shapiro(sample)

            # D'Agostino-Pearson
            try:
                dp_stat, dp_p = stats.normaltest(data)
            except Exception:
                dp_stat, dp_p = np.nan, np.nan

            # Skewness and Kurtosis
            skew = data.skew()
            kurt = data.kurtosis()

            results[col] = {
                "shapiro_stat": sw_stat, "shapiro_p": sw_p,
                "dagostino_stat": dp_stat, "dagostino_p": dp_p,
                "skewness": skew, "kurtosis": kurt,
                "is_normal_shapiro": sw_p > 0.05,
                "is_normal_dagostino": dp_p > 0.05 if pd.notna(dp_p) else None,
            }

    norm_df = pd.DataFrame(results).T
    norm_df.to_csv(OUT / "normality_tests.csv")

    # QQ plots
    n = len(test_cols)
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(4 * int(np.ceil(n / 2)), 8))
    axes = axes.flatten()

    for i, col in enumerate(test_cols):
        data = df[col].dropna()
        stats.probplot(data, plot=axes[i])
        axes[i].set_title(f"Q-Q: {col}", fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Q-Q Plots (Normality Check)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "qq_plots.png", dpi=150)
    plt.close(fig)

    normal_vars = norm_df[norm_df["is_normal_shapiro"] == True].index.tolist()
    print(f"    Normal variables (Shapiro p>0.05): {normal_vars if normal_vars else 'None'}")
    print(f"    Saved normality_tests.csv, qq_plots.png")


# ─────────────────────────────────────────────────────────────
# 9. Vedic Planetary Degree Distributions
# ─────────────────────────────────────────────────────────────
def vedic_degree_analysis(df):
    print("\n[9] Vedic Planetary Degree Analysis")

    vedic_degree_cols = [c for c in df.columns if c.startswith("vedic_") and c.endswith("_full_degree")]
    if not vedic_degree_cols:
        print("    No Vedic degree columns found")
        return

    # Distribute degrees on a circle (polar plot)
    n = len(vedic_degree_cols)
    fig, axes = plt.subplots(3, 5, figsize=(25, 15), subplot_kw={"projection": "polar"})
    axes = axes.flatten()

    for i, col in enumerate(vedic_degree_cols):
        if i >= 15:
            break
        ax = axes[i]
        planet_name = col.replace("vedic_", "").replace("_full_degree", "")
        degrees = df[col].dropna().values
        radians = np.deg2rad(degrees)

        # Color by wind
        winds = df.loc[df[col].notna(), "wind"]
        sc = ax.scatter(radians, winds, c=winds, cmap="YlOrRd", s=8, alpha=0.6)
        ax.set_title(planet_name, fontsize=9, pad=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Vedic Planet Positions (Sidereal Degrees) vs Wind", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "vedic_polar_wind.png", dpi=150)
    plt.close(fig)

    # Circular-linear correlation of Vedic degrees with wind.
    # Vedic planetary degrees are CYCLICAL (0–360°). Standard Pearson r assumes
    # linearity and treats degree 359 as far from degree 1, which is incorrect for
    # angular data. Instead, we decompose each degree into sin and cos components and
    # compute Pearson r for each. The max |r| of the two components is reported as the
    # effective circular-linear correlation.
    from scipy.stats import pearsonr as _pearsonr
    circ_rows = []
    for col in vedic_degree_cols:
        tmp = df[[col, "wind"]].dropna()
        if len(tmp) < 30:
            circ_rows.append({"feature": col, "r_sin": np.nan, "r_cos": np.nan, "r_circular_max": np.nan, "n": len(tmp)})
            continue
        rad = np.deg2rad(tmp[col].values)
        r_sin, _ = _pearsonr(np.sin(rad), tmp["wind"].values)
        r_cos, _ = _pearsonr(np.cos(rad), tmp["wind"].values)
        r_circ = r_sin if abs(r_sin) >= abs(r_cos) else r_cos
        circ_rows.append({
            "feature": col,
            "r_sin": float(r_sin),
            "r_cos": float(r_cos),
            "r_circular_max": float(r_circ),
            "n": int(len(tmp)),
            "note": "Circular-linear: max(|r_sin|, |r_cos|); NOT naive Pearson on raw degree"
        })
    vedic_circ_df = pd.DataFrame(circ_rows).sort_values("r_circular_max", ascending=False)
    vedic_circ_df.to_csv(OUT / "vedic_degree_correlations.csv", index=False)

    # Also compute naive Pearson for comparison (to show the difference)
    vedic_corr_naive = df[vedic_degree_cols + ["wind"]].corr()["wind"].drop("wind").sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ax = axes[0]
    r_circ_vals = vedic_circ_df.set_index("feature")["r_circular_max"]
    r_circ_vals.plot.barh(ax=ax, color=["steelblue" if v > 0 else "coral" for v in r_circ_vals.values])
    ax.set_title("Vedic Degree vs Wind: Circular-Linear Correlation\n(max of sin/cos decomposition)")
    ax.set_xlabel("Circular-Linear r")
    ax.axvline(0, color="black", linestyle="-")

    ax = axes[1]
    vedic_corr_naive.plot.barh(ax=ax, color=["steelblue" if v > 0 else "coral" for v in vedic_corr_naive.values])
    ax.set_title("Vedic Degree vs Wind: Naive Pearson r (INCORRECT for cyclical data)\nShown for comparison only")
    ax.set_xlabel("Pearson r (raw degree)")
    ax.axvline(0, color="black", linestyle="-")

    fig.suptitle("Vedic Planetary Degree Correlation with Wind", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "vedic_degree_corr_bar.png", dpi=150)
    plt.close(fig)

    print(f"    Analyzed {len(vedic_degree_cols)} Vedic planetary degrees (circular-linear method)")
    print(f"    Saved vedic_polar_wind.png, vedic_degree_correlations.csv")


# ─────────────────────────────────────────────────────────────
# 10. VIF (Variance Inflation Factor)
# ─────────────────────────────────────────────────────────────
def vif_analysis(df):
    print("\n[10] Variance Inflation Factor (Multicollinearity)")

    # Use a subset of important features to keep VIF manageable
    cols = ["latitude", "longitude", "hour_utc", "day_of_year",
            "sun_altitude_deg", "sun_azimuth_deg", "moon_altitude_deg", "moon_azimuth_deg",
            "sun_ecliptic_lon_deg", "moon_ecliptic_lon_deg",
            "jupiter_altitude_deg", "saturn_altitude_deg",
            "dinamana_minutes", "elevation"]
    cols = [c for c in cols if c in df.columns]

    df_clean = df[cols].dropna()
    if len(df_clean) < 20:
        print("    Insufficient data for VIF")
        return

    from numpy.linalg import inv
    X = df_clean.values
    X = StandardScaler().fit_transform(X)
    corr_matrix = np.corrcoef(X, rowvar=False)

    try:
        corr_inv = inv(corr_matrix)
        vif = np.diag(corr_inv)
        vif_df = pd.DataFrame({"feature": cols, "VIF": vif}).sort_values("VIF", ascending=False)
    except Exception:
        # Fallback: compute VIF one at a time
        from sklearn.linear_model import LinearRegression
        vifs = []
        for i, col in enumerate(cols):
            others = [c for c in cols if c != col]
            X_others = df_clean[others].values
            y_col = df_clean[col].values
            r2 = LinearRegression().fit(X_others, y_col).score(X_others, y_col)
            vif_val = 1 / (1 - r2) if r2 < 1 else np.inf
            vifs.append({"feature": col, "VIF": vif_val})
        vif_df = pd.DataFrame(vifs).sort_values("VIF", ascending=False)

    vif_df.to_csv(OUT / "vif_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["red" if v > 10 else "orange" if v > 5 else "green" for v in vif_df["VIF"]]
    ax.barh(range(len(vif_df)), vif_df["VIF"].values, color=colors)
    ax.set_yticks(range(len(vif_df)))
    ax.set_yticklabels(vif_df["feature"].values, fontsize=9)
    ax.axvline(5, color="orange", linestyle="--", label="Moderate (VIF=5)")
    ax.axvline(10, color="red", linestyle="--", label="High (VIF=10)")
    ax.set_title("Variance Inflation Factor")
    ax.set_xlabel("VIF")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "vif_barplot.png", dpi=150)
    plt.close(fig)

    high_vif = vif_df[vif_df["VIF"] > 10]["feature"].tolist()
    print(f"    High VIF (>10) features: {high_vif}")
    print(f"    Saved vif_analysis.csv, vif_barplot.png")


# ─────────────────────────────────────────────────────────────
# 11. Spectral / Periodicity Analysis
# ─────────────────────────────────────────────────────────────
def spectral_analysis(df):
    print("\n[11] Spectral / Periodicity Analysis")

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, sid in enumerate(df["storm_id"].unique()):
        if i >= 8:
            break
        sd = df[df["storm_id"] == sid].sort_values("track_num")
        ax = axes[i]

        wind = sd["wind"].dropna().values
        if len(wind) >= 8:
            freqs, psd = periodogram(wind, fs=1.0)  # fs=1 (per observation)
            ax.semilogy(freqs[1:], psd[1:], color="steelblue")
            ax.set_title(sd["storm_name"].iloc[0], fontsize=10)
            ax.set_xlabel("Frequency")
            ax.set_ylabel("PSD")

    fig.suptitle("Power Spectral Density of Wind Speed", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "spectral_analysis.png", dpi=150)
    plt.close(fig)

    print(f"    Saved spectral_analysis.png")


# ─────────────────────────────────────────────────────────────
# 12. Comprehensive Summary Report
# ─────────────────────────────────────────────────────────────
def summary_report(df):
    print("\n[12] Generating Comprehensive Summary Report")

    report = []
    report.append("=" * 80)
    report.append("  COMPREHENSIVE EDA SUMMARY REPORT — 2022-2026 Storm Data")
    report.append("=" * 80)
    report.append("")
    report.append(f"Dataset: {df.shape[0]} observations, {df.shape[1]} columns")
    report.append(f"Storms: {df['storm_id'].nunique()}")
    report.append(f"Date range: {df['date_utc'].min()} to {df['date_utc'].max()}")
    report.append("")

    # Key statistics
    report.append("--- KEY STATISTICS ---")
    report.append(f"Wind: mean={df['wind'].mean():.1f}, std={df['wind'].std():.1f}, "
                  f"min={df['wind'].min():.0f}, max={df['wind'].max():.0f}")
    p = df['pressure'].dropna()
    report.append(f"Pressure: mean={p.mean():.1f}, std={p.std():.1f}, "
                  f"min={p.min():.0f}, max={p.max():.0f} (n={len(p)})")
    report.append(f"Latitude range: [{df['latitude'].min():.1f}, {df['latitude'].max():.1f}]")
    report.append(f"Longitude range: [{df['longitude'].min():.1f}, {df['longitude'].max():.1f}]")
    report.append("")

    # Feature counts
    num_cols = df.select_dtypes(include=[np.number]).columns
    report.append("--- FEATURE COUNTS ---")
    report.append(f"Total numeric features: {len(num_cols)}")
    astro_cols = [c for c in num_cols if any(c.startswith(b + "_") for b in
                  ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"])]
    vedic_cols = [c for c in num_cols if c.startswith("vedic_")]
    report.append(f"Astronomical (Skyfield) features: {len(astro_cols)}")
    report.append(f"Vedic planetary features: {len(vedic_cols)}")
    report.append(f"Derived features: ~10 (wind_change, displacement, speed, etc.)")
    report.append("")

    # Missing data
    missing_pct = (df.isnull().sum() / len(df) * 100)
    high_miss = missing_pct[missing_pct > 5].sort_values(ascending=False)
    report.append("--- MISSING DATA (>5%) ---")
    for col, pct in high_miss.items():
        report.append(f"  {col}: {pct:.1f}%")
    report.append("")

    report_text = "\n".join(report)
    with open(OUT / "comprehensive_summary.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n    Saved comprehensive_summary.txt")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  COMPREHENSIVE EDA — 2022-2026 Storm Data")
    print("=" * 60)

    df = load()

    lifecycle_analysis(df)
    vedic_crosstab(df)
    anomaly_detection(df)
    autocorrelation_analysis(df)
    day_night_analysis(df)
    retrograde_analysis(df)
    basin_analysis(df)
    normality_tests(df)
    vedic_degree_analysis(df)
    vif_analysis(df)
    spectral_analysis(df)
    summary_report(df)

    print("\n" + "=" * 60)
    print(f"  Comprehensive EDA complete. All outputs in: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
