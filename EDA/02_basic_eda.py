"""
02_basic_eda.py
===============
Basic Exploratory Data Analysis on the flattened 2024-2026 storm data.
Generates: descriptive stats, missing value analysis, distributions,
correlation matrices, storm trajectory maps, and box plots.
All outputs saved to EDA/output/basic_eda/
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

OUT = Path(__file__).resolve().parent / "output" / "basic_eda"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(__file__).resolve().parent / "output" / "storm_data_all.parquet"

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)


def load():
    df = pd.read_parquet(DATA)
    # Coerce core numeric columns that may survive as object after Parquet round-trip
    numeric_cols = [
        "wind", "pressure", "longitude", "latitude",
        "displacement", "movement_speed", "wind_change", "pressure_change",
        "spline_length_deg", "spline_displacement", "spline_bearing_deg", "spline_curvature",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────
# 1. Descriptive Statistics
# ─────────────────────────────────────────────────────────────
def descriptive_stats(df):
    print("\n[1] Descriptive Statistics")
    desc = df.describe(include="all").T
    desc.to_csv(OUT / "descriptive_stats.csv")
    print(f"    Saved descriptive_stats.csv ({desc.shape})")

    # Core meteorological stats
    core_cols = ["wind", "pressure", "longitude", "latitude", "displacement",
                 "movement_speed", "wind_change", "pressure_change"]
    core_cols = [c for c in core_cols if c in df.columns]
    core_desc = df[core_cols].describe().T
    core_desc.to_csv(OUT / "core_meteo_stats.csv")
    print(f"    Saved core_meteo_stats.csv")


# ─────────────────────────────────────────────────────────────
# 2. Missing Value Analysis
# ─────────────────────────────────────────────────────────────
def missing_analysis(df):
    print("\n[2] Missing Value Analysis")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"count": missing, "pct": missing_pct})
    missing_df = missing_df[missing_df["count"] > 0].sort_values("pct", ascending=False)
    missing_df.to_csv(OUT / "missing_values.csv")

    # Heatmap of missing values for key columns
    key_cols = [c for c in df.columns if df[c].isnull().sum() > 0 and df[c].isnull().sum() < len(df)]
    if len(key_cols) > 40:
        key_cols = key_cols[:40]

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(df[key_cols].isnull().T, cbar=True, cmap="YlOrRd", ax=ax)
    ax.set_title("Missing Values Heatmap (Top Columns with Missing Data)")
    ax.set_xlabel("Track Observation Index")
    fig.tight_layout()
    fig.savefig(OUT / "missing_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"    Saved missing_values.csv, missing_heatmap.png")


# ─────────────────────────────────────────────────────────────
# 3. Distribution Plots
# ─────────────────────────────────────────────────────────────
def distribution_plots(df):
    print("\n[3] Distribution Plots")

    # Wind distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    df["wind"].dropna().hist(bins=30, ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Wind Speed Distribution (knots)")
    ax.set_xlabel("Wind (kt)")
    ax.axvline(df["wind"].mean(), color="red", linestyle="--", label=f"Mean={df['wind'].mean():.1f}")
    ax.legend()

    ax = axes[0, 1]
    df["pressure"].dropna().hist(bins=30, ax=ax, color="coral", edgecolor="black")
    ax.set_title("Pressure Distribution (hPa)")
    ax.set_xlabel("Pressure (hPa)")
    ax.axvline(df["pressure"].dropna().mean(), color="red", linestyle="--",
               label=f"Mean={df['pressure'].dropna().mean():.1f}")
    ax.legend()

    ax = axes[1, 0]
    df["longitude"].hist(bins=30, ax=ax, color="seagreen", edgecolor="black")
    ax.set_title("Longitude Distribution")
    ax.set_xlabel("Longitude (°)")

    ax = axes[1, 1]
    df["latitude"].hist(bins=30, ax=ax, color="mediumpurple", edgecolor="black")
    ax.set_title("Latitude Distribution")
    ax.set_xlabel("Latitude (°)")

    fig.suptitle("Core Variable Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "distributions_core.png", dpi=150)
    plt.close(fig)

    # Wind distribution by storm type — KDE curves (no overlapping bars)
    from scipy.stats import gaussian_kde as _kde
    storm_types = df["storm_type"].dropna().unique()
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap_types = plt.cm.tab10(np.linspace(0, 1, len(sorted(storm_types))))
    for ci, st in enumerate(sorted(storm_types)):
        subset = df[df["storm_type"] == st]["wind"].dropna()
        if len(subset) < 5:
            continue
        xs = np.linspace(subset.min(), subset.max(), 300)
        kde = _kde(subset)
        ax.plot(xs, kde(xs), linewidth=2, label=f"{st} (n={len(subset)})", color=cmap_types[ci])
        ax.fill_between(xs, kde(xs), alpha=0.15, color=cmap_types[ci])
    ax.set_title("Wind Speed Distribution by Storm Type (KDE)", fontsize=13)
    ax.set_xlabel("Wind (kt)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9,
              borderaxespad=0, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "wind_by_storm_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Wind by storm — top 40 storms by peak wind, manual boxplot via violin
    top_storms_wind = (df.groupby("storm_name")["wind"].max()
                         .sort_values(ascending=False).head(40).index.tolist())
    df_top_w = df[df["storm_name"].isin(top_storms_wind)].copy()
    order_w = (df_top_w.groupby("storm_name")["wind"].max()
                       .sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots(figsize=(22, 9))
    groups_w = [df_top_w[df_top_w["storm_name"] == s]["wind"].dropna().values for s in order_w]
    bp = ax.boxplot(groups_w, labels=order_w, patch_artist=True, vert=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)
    ax.set_xticklabels(order_w, rotation=90, fontsize=7)
    ax.set_title("Wind by Storm (Top 40 by Peak Wind)", fontsize=13)
    ax.set_ylabel("Wind (kt)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "wind_by_storm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Pressure by storm — top 40 storms by lowest pressure
    df_pressure = df.dropna(subset=["pressure"])
    if len(df_pressure) > 0:
        top_storms_pres = (df_pressure.groupby("storm_name")["pressure"].min()
                                      .sort_values().head(40).index.tolist())
        df_top_p = df_pressure[df_pressure["storm_name"].isin(top_storms_pres)].copy()
        order_p = (df_top_p.groupby("storm_name")["pressure"].min()
                           .sort_values().index.tolist())
        fig, ax = plt.subplots(figsize=(22, 9))
        groups_p = [df_top_p[df_top_p["storm_name"] == s]["pressure"].dropna().values for s in order_p]
        bp = ax.boxplot(groups_p, labels=order_p, patch_artist=True, vert=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("tomato")
            patch.set_alpha(0.7)
        ax.set_xticklabels(order_p, rotation=90, fontsize=7)
        ax.set_title("Pressure by Storm (Top 40 Deepest)", fontsize=13)
        ax.set_ylabel("Pressure (hPa)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "pressure_by_storm.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"    Saved distributions_core.png, wind_by_storm_type.png, wind_by_storm.png, pressure_by_storm.png")


# ─────────────────────────────────────────────────────────────
# 4. Storm Track Trajectories
# ─────────────────────────────────────────────────────────────
def trajectory_plots(df):
    print("\n[4] Storm Track Trajectories")
    fig, ax = plt.subplots(figsize=(18, 10))

    storms = df["storm_id"].unique()
    year_colors = {2022: "crimson", 2023: "mediumpurple", 2024: "steelblue", 2025: "darkorange", 2026: "seagreen"}
    plotted_years = set()

    for sid in storms:
        sd = df[df["storm_id"] == sid].sort_values("track_num")
        try:
            yr = sd["date_utc"].dropna().iloc[0].year
        except Exception:
            yr = int(str(sid)[-4:]) if str(sid)[-4:].isdigit() else 2024
        color = year_colors.get(yr, "gray")
        lbl = str(yr) if yr not in plotted_years else "_nolegend_"
        plotted_years.add(yr)
        sd_valid = sd.dropna(subset=["longitude", "latitude"])
        if sd_valid.empty:
            continue
        ax.plot(sd_valid["longitude"], sd_valid["latitude"], "-o", markersize=2,
                label=lbl, color=color, linewidth=1, alpha=0.7)
        ax.plot(sd_valid["longitude"].iloc[0], sd_valid["latitude"].iloc[0], "s",
                color=color, markersize=5, zorder=5)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("2022-2026 Storm Tracks")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=12,
              title="Year", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "storm_tracks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Individual storm tracks colored by wind
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, sid in enumerate(storms):
        if i >= 8:
            break
        sd = df[df["storm_id"] == sid].sort_values("track_num")
        ax = axes[i]
        sc = ax.scatter(sd["longitude"], sd["latitude"], c=sd["wind"],
                       cmap="YlOrRd", s=20, edgecolors="black", linewidth=0.3)
        ax.plot(sd["longitude"], sd["latitude"], "-", alpha=0.3, color="gray")
        ax.set_title(sd["storm_name"].iloc[0], fontsize=10)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        plt.colorbar(sc, ax=ax, label="Wind (kt)")

    fig.suptitle("Individual Storm Tracks (colored by wind speed)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "storm_tracks_individual.png", dpi=150)
    plt.close(fig)

    print(f"    Saved storm_tracks.png, storm_tracks_individual.png")


# ─────────────────────────────────────────────────────────────
# 5. Correlation Analysis
# ─────────────────────────────────────────────────────────────
def correlation_analysis(df):
    print("\n[5] Correlation Analysis")

    # Core meteorological + astronomical correlations with wind
    astro_cols = [c for c in df.columns if any(b + "_" in c for b in
                  ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"])]
    target_cols = ["wind", "pressure"] + astro_cols
    target_cols = [c for c in target_cols if c in df.select_dtypes(include=[np.number]).columns]

    corr_with_wind = df[target_cols].corrwith(df["wind"]).sort_values(ascending=False)
    corr_with_wind.to_csv(OUT / "correlation_with_wind.csv")

    # Top correlations bar chart
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    top_pos = corr_with_wind.head(20)
    top_neg = corr_with_wind.tail(20)

    axes[0].barh(range(len(top_pos)), top_pos.values, color="steelblue")
    axes[0].set_yticks(range(len(top_pos)))
    axes[0].set_yticklabels(top_pos.index, fontsize=8)
    axes[0].set_title("Top 20 Positive Correlations with Wind")
    axes[0].set_xlabel("Correlation")

    axes[1].barh(range(len(top_neg)), top_neg.values, color="coral")
    axes[1].set_yticks(range(len(top_neg)))
    axes[1].set_yticklabels(top_neg.index, fontsize=8)
    axes[1].set_title("Top 20 Negative Correlations with Wind")
    axes[1].set_xlabel("Correlation")

    fig.suptitle("Feature Correlations with Wind Speed", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "correlations_with_wind.png", dpi=150)
    plt.close(fig)

    # Heatmap of core features
    core_features = ["wind", "pressure", "longitude", "latitude", "displacement",
                     "movement_speed", "hour_utc", "day_of_year",
                     "sun_altitude_deg", "moon_altitude_deg", "sun_azimuth_deg",
                     "moon_ecliptic_lon_deg", "jupiter_altitude_deg", "saturn_altitude_deg"]
    core_features = [c for c in core_features if c in df.columns]
    corr_mat = df[core_features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, square=True, linewidths=0.5)
    ax.set_title("Correlation Matrix — Core Features")
    fig.tight_layout()
    fig.savefig(OUT / "correlation_matrix_core.png", dpi=150)
    plt.close(fig)

    # Astronomical body correlation heatmap (altitude & azimuth only)
    alt_az_cols = [c for c in df.columns if c.endswith("_altitude_deg") or c.endswith("_azimuth_deg")]
    if alt_az_cols:
        corr_alt_az = df[alt_az_cols + ["wind"]].corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_alt_az, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    ax=ax, linewidths=0.5)
        ax.set_title("Celestial Body Altitude/Azimuth Correlations")
        fig.tight_layout()
        fig.savefig(OUT / "correlation_celestial_alt_az.png", dpi=150)
        plt.close(fig)

    print(f"    Saved correlation files and heatmaps")


# ─────────────────────────────────────────────────────────────
# 6. Categorical Variable Analysis
# ─────────────────────────────────────────────────────────────
def categorical_analysis(df):
    print("\n[6] Categorical Variable Analysis")

    cat_vars = ["tithi", "nakshatra", "yoga", "karana", "weekday", "paksha",
                "moonsign", "sunsign", "drik_ritu", "drik_ayana"]
    cat_vars = [c for c in cat_vars if c in df.columns]

    # Clean 'upto' suffixes for cleaner labels; preserve NaN as NaN (not "None"/"nan")
    for c in cat_vars:
        def _clean_val(x):
            if pd.isna(x):
                return np.nan
            s = str(x)
            if s.lower() in ("none", "nan", ""):
                return np.nan
            return s.split(" upto")[0].strip() if "upto" in s else s
        df[c + "_clean"] = df[c].apply(_clean_val)

    # Count plots
    n_vars = len(cat_vars)
    fig, axes = plt.subplots(int(np.ceil(n_vars / 2)), 2, figsize=(18, 4 * int(np.ceil(n_vars / 2))))
    axes = axes.flatten()

    for i, c in enumerate(cat_vars):
        ax = axes[i]
        counts = df[c + "_clean"].value_counts().head(15)
        counts.plot.barh(ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(f"{c.title()} Distribution")
        ax.set_xlabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Panchang Categorical Variable Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "categorical_distributions.png", dpi=150)
    plt.close(fig)

    # Mean wind by categorical variable
    fig, axes = plt.subplots(int(np.ceil(n_vars / 2)), 2, figsize=(18, 4 * int(np.ceil(n_vars / 2))))
    axes = axes.flatten()

    for i, c in enumerate(cat_vars):
        ax = axes[i]
        mean_wind = df.groupby(c + "_clean")["wind"].mean().sort_values(ascending=False).head(15)
        mean_wind.plot.barh(ax=ax, color="coral", edgecolor="black")
        ax.set_title(f"Mean Wind by {c.title()}")
        ax.set_xlabel("Mean Wind (kt)")
        ax.axvline(df["wind"].mean(), color="black", linestyle="--", alpha=0.5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Mean Wind Speed by Panchang Categories", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "mean_wind_by_category.png", dpi=150)
    plt.close(fig)

    # Median wind by categorical variable
    fig, axes = plt.subplots(int(np.ceil(n_vars / 2)), 2, figsize=(18, 4 * int(np.ceil(n_vars / 2))))
    axes = axes.flatten()

    overall_median = df["wind"].median()
    for i, c in enumerate(cat_vars):
        ax = axes[i]
        grp = df.groupby(c + "_clean")["wind"]
        median_wind = grp.median().sort_values(ascending=False).head(15)
        q1 = grp.quantile(0.25).reindex(median_wind.index)
        q3 = grp.quantile(0.75).reindex(median_wind.index)
        xerr_low  = (median_wind - q1).clip(lower=0).values
        xerr_high = (q3 - median_wind).clip(lower=0).values

        ax.barh(range(len(median_wind)), median_wind.values,
                color="steelblue", edgecolor="black", alpha=0.85)
        ax.errorbar(median_wind.values, range(len(median_wind)),
                    xerr=[xerr_low, xerr_high],
                    fmt="none", color="black", capsize=3, linewidth=1)
        ax.set_yticks(range(len(median_wind)))
        ax.set_yticklabels(median_wind.index, fontsize=8)
        ax.set_title(f"Median Wind by {c.title()}")
        ax.set_xlabel("Median Wind (kt)  [error bars = IQR]")
        ax.axvline(overall_median, color="tomato", linestyle="--", alpha=0.6,
                   label=f"Overall median={overall_median:.0f} kt")
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Median Wind Speed by Panchang Categories  (error bars = IQR)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "median_wind_by_category.png", dpi=150)
    plt.close(fig)

    print(f"    Saved categorical_distributions.png, mean_wind_by_category.png, median_wind_by_category.png")


# ─────────────────────────────────────────────────────────────
# 6b. Quartile Wind Screening
# ─────────────────────────────────────────────────────────────
def quartile_wind_analysis(df):
    """
    Checks whether mean is a reliable summary for wind by Panchang category.

    Approach
    --------
    1. Box plots per variable (median / IQR / outliers) — more honest than bars.
    2. Mean vs Median divergence — if they differ substantially the distribution
       is skewed and mean is a poor summary.
    3. Kruskal-Wallis H-test per variable — non-parametric ANOVA that tells us
       whether category groupings carry statistically significant wind differences
       (H₀: all categories drawn from the same distribution).
    4. Summary CSV with mean, median, Q1, Q3, IQR, std, skew, K-W H, p-value.
    """
    from scipy.stats import kruskal, skew as sp_skew
    print("\n[6b] Quartile Wind Screening")

    cat_vars = ["tithi", "nakshatra", "yoga", "karana", "weekday", "paksha",
                "moonsign", "sunsign", "drik_ritu", "drik_ayana"]
    cat_vars = [c + "_clean" for c in cat_vars if c + "_clean" in df.columns]

    overall_mean   = df["wind"].mean()
    overall_median = df["wind"].median()
    overall_skew   = sp_skew(df["wind"].dropna())
    print(f"    Overall wind — mean={overall_mean:.1f} kt  "
          f"median={overall_median:.1f} kt  skew={overall_skew:.3f}")
    if abs(overall_mean - overall_median) / df["wind"].std() > 0.15:
        print("    [!] Mean vs median notable divergence -- "
              "prefer median/IQR for category screening.")
    else:
        print("    [ok] Mean and median are close -- mean is acceptable.")

    # ── 1. Box plots per variable ────────────────────────────────────────────
    n_vars = len(cat_vars)
    rows = int(np.ceil(n_vars / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(22, 5 * rows))
    axes = axes.flatten()

    for i, c in enumerate(cat_vars):
        ax = axes[i]
        # Top-15 categories by median wind, descending
        order = (df.groupby(c)["wind"].median()
                   .sort_values(ascending=False).head(15).index.tolist())
        groups = [df[df[c] == cat]["wind"].dropna().values for cat in order]

        bp = ax.boxplot(groups, labels=order, patch_artist=True,
                        vert=True, notch=False, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.4,
                                        markerfacecolor="gray"))
        # Colour boxes by median value (yellow → red gradient)
        medians = [np.median(g) if len(g) else 0 for g in groups]
        norm = plt.Normalize(min(medians), max(medians) + 1e-9)
        cmap = plt.cm.YlOrRd
        for patch, med in zip(bp["boxes"], medians):
            patch.set_facecolor(cmap(norm(med)))
            patch.set_alpha(0.8)

        ax.axhline(overall_median, color="steelblue", linestyle="--",
                   linewidth=1.2, label=f"Overall median={overall_median:.1f}")
        ax.axhline(overall_mean, color="tomato", linestyle=":",
                   linewidth=1.2, label=f"Overall mean={overall_mean:.1f}")
        ax.set_title(c.replace("_clean", "").replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Wind (kt)")
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Wind Distribution by Panchang Category  "
                 "(box=Q1/median/Q3, whiskers=1.5×IQR, dots=outliers)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "quartile_wind_by_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Mean vs Median divergence chart ──────────────────────────────────
    fig, axes = plt.subplots(rows, 2, figsize=(22, 4 * rows))
    axes = axes.flatten()

    for i, c in enumerate(cat_vars):
        ax = axes[i]
        grp = df.groupby(c)["wind"]
        stats_local = grp.agg(["mean", "median", "std", "count"]).reset_index()
        stats_local = stats_local[stats_local["count"] >= 5]  # min sample guard
        stats_local["divergence"] = (stats_local["mean"] - stats_local["median"]).abs()
        stats_local = stats_local.sort_values("median", ascending=False).head(15)

        x = np.arange(len(stats_local))
        w = 0.35
        bars_mean   = ax.bar(x - w/2, stats_local["mean"],   w, label="Mean",
                             color="tomato",     alpha=0.8, edgecolor="black", linewidth=0.5)
        bars_median = ax.bar(x + w/2, stats_local["median"], w, label="Median",
                             color="steelblue",  alpha=0.8, edgecolor="black", linewidth=0.5)

        # Error bars on mean (±1 std)
        ax.errorbar(x - w/2, stats_local["mean"], yerr=stats_local["std"],
                    fmt="none", color="black", capsize=3, linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(stats_local[c], rotation=45, ha="right", fontsize=7)
        ax.set_title(c.replace("_clean", "").replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Wind (kt)")
        ax.axhline(overall_median, color="steelblue", linestyle="--", alpha=0.4, linewidth=1)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Mean (±σ) vs Median Wind by Panchang Category  "
                 "(large gaps → mean is noisy; trust median)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "mean_vs_median_wind.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 3. Kruskal-Wallis significance test ─────────────────────────────────
    # NOTE on multiple comparisons: Testing 10 Vedic categories with alpha=0.05
    # expects ~0.5 false positives by chance. We apply Benjamini-Hochberg FDR
    # correction to control the false discovery rate. Both raw and corrected
    # significance are reported. Only the BH-corrected column should be used
    # for drawing conclusions.
    # NOTE on collinearity: Panchang categories (Tithi, Nakshatra, Yoga, Karana,
    # Moonsign, etc.) are derived from Sun and Moon positions. Any correlation with
    # wind is likely a re-expression of the same astronomical signal already present
    # in the sun_*/moon_* columns. Significant KW tests may reflect this collinearity
    # rather than an independent categorical effect.
    from statsmodels.stats.multitest import multipletests as _multipletests
    kw_results = []
    for c in cat_vars:
        groups_kw = [grp["wind"].dropna().values
                     for _, grp in df.groupby(c)
                     if len(grp["wind"].dropna()) >= 5]
        if len(groups_kw) < 2:
            continue
        try:
            H, p = kruskal(*groups_kw)
        except Exception:
            H, p = np.nan, np.nan
        kw_results.append({
            "variable": c.replace("_clean", ""),
            "kw_H": round(H, 3),
            "kw_p": p,
            "significant_p05": p < 0.05,
            "n_categories": len(groups_kw),
        })

    kw_df = pd.DataFrame(kw_results)
    if not kw_df.empty:
        kw_df = kw_df.sort_values("kw_H", ascending=False)

        # Apply Benjamini-Hochberg FDR correction
        valid_mask = kw_df["kw_p"].notna()
        if valid_mask.sum() >= 2:
            reject, p_adj, _, _ = _multipletests(
                kw_df.loc[valid_mask, "kw_p"].values,
                alpha=0.05, method="fdr_bh"
            )
            kw_df.loc[valid_mask, "p_adjusted_bh"] = p_adj
            kw_df.loc[valid_mask, "significant_after_bh_correction"] = reject
        else:
            kw_df["p_adjusted_bh"] = kw_df["kw_p"]
            kw_df["significant_after_bh_correction"] = kw_df["significant_p05"]

    kw_df.to_csv(OUT / "kruskal_wallis_wind.csv", index=False)

    print(f"\n    Kruskal-Wallis results (H-stat, raw p<0.05, BH-corrected):")
    if kw_df.empty:
        print("      No variables had enough data for K-W test (need >=2 groups with >=5 obs each).")
    else:
        for _, row in kw_df.iterrows():
            raw_sig = "[sig]" if row["significant_p05"] else "[not sig]"
            bh_sig = "[BH sig]" if row.get("significant_after_bh_correction", False) else "[BH not sig]"
            p_adj = row.get("p_adjusted_bh", np.nan)
            print(f"      {row['variable']:<15}  H={row['kw_H']:.1f}  "
                  f"p={row['kw_p']:.4f} {raw_sig}  "
                  f"p_adj={p_adj:.4f} {bh_sig}")

    # ── 4. Summary stats CSV ─────────────────────────────────────────────────
    summary_rows = []
    for c in cat_vars:
        grp = df.groupby(c)["wind"]
        for cat_val, group_data in grp:
            vals = group_data.dropna().values
            if len(vals) < 3:
                continue
            q1, q3 = np.percentile(vals, [25, 75])
            summary_rows.append({
                "variable": c.replace("_clean", ""),
                "category": cat_val,
                "n": len(vals),
                "mean": round(vals.mean(), 2),
                "median": round(np.median(vals), 2),
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(q3 - q1, 2),
                "std": round(vals.std(), 2),
                "skewness": round(sp_skew(vals), 3),
                "mean_median_gap": round(abs(vals.mean() - np.median(vals)), 2),
            })

    summary_df = pd.DataFrame(summary_rows)
    # Wind is right-skewed (skew ~1.5+). Median is the preferred central tendency.
    # The 'mean' column is provided for reference but should not be used as the
    # primary summary statistic for category comparisons.
    summary_df["preferred_central_tendency"] = "median"
    summary_df["skew_note"] = "Wind is right-skewed; use median/IQR, not mean/std"
    summary_df.to_csv(OUT / "wind_quartile_summary.csv", index=False)

    print(f"\n    Saved quartile_wind_by_category.png, mean_vs_median_wind.png,")
    print(f"           kruskal_wallis_wind.csv (with BH-corrected p-values), wind_quartile_summary.csv")


# ─────────────────────────────────────────────────────────────
# 7. Time Series Plots
# ─────────────────────────────────────────────────────────────
def time_series_plots(df):
    print("\n[7] Time Series Plots")

    fig, axes = plt.subplots(3, 1, figsize=(22, 14), sharex=False)
    storms = df["storm_id"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(storms)))

    for i, sid in enumerate(storms):
        sd = df[df["storm_id"] == sid].sort_values("date_utc")
        sd_dated = sd.dropna(subset=["date_utc"])
        if sd_dated.empty:
            continue
        axes[0].plot(sd_dated["date_utc"], sd_dated["wind"], "-o", markersize=2,
                    label=sd_dated["storm_name"].iloc[0], color=colors[i], linewidth=1)
        axes[1].plot(sd_dated["date_utc"], sd_dated["pressure"], "-o", markersize=2,
                    color=colors[i], linewidth=1)
        if "displacement" in sd_dated.columns:
            axes[2].plot(sd_dated["date_utc"], sd_dated["displacement"], "-o", markersize=2,
                        color=colors[i], linewidth=1)

    axes[0].set_ylabel("Wind (kt)")
    axes[0].set_title("Wind Speed Over Time by Storm")
    # Legend outside the plot to the right, 2 columns, tiny font
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6,
                   ncol=2, borderaxespad=0, framealpha=0.7, title="Storm")
    axes[1].set_ylabel("Pressure (hPa)")
    axes[1].set_title("Pressure Over Time by Storm")
    axes[2].set_ylabel("Displacement (°)")
    axes[2].set_title("Track Displacement Over Time")
    axes[2].set_xlabel("Date (UTC)")

    fig.tight_layout()
    fig.savefig(OUT / "time_series.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"    Saved time_series.png")


# ─────────────────────────────────────────────────────────────
# 8. Wind-Pressure Relationship
# ─────────────────────────────────────────────────────────────
def wind_pressure_analysis(df):
    print("\n[8] Wind-Pressure Relationship")

    df_valid = df.dropna(subset=["wind", "pressure"])
    if len(df_valid) < 5:
        print("    Insufficient data with both wind and pressure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter
    ax = axes[0]
    for sid in df_valid["storm_id"].unique():
        sd = df_valid[df_valid["storm_id"] == sid]
        ax.scatter(sd["wind"], sd["pressure"], label=sd["storm_name"].iloc[0],
                  s=20, alpha=0.7)
    ax.set_xlabel("Wind (kt)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title("Wind vs Pressure")
    ax.legend(fontsize=7)

    # Regression
    ax = axes[1]
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(df_valid["wind"], df_valid["pressure"])
    ax.scatter(df_valid["wind"], df_valid["pressure"], alpha=0.4, s=15)
    x_line = np.linspace(df_valid["wind"].min(), df_valid["wind"].max(), 100)
    ax.plot(x_line, intercept + slope * x_line, "r-", linewidth=2,
            label=f"y={slope:.2f}x+{intercept:.0f}\nR²={r**2:.3f}, p={p:.2e}")
    ax.set_xlabel("Wind (kt)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title("Wind-Pressure Linear Regression")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "wind_pressure.png", dpi=150)
    plt.close(fig)

    print(f"    R²={r**2:.4f}, slope={slope:.2f}, p={p:.2e}")
    print(f"    Saved wind_pressure.png")


# ─────────────────────────────────────────────────────────────
# 9. Per-Storm Summary Table
# ─────────────────────────────────────────────────────────────
def storm_summary(df):
    print("\n[9] Per-Storm Summary")

    summary = df.groupby(["storm_id", "storm_name", "storm_type"]).agg(
        n_tracks=("track_num", "count"),
        max_wind=("wind", "max"),
        min_pressure=("pressure", "min"),
        mean_wind=("wind", "mean"),
        std_wind=("wind", "std"),
        total_displacement=("displacement", "sum"),
        max_displacement=("displacement", "max"),
        lat_range=("latitude", lambda x: x.max() - x.min()),
        lon_range=("longitude", lambda x: x.max() - x.min()),
        duration_hours=("hours_elapsed", "sum"),
    ).round(2)

    summary.to_csv(OUT / "storm_summary.csv")
    print(f"    Saved storm_summary.csv")
    print(summary.to_string())


# ─────────────────────────────────────────────────────────────
# 10. Pair Plot of Key Features
# ─────────────────────────────────────────────────────────────
def pairplot_key_features(df):
    print("\n[10] Pair Plot of Key Features")

    pair_cols = ["wind", "pressure", "latitude", "longitude", "sun_altitude_deg",
                 "moon_altitude_deg", "storm_type"]
    pair_cols = [c for c in pair_cols if c in df.columns]

    df_clean = df[pair_cols].dropna()
    if len(df_clean) > 10:
        g = sns.pairplot(df_clean, hue="storm_type", diag_kind="kde",
                        plot_kws={"alpha": 0.5, "s": 15})
        g.figure.suptitle("Key Features Pair Plot", y=1.02, fontsize=14, fontweight="bold")
        g.figure.savefig(OUT / "pairplot_key.png", dpi=120)
        plt.close(g.figure)
        print(f"    Saved pairplot_key.png")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BASIC EDA — 2022-2026 Storm Data")
    print("=" * 60)

    df = load()

    descriptive_stats(df)
    missing_analysis(df)
    distribution_plots(df)
    trajectory_plots(df)
    correlation_analysis(df)
    categorical_analysis(df)
    quartile_wind_analysis(df)
    time_series_plots(df)
    wind_pressure_analysis(df)
    storm_summary(df)
    pairplot_key_features(df)

    print("\n" + "=" * 60)
    print(f"  Basic EDA complete. All outputs in: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
