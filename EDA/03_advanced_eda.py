"""
03_advanced_eda.py
==================
Advanced EDA: PCA, Clustering, Feature Importance (Random Forest, XGBoost, LightGBM),
Mutual Information, Time-series decomposition, Interaction effects.
All outputs saved to EDA/output/advanced_eda/
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

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score, GroupKFold
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
import xgboost as xgb
import lightgbm as lgb

OUT = Path(__file__).resolve().parent / "output" / "advanced_eda"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(__file__).resolve().parent / "output" / "storm_data_all.parquet"

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.0)


def load():
    df = pd.read_parquet(DATA)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_numeric_features(df, exclude_targets=True):
    """Get clean numeric feature columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude target/derived/ID columns
    exclude = ["track_num", "storm_max_wind", "storm_forecast"]
    if exclude_targets:
        exclude += ["wind", "pressure", "wind_change", "pressure_change"]

    # Exclude pure derived columns that would leak
    exclude += ["movement_speed", "displacement", "lon_change", "lat_change", "hours_elapsed"]

    return [c for c in num_cols if c not in exclude and df[c].notna().sum() > len(df) * 0.5]


# ─────────────────────────────────────────────────────────────
# 1. PCA on Astronomical Features
# ─────────────────────────────────────────────────────────────
def pca_analysis(df):
    print("\n[1] PCA on Astronomical Features")

    # Skyfield astronomical columns
    astro_cols = [c for c in df.columns if any(c.startswith(b + "_") for b in
                  ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"])]
    astro_cols = [c for c in astro_cols if c in df.select_dtypes(include=[np.number]).columns]

    df_astro = df[astro_cols].dropna()
    if len(df_astro) < 10:
        print("    Insufficient data for PCA")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_astro)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    axes[0].plot(range(1, len(cumvar) + 1), cumvar, "bo-", markersize=3)
    axes[0].set_xlabel("Number of Components")
    axes[0].set_ylabel("Cumulative Explained Variance")
    axes[0].set_title("PCA — Explained Variance (Astronomical Features)")
    axes[0].axhline(0.95, color="red", linestyle="--", label="95%")
    axes[0].axhline(0.90, color="orange", linestyle="--", label="90%")
    n95 = np.argmax(cumvar >= 0.95) + 1
    n90 = np.argmax(cumvar >= 0.90) + 1
    axes[0].axvline(n95, color="red", linestyle=":", alpha=0.5)
    axes[0].legend()

    # First 2 PCs colored by wind
    wind_vals = df.loc[df_astro.index, "wind"]
    sc = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=wind_vals, cmap="YlOrRd",
                        s=15, alpha=0.7, edgecolors="black", linewidth=0.2)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].set_title("PC1 vs PC2 (colored by Wind)")
    plt.colorbar(sc, ax=axes[1], label="Wind (kt)")

    fig.tight_layout()
    fig.savefig(OUT / "pca_astronomical.png", dpi=150)
    plt.close(fig)

    # PCA loadings for top components
    loadings = pd.DataFrame(pca.components_[:5], columns=astro_cols,
                           index=[f"PC{i+1}" for i in range(5)])
    loadings.to_csv(OUT / "pca_loadings.csv")

    # Top loadings for PC1
    pc1_loadings = loadings.loc["PC1"].abs().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    pc1_loadings.head(20).plot.barh(ax=ax, color="steelblue")
    ax.set_title("Top 20 Features by |PC1 Loading|")
    ax.set_xlabel("|Loading|")
    fig.tight_layout()
    fig.savefig(OUT / "pca_pc1_loadings.png", dpi=150)
    plt.close(fig)

    print(f"    90% variance at {n90} components, 95% at {n95} components (of {len(astro_cols)})")
    print(f"    Saved pca_astronomical.png, pca_loadings.csv, pca_pc1_loadings.png")


# ─────────────────────────────────────────────────────────────
# 2. Feature Importance — Multiple Models
# ─────────────────────────────────────────────────────────────
def feature_importance(df):
    print("\n[2] Feature Importance (RF, XGBoost, LightGBM)")

    feature_cols = get_numeric_features(df, exclude_targets=True)
    target = "wind"

    df_clean = df[feature_cols + [target, "storm_id"]].dropna(subset=feature_cols + [target])
    X = df_clean[feature_cols].values
    y = df_clean[target].values
    groups = df_clean["storm_id"].values  # for GroupKFold

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # GroupKFold prevents the same storm from appearing in both train and test folds.
    # Standard KFold (cv=5) would allow observations from the same storm to appear in
    # both train and test, inflating CV R² due to within-storm autocorrelation.
    gkf = GroupKFold(n_splits=5)

    results = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    results["RandomForest"] = rf_imp
    rf_cv = cross_val_score(rf, X_scaled, y, cv=gkf, groups=groups, scoring="r2")
    print(f"    RF Group CV R²: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  random_state=42, n_jobs=-1, verbosity=0)
    xgb_model.fit(X_scaled, y)
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    results["XGBoost"] = xgb_imp
    xgb_cv = cross_val_score(xgb_model, X_scaled, y, cv=gkf, groups=groups, scoring="r2")
    print(f"    XGB Group CV R²: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")

    # LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_scaled, y)
    lgb_imp = pd.Series(lgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    lgb_imp = lgb_imp / lgb_imp.sum()  # normalize
    results["LightGBM"] = lgb_imp
    lgb_cv = cross_val_score(lgb_model, X_scaled, y, cv=gkf, groups=groups, scoring="r2")
    print(f"    LGB Group CV R²: {lgb_cv.mean():.4f} ± {lgb_cv.std():.4f}")

    # Combined importance
    all_imp = pd.DataFrame(results)
    all_imp["mean_importance"] = all_imp.mean(axis=1)
    all_imp = all_imp.sort_values("mean_importance", ascending=False)
    all_imp.to_csv(OUT / "feature_importance_all.csv")

    # Plot top 30
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    for i, (name, imp) in enumerate(results.items()):
        top30 = imp.head(30)
        axes[i].barh(range(len(top30)), top30.values, color=["steelblue", "coral", "seagreen"][i])
        axes[i].set_yticks(range(len(top30)))
        axes[i].set_yticklabels(top30.index, fontsize=7)
        axes[i].set_title(f"{name} — Top 30")
        axes[i].set_xlabel("Importance")
        axes[i].invert_yaxis()

    fig.suptitle("Feature Importance for Wind Speed Prediction", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance_top30.png", dpi=150)
    plt.close(fig)

    # Mean importance — top and bottom
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    top25 = all_imp["mean_importance"].head(25)
    axes[0].barh(range(len(top25)), top25.values, color="steelblue")
    axes[0].set_yticks(range(len(top25)))
    axes[0].set_yticklabels(top25.index, fontsize=8)
    axes[0].set_title("Top 25 Most Important Features (Mean)")
    axes[0].invert_yaxis()

    bottom25 = all_imp["mean_importance"].tail(25)
    axes[1].barh(range(len(bottom25)), bottom25.values, color="lightcoral")
    axes[1].set_yticks(range(len(bottom25)))
    axes[1].set_yticklabels(bottom25.index, fontsize=8)
    axes[1].set_title("Bottom 25 Least Important Features (Mean)")
    axes[1].invert_yaxis()

    fig.suptitle("Important vs Unimportant Variables", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "important_vs_unimportant.png", dpi=150)
    plt.close(fig)

    print(f"    Saved feature_importance_all.csv, feature_importance_top30.png, important_vs_unimportant.png")

    # Save summary of important/unimportant
    threshold = all_imp["mean_importance"].median()
    important = all_imp[all_imp["mean_importance"] >= threshold].index.tolist()
    unimportant = all_imp[all_imp["mean_importance"] < threshold].index.tolist()

    any_negative_r2 = any([rf_cv.mean() < 0, xgb_cv.mean() < 0, lgb_cv.mean() < 0])

    with open(OUT / "variable_importance_summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("VARIABLE IMPORTANCE SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        if any_negative_r2:
            f.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            f.write("CAUTION: ONE OR MORE MODELS SHOW NEGATIVE GROUP CV R².\n")
            f.write("Negative R² means the model performs WORSE than a mean\n")
            f.write("baseline (predicting the average wind for every observation).\n")
            f.write("Feature importance rankings computed from these models are\n")
            f.write("unreliable — they reflect the model's internal structure but\n")
            f.write("NOT predictive power over unseen storms.\n")
            f.write("This could be due to:\n")
            f.write("  1. Insufficient signal in astronomical/vedic features alone.\n")
            f.write("  2. High variance across storms (each storm is unique).\n")
            f.write("  3. Missing meteorological drivers (SST, wind shear, etc.).\n")
            f.write("Rankings below show relative importance WITHIN these models,\n")
            f.write("not evidence that any feature is truly predictive of wind.\n")
            f.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
        f.write(f"Model Group CV R² Scores (GroupKFold=5, grouped by storm_id):\n")
        f.write(f"  Random Forest:  {rf_cv.mean():.4f} ± {rf_cv.std():.4f}\n")
        f.write(f"  XGBoost:        {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}\n")
        f.write(f"  LightGBM:       {lgb_cv.mean():.4f} ± {lgb_cv.std():.4f}\n\n")
        f.write(f"IMPORTANT Variables ({len(important)}):\n")
        for v in important:
            f.write(f"  {v}: {all_imp.loc[v, 'mean_importance']:.6f}\n")
        f.write(f"\nUNIMPORTANT Variables ({len(unimportant)}):\n")
        for v in unimportant:
            f.write(f"  {v}: {all_imp.loc[v, 'mean_importance']:.6f}\n")

    print(f"    Saved variable_importance_summary.txt")
    return results


# ─────────────────────────────────────────────────────────────
# 3. Mutual Information
# ─────────────────────────────────────────────────────────────
def mutual_information_analysis(df):
    print("\n[3] Mutual Information Analysis")

    feature_cols = get_numeric_features(df, exclude_targets=True)
    target = "wind"

    df_clean = df[feature_cols + [target]].dropna()
    X = df_clean[feature_cols].values
    y = df_clean[target].values

    mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
    mi_series.to_csv(OUT / "mutual_information.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    top30 = mi_series.head(30)
    top30.plot.barh(ax=ax, color="mediumpurple")
    ax.set_title("Top 30 Features by Mutual Information with Wind")
    ax.set_xlabel("Mutual Information")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "mutual_information_top30.png", dpi=150)
    plt.close(fig)

    print(f"    Saved mutual_information.csv, mutual_information_top30.png")


# ─────────────────────────────────────────────────────────────
# 4. Clustering Analysis
# ─────────────────────────────────────────────────────────────
def clustering_analysis(df):
    print("\n[4] Clustering Analysis")

    feature_cols = get_numeric_features(df, exclude_targets=False)
    # Use a subset for clustering
    cluster_cols = ["wind", "pressure", "latitude", "longitude",
                    "sun_altitude_deg", "moon_altitude_deg", "sun_azimuth_deg",
                    "moon_ecliptic_lon_deg"]
    cluster_cols = [c for c in cluster_cols if c in feature_cols]

    df_clean = df[cluster_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # Elbow method
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(K_range), inertias, "bo-")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method for KMeans")

    # KMeans with k=4
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    df_clean_idx = df_clean.index
    df.loc[df_clean_idx, "cluster"] = labels

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set1", s=15, alpha=0.7)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("KMeans Clusters (k=4) in PCA Space")

    fig.tight_layout()
    fig.savefig(OUT / "clustering_kmeans.png", dpi=150)
    plt.close(fig)

    # Cluster profiles
    df_clustered = df.loc[df_clean_idx].copy()
    df_clustered["cluster"] = labels
    cluster_profile = df_clustered.groupby("cluster")[
        ["wind", "pressure", "latitude", "longitude"]
    ].agg(["mean", "std", "count"])
    cluster_profile.to_csv(OUT / "cluster_profiles.csv")

    # Hierarchical clustering dendrogram (on storm-level aggregates)
    # Cap at 100 storms (highest-peak-wind) to keep figure renderable
    storm_agg = df.groupby("storm_id")[cluster_cols].mean().dropna()
    if len(storm_agg) > 100:
        top_wind = df.groupby("storm_id")["wind"].max().nlargest(100).index
        storm_agg = storm_agg.loc[storm_agg.index.isin(top_wind)]
    if len(storm_agg) >= 3:
        Z = linkage(storm_agg.values, method="ward")
        n_leaves = len(storm_agg)
        fig_w = max(20, n_leaves * 0.32)
        fig, ax = plt.subplots(figsize=(fig_w, 10))
        dendrogram(Z, labels=storm_agg.index.tolist(), ax=ax,
                   leaf_font_size=6, leaf_rotation=90)
        ax.set_title(f"Hierarchical Clustering of Storms (top {n_leaves} by peak wind)", fontsize=13)
        ax.set_ylabel("Distance")
        fig.tight_layout()
        fig.savefig(OUT / "dendrogram_storms.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"    Saved clustering_kmeans.png, cluster_profiles.csv, dendrogram_storms.png")


# ─────────────────────────────────────────────────────────────
# 5. Vedic Category vs Intensity Analysis
# ─────────────────────────────────────────────────────────────
def vedic_intensity_analysis(df):
    print("\n[5] Vedic Category vs Intensity Analysis")

    vedic_cats = ["tithi", "nakshatra", "yoga", "karana", "paksha", "moonsign", "weekday"]
    vedic_cats = [c for c in vedic_cats if c in df.columns]

    from scipy.stats import kruskal

    results = {}
    for cat in vedic_cats:
        df_temp = df[[cat, "wind"]].dropna()
        df_temp = df_temp.copy()
        df_temp[cat + "_clean"] = df_temp[cat].astype(str).apply(
            lambda x: x.split(" upto")[0].strip())
        groups = df_temp.groupby(cat + "_clean")["wind"]
        group_list = [g.values for _, g in groups if len(g) >= 3]

        if len(group_list) >= 2:
            try:
                H, p_val = kruskal(*group_list)
            except Exception:
                H, p_val = float("nan"), 1.0
            results[cat] = {"H-statistic": H, "p-value": p_val,
                            "n_groups": len(group_list),
                            "significant": p_val < 0.05 if not pd.isna(p_val) else False}

    kw_df = pd.DataFrame(results).T
    kw_df.to_csv(OUT / "vedic_kruskal_wallis.csv")

    # Box plots for significant categories, sorted by median
    sig_cats = [c for c in vedic_cats if results.get(c, {}).get("significant", False)]

    if sig_cats:
        fig, axes = plt.subplots(len(sig_cats), 1, figsize=(14, 5 * len(sig_cats)))
        if len(sig_cats) == 1:
            axes = [axes]
        for i, cat in enumerate(sig_cats):
            df_temp = df[[cat, "wind"]].dropna().copy()
            df_temp[cat + "_clean"] = df_temp[cat].astype(str).apply(
                lambda x: x.split(" upto")[0].strip())
            # Sort by median (more robust than mean for skewed wind data)
            order = (df_temp.groupby(cat + "_clean")["wind"]
                     .median().sort_values(ascending=False).index)
            sns.boxplot(data=df_temp, x=cat + "_clean", y="wind", order=order, ax=axes[i])
            H_val = results[cat]['H-statistic']
            p_val = results[cat]['p-value']
            axes[i].set_title(
                f"Wind by {cat.title()} "
                f"(Kruskal-Wallis H={H_val:.1f}, p={p_val:.4f}) — sorted by median")
            axes[i].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(OUT / "vedic_significant_boxplots.png", dpi=150)
        plt.close(fig)
        print(f"    Significant Vedic categories (KW p<0.05): {sig_cats}")
    else:
        print("    No Vedic categories showed significant Kruskal-Wallis results")

    print(f"    Saved vedic_kruskal_wallis.csv")


# ─────────────────────────────────────────────────────────────
# 6. Feature Correlation Clustering
# ─────────────────────────────────────────────────────────────
def feature_correlation_clustering(df):
    print("\n[6] Feature Correlation Clustering (identify redundant features)")

    feature_cols = get_numeric_features(df, exclude_targets=False)
    corr = df[feature_cols].corr().abs()
    # Drop columns/rows that are all NaN, fill remaining NaN with 0
    corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all").fillna(0)

    # Clustermap
    try:
        g = sns.clustermap(corr, cmap="YlOrRd", figsize=(20, 18),
                           method="ward", linewidths=0, vmin=0, vmax=1)
        g.fig.suptitle("Feature Correlation Clustermap", fontsize=14, fontweight="bold", y=1.01)
        g.fig.savefig(OUT / "feature_clustermap.png", dpi=100)
        plt.close(g.fig)
    except Exception as e:
        print(f"    Clustermap skipped: {e}")

    # Find highly correlated feature pairs
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and val > 0.95:
                high_corr.append({"feature_1": idx, "feature_2": col, "correlation": val})

    high_corr_df = pd.DataFrame(high_corr).sort_values("correlation", ascending=False)
    high_corr_df.to_csv(OUT / "highly_correlated_pairs.csv", index=False)
    print(f"    Found {len(high_corr_df)} feature pairs with |r| > 0.95")
    print(f"    Saved feature_clustermap.png, highly_correlated_pairs.csv")


# ─────────────────────────────────────────────────────────────
# 7. Intensity Rate of Change Analysis
# ─────────────────────────────────────────────────────────────
def intensity_change_analysis(df):
    print("\n[7] Intensity Change Analysis")

    df_delta = df.dropna(subset=["wind_change"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Wind change distribution
    axes[0, 0].hist(df_delta["wind_change"], bins=30, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Wind Change Distribution (Δ kt per step)")
    axes[0, 0].set_xlabel("Δ Wind (kt)")
    axes[0, 0].axvline(0, color="red", linestyle="--")

    # Rapid intensification events
    ri_threshold = 30  # 30 kt increase in 24h is typical RI threshold
    ri_events = df_delta[df_delta["wind_change"] >= ri_threshold]
    rw_events = df_delta[df_delta["wind_change"] <= -ri_threshold]

    axes[0, 1].bar(["Rapid Intensification\n(>=30kt)", "Rapid Weakening\n(<=-30kt)", "Normal"],
                   [len(ri_events), len(rw_events), len(df_delta) - len(ri_events) - len(rw_events)],
                   color=["red", "blue", "gray"])
    axes[0, 1].set_title("Intensity Change Categories")
    axes[0, 1].set_ylabel("Count")

    # Wind change vs sun altitude
    if "sun_altitude_deg" in df_delta.columns:
        axes[1, 0].scatter(df_delta["sun_altitude_deg"], df_delta["wind_change"],
                          alpha=0.4, s=10, color="steelblue")
        axes[1, 0].set_xlabel("Sun Altitude (°)")
        axes[1, 0].set_ylabel("Δ Wind (kt)")
        axes[1, 0].set_title("Wind Change vs Sun Altitude")
        axes[1, 0].axhline(0, color="red", linestyle="--", alpha=0.5)

    # Wind change vs moon phase proxy (moon ecliptic lon)
    if "moon_ecliptic_lon_deg" in df_delta.columns:
        axes[1, 1].scatter(df_delta["moon_ecliptic_lon_deg"], df_delta["wind_change"],
                          alpha=0.4, s=10, color="mediumpurple")
        axes[1, 1].set_xlabel("Moon Ecliptic Longitude (°)")
        axes[1, 1].set_ylabel("Δ Wind (kt)")
        axes[1, 1].set_title("Wind Change vs Moon Position")
        axes[1, 1].axhline(0, color="red", linestyle="--", alpha=0.5)

    fig.suptitle("Storm Intensity Change Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "intensity_change.png", dpi=150)
    plt.close(fig)

    print(f"    RI events (>=30kt change): {len(ri_events)}")
    print(f"    RW events (<=-30kt change): {len(rw_events)}")
    print(f"    Saved intensity_change.png")


# ─────────────────────────────────────────────────────────────
# 8. Planetary Aspect Analysis
# ─────────────────────────────────────────────────────────────
def planetary_aspect_analysis(df):
    print("\n[8] Planetary Aspect / Angular Separation Analysis")

    # Compute angular separations between planet pairs using ecliptic longitude
    planets = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn"]
    lon_cols = {p: f"{p}_ecliptic_lon_deg" for p in planets}

    aspect_cols = []
    for i, p1 in enumerate(planets):
        for p2 in planets[i + 1:]:
            col1, col2 = lon_cols[p1], lon_cols[p2]
            if col1 in df.columns and col2 in df.columns:
                new_col = f"aspect_{p1}_{p2}"
                # Angular separation (0-180°)
                diff = (df[col1] - df[col2]).abs()
                df[new_col] = diff.apply(lambda x: min(x, 360 - x) if pd.notna(x) else np.nan)
                aspect_cols.append(new_col)

    if aspect_cols:
        # Correlation of aspects with wind
        aspect_corr = df[aspect_cols + ["wind"]].corr()["wind"].drop("wind").sort_values(ascending=False)
        aspect_corr.to_csv(OUT / "aspect_correlations.csv")

        fig, ax = plt.subplots(figsize=(12, 8))
        aspect_corr.plot.barh(ax=ax, color=["steelblue" if v > 0 else "coral" for v in aspect_corr.values])
        ax.set_title("Planetary Aspect Correlations with Wind Speed")
        ax.set_xlabel("Correlation")
        ax.axvline(0, color="black", linestyle="-")
        fig.tight_layout()
        fig.savefig(OUT / "aspect_correlations.png", dpi=150)
        plt.close(fig)

        print(f"    Computed {len(aspect_cols)} aspect angles")
        print(f"    Saved aspect_correlations.csv, aspect_correlations.png")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ADVANCED EDA — 2022-2026 Storm Data")
    print("=" * 60)

    df = load()

    pca_analysis(df)
    feature_importance(df)
    mutual_information_analysis(df)
    clustering_analysis(df)
    vedic_intensity_analysis(df)
    feature_correlation_clustering(df)
    intensity_change_analysis(df)
    planetary_aspect_analysis(df)

    print("\n" + "=" * 60)
    print(f"  Advanced EDA complete. All outputs in: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
