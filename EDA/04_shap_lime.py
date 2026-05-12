"""
04_shap_lime.py
===============
SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations)
analysis for storm wind speed prediction using astronomical/Vedic features.
All outputs saved to EDA/output/shap_lime/
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
import shap
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb

OUT = Path(__file__).resolve().parent / "output" / "shap_lime"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(__file__).resolve().parent / "output" / "storm_data_all.parquet"

sns.set_theme(style="whitegrid", font_scale=1.0)


def load():
    df = pd.read_parquet(DATA)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_features(df):
    """Get numeric feature columns, excluding targets and leakage."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["track_num", "storm_max_wind", "storm_forecast",
               "wind", "pressure", "wind_change", "pressure_change",
               "movement_speed", "displacement", "lon_change", "lat_change", "hours_elapsed"]
    features = [c for c in num_cols if c not in exclude and df[c].notna().sum() > len(df) * 0.8]
    return features


def prepare_data(df, features, target="wind"):
    """Prepare clean X, y arrays."""
    df_clean = df[features + [target]].dropna()
    X = df_clean[features].values
    y = df_clean[target].values
    return X, y, df_clean


# ─────────────────────────────────────────────────────────────
# 1. SHAP Analysis with XGBoost
# ─────────────────────────────────────────────────────────────
def shap_analysis(df, features):
    print("\n[1] SHAP Analysis")

    X, y, df_clean = prepare_data(df, features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                              random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"    XGBoost R² — Train: {train_score:.4f}, Test: {test_score:.4f}")
    # NOTE ON R² DISCREPANCY WITH 03_advanced_eda.py:
    # This script may report a much higher test R² (e.g. 0.75) vs the cross-validated
    # R² in 03_advanced_eda.py (which can be negative). The reasons are:
    #   1. FEATURE SUBSET: This script filters features to >80% completeness, keeping
    #      only the most data-rich columns. 03_advanced_eda uses a broader feature set
    #      (>50% completeness), introducing many sparse/noisy features.
    #   2. SPLIT METHOD: This uses a random 80/20 split, which can place observations
    #      from the same storm in both train and test — inflating R² via autocorrelation.
    #      03_advanced_eda uses GroupKFold (grouped by storm_id), giving a stricter
    #      out-of-storm generalization estimate.
    #   3. IMPLICATION: The high R² here is partly an artifact of within-storm
    #      autocorrelation (consecutive track observations for the same storm are highly
    #      correlated). A model that predicts "current wind ≈ next wind" would score
    #      well without learning any causal structure. The GroupKFold R² is the more
    #      honest measure of generalization to unseen storms.

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # --- SHAP Summary Plot (beeswarm) ---
    fig, ax = plt.subplots(figsize=(14, 12))
    shap.summary_plot(shap_values, X_test, feature_names=features,
                      show=False, max_display=30)
    plt.title("SHAP Summary — Top 30 Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # --- SHAP Bar Plot (mean |SHAP|) ---
    fig, ax = plt.subplots(figsize=(14, 12))
    shap.summary_plot(shap_values, X_test, feature_names=features,
                      plot_type="bar", show=False, max_display=30)
    plt.title("SHAP Mean |Value| — Top 30 Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # --- SHAP values as DataFrame ---
    shap_df = pd.DataFrame(shap_values, columns=features)
    shap_importance = shap_df.abs().mean().sort_values(ascending=False)
    shap_importance.to_csv(OUT / "shap_importance.csv")

    # --- Top 25 / Bottom 25 ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    top25 = shap_importance.head(25)
    axes[0].barh(range(len(top25)), top25.values, color="steelblue")
    axes[0].set_yticks(range(len(top25)))
    axes[0].set_yticklabels(top25.index, fontsize=8)
    axes[0].set_title("SHAP — Top 25 IMPORTANT Features")
    axes[0].set_xlabel("Mean |SHAP value|")
    axes[0].invert_yaxis()

    bottom25 = shap_importance.tail(25)
    axes[1].barh(range(len(bottom25)), bottom25.values, color="lightcoral")
    axes[1].set_yticks(range(len(bottom25)))
    axes[1].set_yticklabels(bottom25.index, fontsize=8)
    axes[1].set_title("SHAP — Bottom 25 UNIMPORTANT Features")
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].invert_yaxis()

    fig.suptitle("SHAP: Important vs Unimportant Variables", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "shap_important_vs_unimportant.png", dpi=150)
    plt.close(fig)

    # --- SHAP Dependence Plots for top 6 features ---
    top6 = shap_importance.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    for i, feat in enumerate(top6):
        feat_idx = features.index(feat)
        ax = axes_flat[i]
        ax.scatter(X_test[:, feat_idx], shap_values[:, feat_idx],
                  alpha=0.5, s=10, color="steelblue")
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"Dependence: {feat}", fontsize=9)
        ax.axhline(0, color="red", linestyle="--", alpha=0.3)
    fig.suptitle("SHAP Dependence Plots — Top 6 Features", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "shap_dependence_top6.png", dpi=150)
    plt.close(fig)

    # --- Force plot for specific instances (save as text summary) ---
    with open(OUT / "shap_instance_explanations.txt", "w") as f:
        f.write("SHAP Instance Explanations\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"XGBoost R² — Train: {train_score:.4f}, Test: {test_score:.4f}\n")
        f.write(f"Expected value (base): {explainer.expected_value:.2f}\n\n")
        f.write("IMPORTANT CAVEAT — R² INTERPRETATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"The test R²={test_score:.4f} here reflects a RANDOM 80/20 split.\n")
        f.write("This splits within storms, so training and test observations from\n")
        f.write("the same storm can appear in both sets. Consecutive track points\n")
        f.write("for a storm are highly autocorrelated — this inflates R² artificially.\n")
        f.write("03_advanced_eda.py uses GroupKFold (by storm_id) which prevents this\n")
        f.write("and gives a more honest out-of-storm generalization estimate.\n")
        f.write("SHAP importance rankings here reflect which features the model learned\n")
        f.write("to exploit, but do not confirm causal relationships.\n")
        f.write("For production use, average SHAP values across multiple random seeds\n")
        f.write("to assess stability of importance rankings.\n")
        f.write("-" * 40 + "\n\n")

        # Explain 5 diverse instances
        indices = [0, len(X_test) // 4, len(X_test) // 2, 3 * len(X_test) // 4, len(X_test) - 1]
        for idx in indices:
            if idx < 0 or idx >= len(X_test):
                continue
            actual = y_test[idx]
            predicted = model.predict(X_test[idx:idx + 1])[0]
            sv = shap_values[idx]
            top_pos = np.argsort(sv)[-5:][::-1]
            top_neg = np.argsort(sv)[:5]

            f.write(f"Instance {idx}: Actual={actual:.1f}, Predicted={predicted:.1f}\n")
            f.write(f"  Top 5 POSITIVE contributions:\n")
            for j in top_pos:
                f.write(f"    {features[j]}: SHAP={sv[j]:+.2f} (value={X_test[idx, j]:.4f})\n")
            f.write(f"  Top 5 NEGATIVE contributions:\n")
            for j in top_neg:
                f.write(f"    {features[j]}: SHAP={sv[j]:+.2f} (value={X_test[idx, j]:.4f})\n")
            f.write("\n")

    print(f"    Saved SHAP outputs: summary, bar, dependence, importance CSV")
    return model, X_train, X_test, y_train, y_test, features


# ─────────────────────────────────────────────────────────────
# 2. SHAP Interaction Analysis
# ─────────────────────────────────────────────────────────────
def shap_interaction(model, X_test, features):
    print("\n[2] SHAP Interaction Analysis")

    # Use a subsample for interaction (expensive)
    n_sample = min(50, len(X_test))
    X_sample = X_test[:n_sample]

    explainer = shap.TreeExplainer(model)
    try:
        shap_interaction_values = explainer.shap_interaction_values(X_sample)

        # Mean absolute interaction
        mean_interaction = np.abs(shap_interaction_values).mean(axis=0)

        # Top interaction pairs
        n_feat = mean_interaction.shape[0]
        interactions = []
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                interactions.append({
                    "feature_1": features[i],
                    "feature_2": features[j],
                    "mean_interaction": mean_interaction[i, j]
                })

        int_df = pd.DataFrame(interactions).sort_values("mean_interaction", ascending=False)
        int_df.head(50).to_csv(OUT / "shap_top_interactions.csv", index=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        top20 = int_df.head(20)
        labels = [f"{r['feature_1']}\n× {r['feature_2']}" for _, r in top20.iterrows()]
        ax.barh(range(len(top20)), top20["mean_interaction"].values, color="mediumpurple")
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title("Top 20 SHAP Feature Interactions")
        ax.set_xlabel("Mean |Interaction|")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(OUT / "shap_interactions_top20.png", dpi=150)
        plt.close(fig)

        print(f"    Saved shap_top_interactions.csv, shap_interactions_top20.png")
    except Exception as e:
        print(f"    Interaction analysis failed: {e}")


# ─────────────────────────────────────────────────────────────
# 3. LIME Analysis
# ─────────────────────────────────────────────────────────────
def lime_analysis(df, features, model, X_train, X_test, y_test):
    print("\n[3] LIME Analysis")

    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, feature_names=features, mode="regression",
        verbose=False, random_state=42
    )

    # Explain multiple instances across intensity spectrum
    wind_vals = y_test
    quartiles = np.percentile(wind_vals, [10, 25, 50, 75, 90])
    instance_indices = []
    for q in quartiles:
        idx = np.argmin(np.abs(wind_vals - q))
        instance_indices.append(idx)

    # Aggregate feature importance across all explanations
    lime_importances = {}
    all_explanations_text = []

    for i, idx in enumerate(instance_indices):
        exp = explainer.explain_instance(
            X_test[idx], model.predict, num_features=len(features),
            num_samples=1000
        )

        # Collect per-feature importance
        for feat_name, weight in exp.as_list():
            clean_name = feat_name
            # LIME uses conditions like "feature <= 0.5", extract feature name
            for f in features:
                if f in feat_name:
                    clean_name = f
                    break
            if clean_name not in lime_importances:
                lime_importances[clean_name] = []
            lime_importances[clean_name].append(abs(weight))

        # Save individual explanation — custom high-res plot
        exp_list = exp.as_list()
        n_feats = len(exp_list)
        predicted = model.predict(X_test[idx:idx + 1])[0]

        # Sort by absolute weight for display
        exp_sorted = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)
        labels_lime = [e[0] for e in exp_sorted]
        weights_lime = [e[1] for e in exp_sorted]
        colors_lime = ["#2ca02c" if w > 0 else "#d62728" for w in weights_lime]

        # Dynamic height: 0.35 inch per feature, min 10
        fig_h = max(10, n_feats * 0.35 + 3)
        fig, ax = plt.subplots(figsize=(16, fig_h))
        y_pos = range(len(labels_lime))
        ax.barh(y_pos, weights_lime, color=colors_lime, edgecolor="black", linewidth=0.3, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_lime, fontsize=8, family="monospace")
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("LIME Weight", fontsize=11)
        ax.set_title(
            f"LIME Explanation — Instance {idx}  |  "
            f"Actual Wind = {wind_vals[idx]:.0f} kt  |  "
            f"Predicted = {predicted:.1f} kt",
            fontsize=13, fontweight="bold", pad=15
        )
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"lime_instance_{i}_wind{wind_vals[idx]:.0f}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)

        # Text
        all_explanations_text.append(f"\nInstance {idx} (Wind={wind_vals[idx]:.0f} kt):")
        all_explanations_text.append(f"  Predicted: {model.predict(X_test[idx:idx+1])[0]:.1f}")
        all_explanations_text.append(f"  Intercept: {exp.intercept[0]:.2f}" if hasattr(exp, 'intercept') else "")
        for feat_name, weight in exp.as_list()[:10]:
            all_explanations_text.append(f"  {feat_name}: {weight:+.4f}")

    # Aggregate LIME importance
    lime_agg = pd.Series({k: np.mean(v) for k, v in lime_importances.items()}).sort_values(ascending=False)
    lime_agg.to_csv(OUT / "lime_importance_aggregated.csv")

    # Plot aggregate LIME importance
    fig, ax = plt.subplots(figsize=(14, 10))
    top30 = lime_agg.head(30)
    top30.plot.barh(ax=ax, color="seagreen")
    ax.set_title("LIME — Aggregated Feature Importance (Top 30)")
    ax.set_xlabel("Mean |LIME Weight|")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "lime_importance_aggregated.png", dpi=150)
    plt.close(fig)

    # Top 25 vs Bottom 25
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    top25 = lime_agg.head(25)
    axes[0].barh(range(len(top25)), top25.values, color="seagreen")
    axes[0].set_yticks(range(len(top25)))
    axes[0].set_yticklabels(top25.index, fontsize=8)
    axes[0].set_title("LIME — Top 25 IMPORTANT Features")
    axes[0].set_xlabel("Mean |LIME Weight|")
    axes[0].invert_yaxis()

    bottom25 = lime_agg.tail(25)
    axes[1].barh(range(len(bottom25)), bottom25.values, color="lightsalmon")
    axes[1].set_yticks(range(len(bottom25)))
    axes[1].set_yticklabels(bottom25.index, fontsize=8)
    axes[1].set_title("LIME — Bottom 25 UNIMPORTANT Features")
    axes[1].set_xlabel("Mean |LIME Weight|")
    axes[1].invert_yaxis()

    fig.suptitle("LIME: Important vs Unimportant Variables", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "lime_important_vs_unimportant.png", dpi=150)
    plt.close(fig)

    # Save text explanations
    with open(OUT / "lime_explanations.txt", "w") as f:
        f.write("LIME Instance Explanations\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(all_explanations_text))

    print(f"    Saved LIME outputs: {len(instance_indices)} instance plots, aggregated importance")


# ─────────────────────────────────────────────────────────────
# 4. SHAP vs LIME Comparison
# ─────────────────────────────────────────────────────────────
def compare_shap_lime():
    print("\n[4] SHAP vs LIME Comparison")

    shap_imp = pd.read_csv(OUT / "shap_importance.csv", index_col=0)
    shap_imp.columns = ["shap_importance"]
    lime_imp = pd.read_csv(OUT / "lime_importance_aggregated.csv", index_col=0)
    lime_imp.columns = ["lime_importance"]

    # Normalize both to [0, 1]
    shap_norm = shap_imp / shap_imp.max()
    lime_norm = lime_imp / lime_imp.max()

    combined = shap_norm.join(lime_norm, how="outer").fillna(0)
    combined["mean_rank"] = (combined.rank(ascending=False)).mean(axis=1)
    combined = combined.sort_values("mean_rank")
    combined.to_csv(OUT / "shap_vs_lime_comparison.csv")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(combined["shap_importance"], combined["lime_importance"],
              alpha=0.5, s=20, color="steelblue")

    # Label top features
    top_features = combined.head(15).index
    for feat in top_features:
        ax.annotate(feat, (combined.loc[feat, "shap_importance"],
                          combined.loc[feat, "lime_importance"]),
                   fontsize=6, alpha=0.8)

    ax.set_xlabel("Normalized SHAP Importance")
    ax.set_ylabel("Normalized LIME Importance")
    ax.set_title("SHAP vs LIME Feature Importance Comparison")
    ax.plot([0, 1], [0, 1], "r--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "shap_vs_lime_scatter.png", dpi=150)
    plt.close(fig)

    # Rank comparison bar
    fig, ax = plt.subplots(figsize=(14, 10))
    top20_combined = combined.head(20)
    x = range(len(top20_combined))
    width = 0.35
    ax.barh([xi - width / 2 for xi in x], top20_combined["shap_importance"].values,
            width, color="steelblue", label="SHAP")
    ax.barh([xi + width / 2 for xi in x], top20_combined["lime_importance"].values,
            width, color="seagreen", label="LIME")
    ax.set_yticks(list(x))
    ax.set_yticklabels(top20_combined.index, fontsize=8)
    ax.legend()
    ax.set_title("Top 20 Features — SHAP vs LIME")
    ax.set_xlabel("Normalized Importance")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "shap_vs_lime_top20.png", dpi=150)
    plt.close(fig)

    # Consensus: features ranked top by BOTH methods
    shap_top30 = set(shap_imp.nlargest(30, "shap_importance").index)
    lime_top30 = set(lime_imp.nlargest(30, "lime_importance").index)
    consensus = shap_top30 & lime_top30

    with open(OUT / "consensus_important_features.txt", "w", encoding="utf-8") as f:
        f.write("CONSENSUS: Features in Top 30 of BOTH SHAP and LIME\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"SHAP Top 30: {len(shap_top30)} features\n")
        f.write(f"LIME Top 30: {len(lime_top30)} features\n")
        f.write(f"Consensus:   {len(consensus)} features\n\n")
        for feat in sorted(consensus):
            f.write(f"  ✓ {feat}\n")

        # Features unique to each
        shap_only = shap_top30 - lime_top30
        lime_only = lime_top30 - shap_top30
        f.write(f"\nSHAP-only features ({len(shap_only)}):\n")
        for feat in sorted(shap_only):
            f.write(f"  {feat}\n")
        f.write(f"\nLIME-only features ({len(lime_only)}):\n")
        for feat in sorted(lime_only):
            f.write(f"  {feat}\n")

    print(f"    Consensus features (top 30 both): {len(consensus)}")
    print(f"    Saved shap_vs_lime_comparison.csv, scatter, top20, consensus")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SHAP & LIME Analysis — 2022-2026 Storm Data")
    print("=" * 60)

    df = load()
    features = get_features(df)
    print(f"Using {len(features)} numeric features")

    model, X_train, X_test, y_train, y_test, features = shap_analysis(df, features)
    shap_interaction(model, X_test, features)
    lime_analysis(df, features, model, X_train, X_test, y_test)
    compare_shap_lime()

    print("\n" + "=" * 60)
    print(f"  SHAP & LIME analysis complete. All outputs in: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
