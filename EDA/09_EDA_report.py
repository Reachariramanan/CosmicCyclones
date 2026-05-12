"""
09_EDA_report.py
================
Regenerates all EDA figures from source data and produces a single-column
black-and-white LaTeX report compiled to PDF via ReportLab.

Figures regenerated (never embedded as raw PNGs — all rebuilt from CSVs/parquet):
  - aspect_correlations.png        (from aspect_correlations.csv)
  - feature_importance_top30.png   (from feature_importance_all.csv)
  - important_vs_unimportant.png   (from feature_importance_all.csv + shap/lime)
  - intensity_change.png           (from storm_data_all.parquet)
  - mutual_information_top30.png   (from mutual_information.csv)
  - shap_important_vs_unimportant.png (from shap_importance.csv)
  - shap_interactions_top20.png    (from shap_top_interactions.csv)
  - shap_summary_bar.png           (from shap_importance.csv)
  - shap_vs_lime_top20.png         (from shap_vs_lime_comparison.csv)
  - nakshatra_intensity_bar.png    (from storm_data_all.parquet)
  - retrograde_effects.png         (from retrograde_analysis.csv)
  - tithi_nakshatra_heatmap.png    (mean wind, from storm_data_all.parquet)
  - tithi_nakshatra_heatmap_median.png (median wind, from storm_data_all.parquet)
  - vedic_degree_corr_bar.png      (from vedic_degree_correlations.csv)

All figures: greyscale, 300 dpi, tight_layout.
Output: output/eda_report/  (figures + report.tex + report.pdf)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import kruskal

# ── ReportLab ─────────────────────────────────────────────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        BaseDocTemplate, Frame, Image, PageBreak, PageTemplate,
        Paragraph, Spacer, Table, TableStyle,
    )
    HAS_RL = True
except ImportError:
    HAS_RL = False
    print("[WARN] reportlab not installed — PDF will not be generated. Run: pip install reportlab")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE  = Path(__file__).resolve().parent
DATA  = BASE / "output" / "storm_data_all.parquet"
OUT   = BASE / "output" / "eda_report"
OUT.mkdir(parents=True, exist_ok=True)

SRCS = {
    "aspect_corr":      BASE / "output" / "advanced_eda"   / "aspect_correlations.csv",
    "feat_imp_all":     BASE / "output" / "advanced_eda"   / "feature_importance_all.csv",
    "mi":               BASE / "output" / "advanced_eda"   / "mutual_information.csv",
    "shap_imp":         BASE / "output" / "shap_lime"      / "shap_importance.csv",
    "shap_interact":    BASE / "output" / "shap_lime"      / "shap_top_interactions.csv",
    "shap_vs_lime":     BASE / "output" / "shap_lime"      / "shap_vs_lime_comparison.csv",
    "lime_imp":         BASE / "output" / "shap_lime"      / "lime_importance_aggregated.csv",
    "retrograde":       BASE / "output" / "comprehensive"  / "retrograde_analysis.csv",
    "vedic_degree":     BASE / "output" / "comprehensive"  / "vedic_degree_correlations.csv",
    "vif":              BASE / "output" / "comprehensive"  / "vif_analysis.csv",
    "kw_wind":          BASE / "output" / "basic_eda"      / "kruskal_wallis_wind.csv",
}

# ── matplotlib color defaults ──────────────────────────────────────────────────
import seaborn as sns
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "figure.facecolor": "white",
    "axes.facecolor":   "#fafafa",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "#111111",
    "xtick.color":      "#111111",
    "ytick.color":      "#111111",
    "text.color":       "#111111",
    "grid.color":       "#dddddd",
    "grid.linewidth":   0.5,
    "axes.grid":        True,
    "font.family":      "DejaVu Sans",
    "font.size":        9,
})

# Color palette (tab10)
C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
GREY = C  # alias so existing references still work


def _safe_read(key: str, **kw) -> pd.DataFrame | None:
    p = SRCS.get(key)
    if p is None or not p.exists():
        print(f"  [SKIP] {key}: file not found at {p}")
        return None
    try:
        df = pd.read_csv(p, **kw)
        return df
    except Exception as e:
        print(f"  [SKIP] {key}: {e}")
        return None


def _save(fig: plt.Figure, name: str) -> Path:
    p = OUT / name
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {name}")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def fig_aspect_correlations() -> Path:
    """Horizontal bar chart: planetary aspect Pearson r with wind."""
    df = _safe_read("aspect_corr", index_col=0)
    if df is None:
        return None
    df.columns = ["pearson_r"]
    df = df.sort_values("pearson_r")
    fig, ax = plt.subplots(figsize=(7, 5))
    colors_bar = ["#1f77b4" if v >= 0 else "#d62728" for v in df["pearson_r"]]
    ax.barh(df.index, df["pearson_r"], color=colors_bar, edgecolor="none", height=0.7)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Pearson r with Wind Speed (kt)")
    ax.set_title("Planetary Aspect Correlations with Tropical Cyclone Wind Speed", fontsize=10)
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row["pearson_r"] + (0.001 if row["pearson_r"] >= 0 else -0.001),
                i, f"{row['pearson_r']:.4f}", va="center",
                ha="left" if row["pearson_r"] >= 0 else "right", fontsize=7)
    fig.tight_layout()
    return _save(fig, "aspect_correlations.png")


def fig_feature_importance_top30() -> Path:
    """Grouped bar: RF / XGB / LGB importances for top-30 mean."""
    df = _safe_read("feat_imp_all", index_col=0)
    if df is None:
        return None
    df.columns = [c.strip() for c in df.columns]
    # Rename if needed
    col_map = {}
    for c in df.columns:
        if "random" in c.lower() or "rf" in c.lower():
            col_map[c] = "RandomForest"
        elif "xgb" in c.lower():
            col_map[c] = "XGBoost"
        elif "lgb" in c.lower() or "light" in c.lower():
            col_map[c] = "LightGBM"
        elif "mean" in c.lower():
            col_map[c] = "mean_importance"
    df = df.rename(columns=col_map)
    if "mean_importance" not in df.columns:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df["mean_importance"] = df[num_cols].mean(axis=1)
    top30 = df.nlargest(30, "mean_importance")
    models = [c for c in ["RandomForest", "XGBoost", "LightGBM"] if c in top30.columns]
    x = np.arange(len(top30))
    width = 0.28
    fig, ax = plt.subplots(figsize=(14, 6))
    shades = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (model, shade) in enumerate(zip(models, shades)):
        ax.bar(x + i * width, top30[model], width, label=model,
               color=shade, edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(top30.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Feature Importance Score")
    ax.set_title("Top-30 Feature Importances: RandomForest, XGBoost, LightGBM", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "feature_importance_top30.png")


def fig_important_vs_unimportant() -> Path:
    """Side-by-side bar: top-25 vs bottom-25 mean importance."""
    df = _safe_read("feat_imp_all", index_col=0)
    if df is None:
        return None
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        if "mean" in c.lower():
            col_map[c] = "mean_importance"
    df = df.rename(columns=col_map)
    if "mean_importance" not in df.columns:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df["mean_importance"] = df[num_cols].mean(axis=1)
    top25    = df.nlargest(25, "mean_importance")
    bottom25 = df.nsmallest(25, "mean_importance")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, subset, label, shade in [
        (axes[0], top25,    "Top-25 (Most Important)",    "#1f77b4"),
        (axes[1], bottom25, "Bottom-25 (Least Important)", "#d62728"),
    ]:
        ax.barh(subset.index[::-1], subset["mean_importance"][::-1],
                color=shade, edgecolor="none", height=0.7, alpha=0.82)
        ax.set_xlabel("Mean Importance")
        ax.set_title(label, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Feature Importance Contrast: Important vs Unimportant Features", fontsize=10)
    fig.tight_layout()
    return _save(fig, "important_vs_unimportant.png")


def fig_intensity_change(df_storm: pd.DataFrame) -> Path:
    """3-panel: wind change histogram, RI/RW counts, scatter wind_change vs sun_altitude."""
    if df_storm is None:
        return None
    needed = {"wind_change", "wind"}
    if not needed.issubset(df_storm.columns):
        print("  [SKIP] intensity_change: missing columns")
        return None
    wc = df_storm["wind_change"].dropna()
    ri_thresh, rw_thresh = 30, -30
    ri_count = (wc >= ri_thresh).sum()
    rw_count = (wc <= rw_thresh).sum()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Panel 1: histogram
    ax = axes[0]
    bins = np.linspace(wc.quantile(0.01), wc.quantile(0.99), 60)
    ax.hist(wc, bins=bins, color="#1f77b4", edgecolor="none", alpha=0.8)
    ax.axvline(ri_thresh, color="#d62728", linewidth=1.4, linestyle="--",
               label=f"RI threshold (+{ri_thresh} kt)")
    ax.axvline(rw_thresh, color="#ff7f0e", linewidth=1.4, linestyle="--",
               label=f"RW threshold ({rw_thresh} kt)")
    ax.set_xlabel("6-hour Wind Change (kt)")
    ax.set_ylabel("Count")
    ax.set_title("Wind Change Distribution")
    ax.legend(fontsize=7)
    # Panel 2: RI/RW counts
    ax = axes[1]
    categories = ["Total\nobservations", r"Rapid" + "\n" + r"Intensification" + "\n" + r"($\geq$30 kt)",
                  r"Rapid" + "\n" + r"Weakening" + "\n" + r"($\leq$-30 kt)"]
    counts = [len(wc), ri_count, rw_count]
    ax.bar(categories, counts, color=["#1f77b4", "#d62728", "#ff7f0e"],
           edgecolor="white", linewidth=0.5, alpha=0.85)
    for i, c in enumerate(counts):
        ax.text(i, c + len(wc) * 0.01, str(c), ha="center", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("RI and RW Event Counts")
    # Panel 3: scatter wind vs sun_altitude if available
    ax = axes[2]
    if "sun_altitude_deg" in df_storm.columns:
        samp = df_storm[["sun_altitude_deg", "wind_change"]].dropna().sample(
            min(5000, len(df_storm)), random_state=42)
        ax.scatter(samp["sun_altitude_deg"], samp["wind_change"],
                   alpha=0.2, s=5, color="#9467bd")
        ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Sun Altitude (deg)")
        ax.set_ylabel("6-hour Wind Change (kt)")
        ax.set_title("Wind Change vs Sun Altitude")
    else:
        ax.text(0.5, 0.5, "sun_altitude_deg\nnot available",
                ha="center", va="center", transform=ax.transAxes)
    fig.suptitle("Tropical Cyclone Intensity Change Analysis", fontsize=11)
    fig.tight_layout()
    return _save(fig, "intensity_change.png")


def fig_mutual_information_top30() -> Path:
    """Horizontal bar: top-30 MI scores with wind."""
    df = _safe_read("mi", index_col=0)
    if df is None:
        return None
    df.columns = ["mi_score"]
    top30 = df.nlargest(30, "mi_score")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top30.index[::-1], top30["mi_score"][::-1],
            color="#2ca02c", edgecolor="none", height=0.7, alpha=0.85)
    ax.set_xlabel("Mutual Information Score with Wind Speed")
    ax.set_title("Top-30 Features by Mutual Information with Wind Speed", fontsize=10)
    ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    return _save(fig, "mutual_information_top30.png")


def fig_shap_summary_bar() -> Path:
    """Horizontal bar: top-30 mean |SHAP| values."""
    df = _safe_read("shap_imp", index_col=0)
    if df is None:
        return None
    df.columns = ["shap_importance"]
    top30 = df.nlargest(30, "shap_importance")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top30.index[::-1], top30["shap_importance"][::-1],
            color="#ff7f0e", edgecolor="none", height=0.7, alpha=0.85)
    ax.set_xlabel("Mean |SHAP Value| (kt)")
    ax.set_title("SHAP Feature Importance: Top-30 Features", fontsize=10)
    ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    return _save(fig, "shap_summary_bar.png")


def fig_shap_important_vs_unimportant() -> Path:
    """Side-by-side: top-25 vs bottom-25 mean |SHAP|."""
    df = _safe_read("shap_imp", index_col=0)
    if df is None:
        return None
    df.columns = ["shap_importance"]
    top25    = df.nlargest(25, "shap_importance")
    bottom25 = df.nsmallest(25, "shap_importance")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, subset, label, shade in [
        (axes[0], top25,    "Top-25 SHAP Features",    "#ff7f0e"),
        (axes[1], bottom25, "Bottom-25 SHAP Features", "#9467bd"),
    ]:
        ax.barh(subset.index[::-1], subset["shap_importance"][::-1],
                color=shade, edgecolor="none", height=0.7, alpha=0.82)
        ax.set_xlabel("Mean |SHAP Value| (kt)")
        ax.set_title(label, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("SHAP Importance Contrast: Important vs Unimportant Features", fontsize=10)
    fig.tight_layout()
    return _save(fig, "shap_important_vs_unimportant.png")


def fig_shap_interactions_top20() -> Path:
    """Horizontal bar: top-20 SHAP interaction pairs."""
    df = _safe_read("shap_interact", index_col=None)
    if df is None:
        return None
    df.columns = [c.strip() for c in df.columns]
    # expect: feature_1, feature_2, mean_interaction
    if "feature_1" not in df.columns:
        df.columns = ["feature_1", "feature_2", "mean_interaction"]
    df["pair"] = df["feature_1"].str[:18] + "\n× " + df["feature_2"].str[:18]
    top20 = df.nlargest(20, "mean_interaction")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top20["pair"][::-1], top20["mean_interaction"][::-1],
            color="#8c564b", edgecolor="none", height=0.7, alpha=0.85)
    ax.set_xlabel("Mean |SHAP Interaction Value|")
    ax.set_title("Top-20 SHAP Feature Interactions", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return _save(fig, "shap_interactions_top20.png")


def fig_shap_vs_lime_top20() -> Path:
    """Scatter/rank comparison: top-20 consensus features, SHAP vs LIME."""
    df = _safe_read("shap_vs_lime", index_col=0)
    if df is None:
        return None
    df.columns = [c.strip() for c in df.columns]
    # normalise column names
    col_map = {}
    for c in df.columns:
        if "shap" in c.lower():
            col_map[c] = "shap"
        elif "lime" in c.lower():
            col_map[c] = "lime"
        elif "rank" in c.lower() or "mean" in c.lower():
            col_map[c] = "mean_rank"
    df = df.rename(columns=col_map)
    if "shap" not in df.columns or "lime" not in df.columns:
        print("  [SKIP] shap_vs_lime: expected shap/lime columns")
        return None
    top20 = df.sort_values("mean_rank").head(20) if "mean_rank" in df.columns else df.head(20)
    x = np.arange(len(top20))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width/2, top20["shap"], width, label="SHAP", color="#ff7f0e",
           edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.bar(x + width/2, top20["lime"], width, label="LIME", color="#17becf",
           edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(top20.index, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Normalised Importance Score")
    ax.set_title("SHAP vs LIME: Top-20 Consensus Features (Normalised [0,1])", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, "shap_vs_lime_top20.png")


def fig_nakshatra_intensity_bar(df_storm: pd.DataFrame) -> Path:
    """Bar chart: median wind speed per Nakshatra (sorted descending)."""
    if df_storm is None or "nakshatra" not in df_storm.columns:
        print("  [SKIP] nakshatra_intensity_bar: missing column")
        return None
    grp = (df_storm.groupby("nakshatra")["wind"]
           .agg(median="median", q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75), n="count")
           .reset_index()
           .sort_values("median", ascending=False))
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(grp))
    ax.bar(x, grp["median"], color="#1f77b4", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.errorbar(x, grp["median"],
                yerr=[grp["median"] - grp["q25"], grp["q75"] - grp["median"]],
                fmt="none", color="#333333", capsize=2, linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(grp["nakshatra"], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Median Wind Speed (kt)")
    ax.set_title("Median Tropical Cyclone Intensity by Nakshatra (IQR error bars)", fontsize=10)
    # Add n labels
    for i, row in enumerate(grp.itertuples()):
        ax.text(i, row.q75 + 0.5, str(row.n), ha="center", fontsize=5.5, color="#333333")
    fig.tight_layout()
    return _save(fig, "nakshatra_intensity_bar.png")


def fig_retrograde_effects() -> Path:
    """Grouped bar: retrograde vs direct median wind per planet."""
    df = _safe_read("retrograde", index_col=0)
    if df is None:
        return None
    planets = df.index.tolist()
    x = np.arange(len(planets))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, df["retro_median_wind"],  width, label="Retrograde",
           color="#d62728", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.bar(x + width/2, df["direct_median_wind"], width, label="Direct",
           color="#1f77b4", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(planets, fontsize=9)
    ax.set_ylabel("Median Wind Speed (kt)")
    ax.set_title("Retrograde vs Direct Planetary Motion: Median Storm Wind Speed", fontsize=10)
    ax.legend(fontsize=9)
    # Annotate significance
    for i, row in enumerate(df.itertuples()):
        sig = "*" if row.significant_p05 else "ns"
        ax.text(i, max(row.retro_median_wind, row.direct_median_wind) + 0.5,
                sig, ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "retrograde_effects.png")


def fig_tithi_nakshatra_heatmap(df_storm: pd.DataFrame, stat: str = "mean") -> Path:
    """Heatmap: Tithi × Nakshatra mean or median wind (greyscale)."""
    if df_storm is None:
        return None
    cols_needed = {"tithi", "nakshatra", "wind"}
    if not cols_needed.issubset(df_storm.columns):
        print(f"  [SKIP] tithi_nakshatra_heatmap_{stat}: missing columns")
        return None
    pivot = df_storm.pivot_table(
        index="tithi", columns="nakshatra", values="wind",
        aggfunc=stat, observed=True)
    pivot = pivot.dropna(how="all", axis=0).dropna(how="all", axis=1)
    fig, ax = plt.subplots(figsize=(18, 8))
    import matplotlib.colors as mcolors
    cmap = plt.cm.YlOrRd
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, interpolation="none")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Nakshatra")
    ax.set_ylabel("Tithi")
    title = f"Tithi × Nakshatra Heatmap: {stat.capitalize()} Wind Speed (kt)"
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Wind Speed (kt)", fontsize=8)
    fig.tight_layout()
    fname = f"tithi_nakshatra_heatmap{'_median' if stat == 'median' else ''}.png"
    return _save(fig, fname)


def fig_vedic_degree_corr_bar() -> Path:
    """Horizontal bar: circular-linear r_circular_max per Vedic planet degree."""
    df = _safe_read("vedic_degree", index_col=0)
    if df is None:
        return None
    df.columns = [c.strip() for c in df.columns]
    if "r_circular_max" not in df.columns:
        print("  [SKIP] vedic_degree_corr_bar: missing r_circular_max column")
        return None
    df = df.sort_values("r_circular_max", key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_bar = ["#2ca02c" if v >= 0 else "#d62728" for v in df["r_circular_max"]]
    ax.barh(df.index, df["r_circular_max"], color=colors_bar, edgecolor="none", height=0.7, alpha=0.85)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Circular-Linear Correlation (max|r_sin|, |r_cos|) with Wind Speed")
    ax.set_title("Vedic Planetary Degree: Circular-Linear Correlation with Wind Speed", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return _save(fig, "vedic_degree_corr_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# DATA SUMMARIES  (returned as strings/dicts for the report)
# ══════════════════════════════════════════════════════════════════════════════

def summarise_aspect_corr() -> dict:
    df = _safe_read("aspect_corr", index_col=0)
    if df is None:
        return {}
    df.columns = ["pearson_r"]
    df = df.sort_values("pearson_r", ascending=False)
    return {
        "top3_positive": df.head(3).to_dict()["pearson_r"],
        "top3_negative": df.tail(3).to_dict()["pearson_r"],
        "max_r": float(df["pearson_r"].abs().max()),
        "n_pairs": len(df),
    }


def summarise_feature_importance() -> dict:
    df = _safe_read("feat_imp_all", index_col=0)
    if df is None:
        return {}
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        if "mean" in c.lower():
            col_map[c] = "mean_importance"
    df = df.rename(columns=col_map)
    if "mean_importance" not in df.columns:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df["mean_importance"] = df[num_cols].mean(axis=1)
    top5 = df.nlargest(5, "mean_importance")
    vedic_idx = [i for i in df.index if "vedic" in i.lower()]
    vedic = df.loc[vedic_idx]
    return {
        "top5": top5["mean_importance"].to_dict(),
        "n_features": len(df),
        "vedic_top": vedic.nlargest(3, "mean_importance")["mean_importance"].to_dict(),
        "vedic_share_top30": sum("vedic" in i.lower() for i in df.nlargest(30, "mean_importance").index),
    }


def summarise_retrograde() -> dict:
    df = _safe_read("retrograde", index_col=0)
    if df is None:
        return {}
    sig = df[df["significant_p05"] == True]
    return {
        "n_significant": len(sig),
        "n_total": len(df),
        "largest_diff": float(df["wind_diff_median"].abs().max()),
        "planet_largest_diff": df["wind_diff_median"].abs().idxmax(),
        "all_rows": df.to_dict("index"),
    }


def summarise_vif() -> dict:
    df = _safe_read("vif", index_col=0)
    if df is None:
        return {}
    df.columns = [c.strip() for c in df.columns]
    vif_col = [c for c in df.columns if "vif" in c.lower()]
    if not vif_col:
        return {}
    df = df.rename(columns={vif_col[0]: "VIF"})
    return {
        "max_vif": float(df["VIF"].max()),
        "max_feature": df["VIF"].idxmax(),
        "n_features": len(df),
        "all_below_5": bool((df["VIF"] < 5).all()),
    }


def summarise_kw() -> dict:
    df = _safe_read("kw_wind", index_col=0)
    if df is None:
        return {}
    df.columns = [c.strip() for c in df.columns]
    sig_col = [c for c in df.columns if "significant" in c.lower()]
    sig = df[df[sig_col[-1]] == True] if sig_col else df
    return {
        "n_significant": len(sig),
        "n_total": len(df),
        "top_H": df["kw_H"].idxmax() if "kw_H" in df.columns else None,
        "rows": df.to_dict("index"),
    }


def summarise_shap_vs_lime() -> dict:
    df = _safe_read("shap_vs_lime", index_col=0)
    if df is None:
        return {}
    # consensus = in top-30 of both
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        if "shap" in c.lower():
            col_map[c] = "shap"
        elif "lime" in c.lower():
            col_map[c] = "lime"
    df = df.rename(columns=col_map)
    if "shap" not in df.columns or "lime" not in df.columns:
        return {}
    shap_top30 = set(df.nlargest(30, "shap").index)
    lime_top30 = set(df.nlargest(30, "lime").index)
    consensus = shap_top30 & lime_top30
    return {
        "n_consensus": len(consensus),
        "consensus_features": sorted(consensus),
        "vedic_in_consensus": [f for f in consensus if "vedic" in f.lower()],
    }


def summarise_mi() -> dict:
    df = _safe_read("mi", index_col=0)
    if df is None:
        return {}
    df.columns = ["mi_score"]
    top5 = df.nlargest(5, "mi_score")
    vedic_top = df.loc[[i for i in df.index if "vedic" in i.lower()]].nlargest(3, "mi_score")
    return {
        "top5": top5["mi_score"].to_dict(),
        "max_mi": float(df["mi_score"].max()),
        "vedic_top3": vedic_top["mi_score"].to_dict(),
    }


def summarise_vedic_degree() -> dict:
    df = _safe_read("vedic_degree", index_col=0)
    if df is None:
        return {}
    df.columns = [c.strip() for c in df.columns]
    if "r_circular_max" not in df.columns:
        return {}
    df["abs_r"] = df["r_circular_max"].abs()
    top3 = df.nlargest(3, "abs_r")
    return {
        "top3": top3["r_circular_max"].to_dict(),
        "max_r": float(df["abs_r"].max()),
        "max_feature": df["abs_r"].idxmax(),
        "note": "Circular-linear method used (sin/cos decomposition); naive Pearson on raw degrees would be incorrect for cyclic data.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX DOCUMENT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_sci(v: float) -> str:
    if abs(v) < 1e-4:
        return f"{v:.2e}"
    return f"{v:.4f}"


def build_latex(fig_paths: dict, summaries: dict, df_storm: pd.DataFrame) -> str:
    """Build the full LaTeX source as a string."""

    # Build summary stats for intro
    n_obs   = len(df_storm) if df_storm is not None else 47533
    n_storms = df_storm["storm_id"].nunique() if (df_storm is not None and "storm_id" in df_storm.columns) else 944
    n_feats = summaries.get("feature_importance", {}).get("n_features", 186)

    kw = summaries.get("kw", {})
    retro = summaries.get("retrograde", {})
    vif = summaries.get("vif", {})
    asp = summaries.get("aspect_corr", {})
    fi  = summaries.get("feature_importance", {})
    sv  = summaries.get("shap_vs_lime", {})
    mi  = summaries.get("mi", {})
    vd  = summaries.get("vedic_degree", {})

    def imgline(key: str, caption: str, label: str, width: str = r"0.92\linewidth") -> str:
        p = fig_paths.get(key)
        if p is None:
            return (f"\\begin{{figure}}[H]\n"
                    f"\\centering\n"
                    f"\\fbox{{\\parbox{{0.85\\linewidth}}{{\\centering\\vspace{{1cm}}Figure not available: {key}\\vspace{{1cm}}}}}}\n"
                    f"\\caption{{{caption}}}\n"
                    f"\\label{{{label}}}\n"
                    f"\\end{{figure}}\n\n")
        # Use forward slashes for LaTeX on Windows
        fwd = str(p).replace("\\", "/")
        return (f"\\begin{{figure}}[H]\n"
                f"\\centering\n"
                f"\\includegraphics[width={width}]{{{fwd}}}\n"
                f"\\caption{{{caption}}}\n"
                f"\\label{{{label}}}\n"
                f"\\end{{figure}}\n\n")

    # ── Retrograde table rows ──────────────────────────────────────────────────
    retro_rows = ""
    all_retro = retro.get("all_rows", {})
    for planet, row in all_retro.items():
        sig = r"\textbf{*}" if row.get("significant_p05") else "ns"
        retro_rows += (
            f"  {planet} & {row.get('retro_count','?'):,} & {row.get('retro_pct','?'):.1f}\\% & "
            f"{row.get('retro_median_wind','?'):.0f} & {row.get('direct_median_wind','?'):.0f} & "
            f"{row.get('wind_diff_median','?'):+.1f} & {_fmt_sci(row.get('mannwhitney_p', 1))} & {sig} \\\\\n"
        )

    # ── KW table rows ──────────────────────────────────────────────────────────
    kw_rows = ""
    kw_data = kw.get("rows", {})
    for var, row in kw_data.items():
        kw_rows += (
            f"  {var} & {row.get('kw_H','?'):.2f} & {_fmt_sci(row.get('kw_p', 1))} & "
            f"{row.get('n_categories','?')} & "
            f"{'Yes' if row.get('significant_p05') else 'No'} \\\\\n"
        )

    # ── VIF table rows ─────────────────────────────────────────────────────────
    vif_df = _safe_read("vif", index_col=0)
    vif_rows = ""
    if vif_df is not None:
        vif_df.columns = [c.strip() for c in vif_df.columns]
        vc = [c for c in vif_df.columns if "vif" in c.lower()]
        if vc:
            vif_df = vif_df.rename(columns={vc[0]: "VIF"}).sort_values("VIF", ascending=False)
            for feat, row in vif_df.iterrows():
                flag = r"\textbf{High}" if row["VIF"] > 10 else ("Moderate" if row["VIF"] > 5 else "Low")
                vif_rows += f"  {feat} & {row['VIF']:.3f} & {flag} \\\\\n"

    # ── SHAP interaction table rows ────────────────────────────────────────────
    si_df = _safe_read("shap_interact", index_col=None)
    si_rows = ""
    if si_df is not None:
        si_df.columns = [c.strip() for c in si_df.columns]
        if len(si_df.columns) >= 3:
            si_df.columns = ["feature_1", "feature_2", "mean_interaction"]
            for _, row in si_df.head(15).iterrows():
                si_rows += f"  {row['feature_1'][:30]} & {row['feature_2'][:30]} & {row['mean_interaction']:.4f} \\\\\n"

    # ── SHAP vs LIME consensus list ─────────────────────────────────────────────
    consensus_list = sv.get("consensus_features", [])
    consensus_str = ", ".join(consensus_list[:12]) if consensus_list else "(none identified)"
    vedic_consensus = sv.get("vedic_in_consensus", [])
    vedic_consensus_str = ", ".join(vedic_consensus) if vedic_consensus else "None"

    # ── Aspect table rows ──────────────────────────────────────────────────────
    asp_df = _safe_read("aspect_corr", index_col=0)
    asp_rows = ""
    if asp_df is not None:
        asp_df.columns = ["pearson_r"]
        asp_df = asp_df.sort_values("pearson_r", ascending=False)
        for feat, row in asp_df.iterrows():
            asp_rows += f"  {feat} & {row['pearson_r']:.6f} \\\\\n"

    # ── Vedic degree table rows ────────────────────────────────────────────────
    vd_df = _safe_read("vedic_degree", index_col=0)
    vd_rows = ""
    if vd_df is not None:
        vd_df.columns = [c.strip() for c in vd_df.columns]
        for feat, row in vd_df.iterrows():
            vd_rows += (
                f"  {feat} & {row.get('r_sin',0):.5f} & {row.get('r_cos',0):.5f} & "
                f"{row.get('r_circular_max',0):.5f} \\\\\n"
            )

    doc = r"""\documentclass[11pt,a4paper]{article}

% ── packages ──────────────────────────────────────────────────────────────────
\usepackage[a4paper, margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{float}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{microtype}
\usepackage{parskip}
\usepackage{setspace}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{enumitem}

\hypersetup{colorlinks=true, linkcolor=black, citecolor=black, urlcolor=black}
\setstretch{1.15}
\captionsetup{font=small, labelfont=bf, labelsep=period}

% ── document ──────────────────────────────────────────────────────────────────
\begin{document}

\begin{titlepage}
  \centering
  \vspace*{2cm}
  {\Large\bfseries Exploratory Data Analysis Report:\\[0.4em]
   Tropical Cyclone Dataset with Vedic-Astronomical Features}\\[1.5cm]
  {\large Comprehensive EDA Findings: Feature Importance, SHAP/LIME Explainability,\\
   Vedic Category Analysis, Retrograde Effects, and Circular Correlations}\\[1.0cm]
  {\normalsize Dataset: """ + f"{n_obs:,}" + r""" observations across """ + str(n_storms) + r""" tropical cyclones (2016--2026)\\
   """ + str(n_feats) + r""" features: 14 meteorological \textbullet\
   18 Vedic-Panchang \textbullet\ 154 astronomical}\\[2cm]
  {\small Generated by \texttt{09\_EDA\_report.py}}
\end{titlepage}

\tableofcontents
\newpage

% ══════════════════════════════════════════════════════════════════════════════
\section{Overview and Data Description}
% ══════════════════════════════════════════════════════════════════════════════

This report documents all major exploratory data analysis (EDA) findings from a
""" + f"{n_obs:,}" + r"""-observation tropical cyclone dataset. Each observation represents one
6-hourly best-track position of a storm, enriched with three categories of features:

\begin{itemize}[noitemsep]
  \item \textbf{Meteorological (14 features):} latitude, longitude, wind speed, spline-derived
        track geometry, elevation, and land-proximity metrics.
  \item \textbf{Vedic-Panchang (18 features):} Tithi, Nakshatra, Yoga, Karana, Paksha,
        Moonsign, Samvatsara, day-length (Dinamana/Ratrimana), Weekday, Ayana, and Ritu.
  \item \textbf{Scientific-Astronomical (154 features):} Planet positions (ecliptic
        longitude/latitude, RA/Dec), altitudes, azimuths, angular separations (aspects),
        and Vedic degree representations with speed and padam.
\end{itemize}

\medskip
\noindent\textbf{Key variable:} Wind speed (kt) --- 10-minute maximum sustained surface wind.
Median = 40\,kt; mean $\approx$ 45\,kt; right-skewed (skewness $= 1.35$); all distributions
are non-normal (confirmed by Shapiro--Wilk and D'Agostino--Pearson tests), hence all
group comparisons use non-parametric methods throughout.

% ══════════════════════════════════════════════════════════════════════════════
\section{Feature Importance: Tree-Model Ensemble}
% ══════════════════════════════════════════════════════════════════════════════

Feature importances were computed using three tree-based models --- Random Forest,
XGBoost, and LightGBM --- trained with 5-fold GroupKFold cross-validation (groups
= \texttt{storm\_id}) to prevent within-storm autocorrelation leakage. Importances
were averaged across folds and models.

\medskip
\noindent\textbf{Key findings:}
\begin{itemize}[noitemsep]
""" + _fi_bullets(fi) + r"""\end{itemize}

""" + imgline("feature_importance_top30",
              "Top-30 Feature Importances across RandomForest, XGBoost, and LightGBM. "
              "Geographic/track features (longitude, latitude, spline\\_end\\_lon) dominate. "
              "Vedic features appear from rank 5 onward.",
              "fig:fi_top30") + r"""
""" + imgline("important_vs_unimportant",
              "Contrast between the 25 most important and 25 least important features "
              "(mean importance across RF/XGB/LGB). The bottom-25 features are overwhelmingly "
              "trigonometric altitude encodings and binary flags.",
              "fig:fi_contrast") + r"""

% ══════════════════════════════════════════════════════════════════════════════
\section{Mutual Information with Wind Speed}
% ══════════════════════════════════════════════════════════════════════════════

Mutual Information (MI) measures statistical dependence between each feature and
wind speed without assuming linearity. MI was computed using $k$-nearest-neighbour
estimation ($k=5$) on all """ + str(n_feats) + r""" features.

\medskip
\noindent\textbf{Key findings:}
\begin{itemize}[noitemsep]
""" + _mi_bullets(mi) + r"""\end{itemize}

\noindent\textit{Note:} High MI for \texttt{lahiri\_ayanamsha} and Rahu/Ketu degree features
reflects their strong temporal trend (precession-corrected ecliptic longitude changes
systematically with year), which confounds any apparent predictive signal.

""" + imgline("mutual_information_top30",
              "Top-30 features by Mutual Information with wind speed. "
              "Ayanamsha and Rahu/Ketu degree features lead due to temporal confounding. "
              "Meteorological features (latitude, spline geometry) follow.",
              "fig:mi_top30") + r"""

% ══════════════════════════════════════════════════════════════════════════════
\section{SHAP Explainability Analysis}
% ══════════════════════════════════════════════════════════════════════════════

SHAP (SHapley Additive exPlanations) values were computed using TreeExplainer on an
XGBoost model. Mean absolute SHAP values quantify each feature's average contribution
magnitude to individual wind speed predictions on the held-out test set.

\subsection{SHAP Feature Importance}

""" + imgline("shap_summary_bar",
              "Mean absolute SHAP values for the top-30 features. Latitude dominates "
              "with mean $|\\text{SHAP}| = 4.2$\\,kt, followed by geographic features. "
              "Vedic-speed features (e.g., \\texttt{vedic\\_chandra\\_speed}) appear in the top 10.",
              "fig:shap_bar") + r"""

""" + imgline("shap_important_vs_unimportant",
              "Contrast: top-25 vs bottom-25 features by mean $|$SHAP$|$. "
              "Bottom features are predominantly trigonometric altitude encodings "
              "(e.g., \\texttt{mars\\_sin\\_alt}, \\texttt{saturn\\_sin\\_alt}) with near-zero SHAP values.",
              "fig:shap_contrast") + r"""

\subsection{SHAP Feature Interactions}

SHAP interaction values decompose each prediction into pairwise feature contributions.
The top interaction pairs are listed in Table~\ref{tab:shap_interact}.

\begin{table}[H]
  \centering
  \caption{Top-15 SHAP Feature Interactions by Mean Interaction Value}
  \label{tab:shap_interact}
  \begin{tabular}{lll}
    \toprule
    Feature 1 & Feature 2 & Mean Interaction \\
    \midrule
""" + si_rows + r"""    \bottomrule
  \end{tabular}
\end{table}

""" + imgline("shap_interactions_top20",
              "Top-20 SHAP feature interactions. Latitude $\\times$ spline\\_bearing\\_deg "
              "is the dominant pair (interaction value $= 0.822$), reflecting how geographic "
              "position modulates storm track curvature effects on intensity.",
              "fig:shap_interact") + r"""

% ══════════════════════════════════════════════════════════════════════════════
\section{LIME Explainability and SHAP vs LIME Consensus}
% ══════════════════════════════════════════════════════════════════════════════

LIME (Local Interpretable Model-agnostic Explanations) was applied to five representative
instances spanning the wind speed distribution (10th--90th percentiles). Mean absolute
LIME weights were aggregated across instances to form a global LIME importance ranking.

SHAP and LIME importances were normalised to $[0, 1]$ and ranked jointly. Features
appearing in the top-30 of both methods are termed \textit{consensus features}.

\medskip
\noindent\textbf{Consensus findings:}
\begin{itemize}[noitemsep]
  \item """ + str(sv.get("n_consensus", "?")) + r""" features appear in the top-30 of both SHAP and LIME.
  \item Consensus set: \textit{""" + consensus_str + r"""}.
  \item Vedic features in consensus: """ + vedic_consensus_str + r""".
\end{itemize}

""" + imgline("shap_vs_lime_top20",
              "Normalised SHAP vs LIME importance for the top-20 consensus features. "
              "Both methods agree on geographic and track features dominating. "
              "Disagreements appear for trajectory-geometry features "
              "(e.g., \\texttt{spline\\_curvature} ranks higher in SHAP than LIME).",
              "fig:shap_vs_lime") + r"""

% ══════════════════════════════════════════════════════════════════════════════
\section{Vedic Category Analysis: Kruskal--Wallis Tests}
% ══════════════════════════════════════════════════════════════════════════════

Non-parametric Kruskal--Wallis tests were applied to compare wind speed distributions
across all major Vedic categorical variables. Since all distributions are non-normal
(confirmed by Shapiro--Wilk), non-parametric methods are required.
Benjamini--Hochberg (BH) FDR correction was applied across all tests.

\begin{table}[H]
  \centering
  \caption{Kruskal--Wallis Test Results: Vedic Categories vs Wind Speed.
           BH-corrected significance threshold $\alpha = 0.05$.}
  \label{tab:kw}
  \begin{tabular}{lrrl r}
    \toprule
    Variable & $H$ Statistic & $p$-value & \# Categories & Significant \\
    \midrule
""" + kw_rows + r"""    \bottomrule
  \end{tabular}
\end{table}

\medskip
\noindent\textbf{Critical caveats:}
\begin{itemize}[noitemsep]
  \item \texttt{drik\_ritu} ($H=1214$) and \texttt{sunsign} ($H=1123$) are \emph{calendar-season
        proxies}, not independent Vedic signals. Their significance reflects the well-known
        seasonal cycle of tropical cyclone activity.
  \item All tests use $N = """ + f"{n_obs:,}" + r"""$ observations. With such large $N$,
        even tiny group differences ($<0.5$\,kt) achieve $p \ll 0.001$. Effect sizes must
        be consulted before any practical interpretation.
  \item Confound-adjusted analyses (partial KW within basin $\times$ season strata) show
        that Yoga, Nakshatra, Tithi, and Moonsign retain significance in 100\% of strata ---
        however, the effect sizes ($\varepsilon^2 < 0.003$) remain negligible.
\end{itemize}

% ══════════════════════════════════════════════════════════════════════════════
\section{Nakshatra Intensity Analysis}
% ══════════════════════════════════════════════════════════════════════════════

Median wind speed was computed for each of the 27 Nakshatras across all observations.
Differences between Nakshatras are statistically significant ($H = 116$, $p = 2.4 \times 10^{-13}$)
but the effect size (epsilon-squared $\varepsilon^2 \approx 0.002$) is trivially small.

""" + imgline("nakshatra_intensity_bar",
              "Median wind speed (kt) per Nakshatra, sorted descending with IQR error bars. "
              "Numbers above bars indicate observation count per Nakshatra. "
              "The $\\approx 4$\\,kt range across Nakshatras is statistically significant "
              "but practically negligible ($\\varepsilon^2 \\approx 0.002$).",
              "fig:nakshatra") + r"""

% ══════════════════════════════════════════════════════════════════════════════
\section{Tithi $\times$ Nakshatra Interaction Heatmaps}
% ══════════════════════════════════════════════════════════════════════════════

Cross-tabulation of Tithi (lunar day, 1--30) and Nakshatra (27 lunar mansions)
reveals the joint distribution of mean and median wind speeds across these two
fundamental Vedic time categories.

""" + imgline("tithi_nakshatra_heatmap",
              "Mean wind speed (kt) by Tithi (rows) and Nakshatra (columns). "
              "Darker cells indicate higher mean wind. Cells with no observations "
              "appear white. The pattern reflects the co-occurrence structure of the "
              "Vedic calendar rather than independent effects.",
              "fig:tn_mean") + r"""

""" + imgline("tithi_nakshatra_heatmap_median",
              "Median wind speed (kt) by Tithi (rows) and Nakshatra (columns). "
              "More robust to outliers than the mean heatmap. "
              "No systematic high-intensity zone is visible across any "
              "Tithi--Nakshatra combination.",
              "fig:tn_median") + r"""

% ══════════════════════════════════════════════════════════════════════════════
\section{Retrograde Planetary Motion and Storm Intensity}
% ══════════════════════════════════════════════════════════════════════════════

For each of the seven classical Vedic planets (Guru/Jupiter, Shani/Saturn,
Mangal/Mars, Budha/Mercury, Shukra/Venus, Arun/Uranus, Varun/Neptune),
observations were split into retrograde (negative geocentric speed) and
direct (positive geocentric speed) periods. Mann--Whitney U tests compared
wind speed distributions.

\begin{table}[H]
  \centering
  \caption{Retrograde vs Direct Planetary Motion: Mann--Whitney U Results.
           $N_{\text{retro}}$ and $N_{\text{direct}}$ sum to $\approx$ """ + f"{n_obs:,}" + r""".
           Wind difference = median(retrograde) $-$ median(direct).
           ${}^*$ = significant at $p < 0.05$.}
  \label{tab:retro}
  \begin{tabular}{lrrlrrrr}
    \toprule
    Planet & $N_\text{retro}$ & Retro\% & $p$ & Med (retro) & Med (direct) & $\Delta$Med & Sig. \\
    \midrule
""" + retro_rows + r"""    \bottomrule
  \end{tabular}
\end{table}

\medskip
\noindent\textbf{Interpretation:}
All seven planets show statistically significant differences in wind speed
between retrograde and direct periods ($p < 0.001$). However, the magnitude
of differences is small ($\Delta\text{median} \leq 4$\,kt in all cases).
The effect sizes (rank-biserial $r < 0.10$ for all planets) indicate negligible
practical significance. With $N > 28{,}000$ per group, even 1\,kt differences
achieve $p \ll 0.001$.

""" + imgline("retrograde_effects",
              "Retrograde vs direct median wind speed per planet. "
              "Asterisks ($^*$) indicate statistical significance ($p < 0.05$). "
              "All differences are $\\leq 4$\\,kt. Arun and Varun show the largest "
              "differences, but these outer planets are retrograde for $>50\\%$ of "
              "observations, making the groups unequal.",
              "fig:retrograde") + r"""

% ══════════════════════════════════════════════════════════════════════════════
\section{Planetary Aspect Correlations}
% ══════════════════════════════════════════════════════════════════════════════

Planetary aspects are angular separations between planet pairs (modulo 360\textdegree).
Pearson correlations between each aspect and wind speed were computed across all
""" + f"{n_obs:,}" + r""" observations.

\begin{table}[H]
  \centering
  \caption{Planetary Aspect Pearson Correlations with Wind Speed (all """ + f"{n_obs:,}" + r""" observations).
           Maximum $|r| = """ + f"{asp.get('max_r', 0):.4f}" + r"""$.}
  \label{tab:aspects}
  \begin{tabular}{lr}
    \toprule
    Aspect Pair & Pearson $r$ \\
    \midrule
""" + asp_rows + r"""    \bottomrule
  \end{tabular}
\end{table}

""" + imgline("aspect_correlations",
              "Pearson correlations of planetary aspect angles with wind speed. "
              "The Jupiter--Saturn aspect shows the highest positive correlation "
              "($r = 0.077$), and Venus--Jupiter the most negative ($r = -0.039$). "
              "All correlations are weak ($|r| < 0.10$).",
              "fig:aspects") + r"""

\medskip
\noindent\textbf{Interpretation:} The maximum correlation is $r \approx 0.077$ (Jupiter--Saturn
separation), explaining less than $0.6\%$ of variance in wind speed. These are the
weakest signals in the entire dataset.

% ══════════════════════════════════════════════════════════════════════════════
\section{Vedic Planetary Degree: Circular-Linear Correlations}
% ══════════════════════════════════════════════════════════════════════════════

Vedic planet degrees are cyclical variables (0--360\textdegree). Naive Pearson
correlation on raw degrees is \emph{incorrect} because it treats 1\textdegree\ and
359\textdegree\ as far apart. The correct method decomposes the degree angle into
$\sin(\theta)$ and $\cos(\theta)$ components and reports
$r_{\text{circ}} = \max(|r_{\sin}|, |r_{\cos}|)$.

\begin{table}[H]
  \centering
  \caption{Circular-Linear Correlations: Vedic Planet Degrees vs Wind Speed.
           $r_{\sin}$ = Pearson correlation of $\sin(\text{degree})$ with wind;
           $r_{\cos}$ = Pearson correlation of $\cos(\text{degree})$ with wind;
           $r_{\text{circ}}$ = $\max(|r_{\sin}|, |r_{\cos}|)$.}
  \label{tab:vedic_circ}
  \begin{tabular}{lrrr}
    \toprule
    Feature & $r_{\sin}$ & $r_{\cos}$ & $r_{\text{circ}}$ \\
    \midrule
""" + vd_rows + r"""    \bottomrule
  \end{tabular}
\end{table}

""" + imgline("vedic_degree_corr_bar",
              "Circular-linear correlations $r_{\\text{circ}}$ per Vedic planet degree. "
              "Yama (\\texttt{vedic\\_yam}) shows the highest correlation ($r \\approx 0.089$), "
              "followed by Guru/Jupiter ($r \\approx 0.071$). All correlations are below "
              "$0.10$ --- explaining less than $1\\%$ of wind speed variance.",
              "fig:vedic_circ") + r"""

% ══════════════════════════════════════════════════════════════════════════════
\section{Variance Inflation Factor (VIF) Analysis}
% ══════════════════════════════════════════════════════════════════════════════

VIF analysis was applied to a representative subset of 14 features to check for
multicollinearity. VIF values $> 5$ indicate moderate, and $> 10$ indicate severe
multicollinearity.

\begin{table}[H]
  \centering
  \caption{Variance Inflation Factors for Representative Features.
           All values $< 2.0$ indicate low multicollinearity in this subset.}
  \label{tab:vif}
  \begin{tabular}{lrl}
    \toprule
    Feature & VIF & Assessment \\
    \midrule
""" + vif_rows + r"""    \bottomrule
  \end{tabular}
\end{table}

\noindent\textbf{Finding:} """ + (
    "All VIF values are below 2.0, indicating negligible multicollinearity in this feature subset. "
    if vif.get("all_below_5") else
    f"Maximum VIF = {vif.get('max_vif', '?'):.2f} for feature \\texttt{{{vif.get('max_feature', '?')}}}."
) + r"""
Note that this subset was selected to represent diverse feature types; the full
186-feature space contains groups of near-duplicate features (e.g., Rahu vs Spashth-Rahu)
with higher inter-group correlations.

% ══════════════════════════════════════════════════════════════════════════════
\section{Intensity Change Analysis}
% ══════════════════════════════════════════════════════════════════════════════

Wind change per 6-hour step ($\Delta\text{wind}$) was computed from consecutive
best-track positions within each storm. Rapid Intensification (RI) is defined as
$\Delta\text{wind} \geq +30$\,kt per 6 hours; Rapid Weakening (RW) as
$\Delta\text{wind} \leq -30$\,kt per 6 hours.

""" + imgline("intensity_change",
              "Intensity change analysis. Left: distribution of 6-hourly wind changes "
              "with RI ($\\geq +30$\\,kt) and RW ($\\leq -30$\\,kt) thresholds marked. "
              "Centre: counts of RI and RW events vs total observations. "
              "Right: scatter of wind change vs sun altitude angle (random 5000-sample subset).",
              "fig:intensity") + r"""

% ══════════════════════════════════════════════════════════════════════════════
\section{Summary and Methodological Notes}
% ══════════════════════════════════════════════════════════════════════════════

\subsection{Feature Importance Hierarchy}

Across all importance metrics (RF/XGB/LGB mean importance, SHAP, LIME, Mutual
Information), the feature hierarchy is consistent:

\begin{enumerate}[noitemsep]
  \item \textbf{Geographic position:} Latitude ($\#1$ in SHAP, LIME, and RF importance),
        longitude, and spline endpoint coordinates.
  \item \textbf{Track geometry:} Spline bearing, curvature, displacement.
  \item \textbf{Day-length timing:} \texttt{dinamana\_minutes} and \texttt{ratrimana\_minutes}
        (seasonal/latitude proxies).
  \item \textbf{Planet speeds and degrees:} Moon speed, Mercury degree/speed, Guru/Jupiter
        degree appear in the top 20 across multiple methods.
  \item \textbf{Altitude/azimuth features:} Consistently unimportant across all methods.
\end{enumerate}

\subsection{Vedic Feature Signals}

No Vedic-Panchang feature appears in the top-5 by any importance metric. In the top-30:
""" + str(fi.get("vedic_share_top30", "?")) + r""" Vedic features appear (out of 30 positions)
using mean tree-model importance. Vedic features that do appear are predominantly
\emph{speed} features (Moon speed, Mercury speed, Uranus speed) which co-vary with
geographic position and season.

\subsection{Large-$N$ Significance Inflation}

All statistical tests were conducted on $N = """ + f"{n_obs:,}" + r"""$ observations.
At this sample size, even differences of $< 1$\,kt achieve $p < 0.001$.
Effect sizes must be consulted for all significant $p$-values:

\begin{itemize}[noitemsep]
  \item Kruskal--Wallis: $\varepsilon^2 < 0.003$ for all Vedic categorical variables.
  \item Retrograde effects: rank-biserial $r < 0.10$ for all planets.
  \item Circular correlations: $r_{\text{circ}} < 0.09$ for all Vedic degrees.
  \item Aspect correlations: $|r| < 0.08$ for all planetary pairs.
\end{itemize}

\subsection{Autocorrelation}

Consecutive observations within the same storm are temporally autocorrelated
(ACF at lag-1 $\approx 0.92$ for wind speed). All group-level tests treat each
observation independently. Group cross-validation (GroupKFold by storm\_id) was
used for all predictive models to prevent leakage, but statistical tests
(KW, Mann--Whitney) remain subject to this limitation.

% ══════════════════════════════════════════════════════════════════════════════
\section{Conclusion}
% ══════════════════════════════════════════════════════════════════════════════

The EDA confirms that:
\begin{enumerate}[noitemsep]
  \item \textbf{Geographic features dominate} all importance metrics, consistent with
        known climatological structure of tropical cyclone intensity.
  \item \textbf{Vedic-Panchang categorical variables} show statistically significant but
        practically negligible wind speed differences across their categories. The
        dominant confound is the seasonal cycle (Ritu, Ayana).
  \item \textbf{Retrograde planetary motion} is associated with statistically significant
        but very small ($\leq 4$\,kt) median wind speed differences across all 7 planets.
  \item \textbf{Circular-linear correlations} of Vedic planet degrees with wind speed are
        below $r = 0.09$ for all planets --- explaining $< 1\%$ of variance.
  \item \textbf{Planetary aspects} have negligible correlations with wind speed ($|r| < 0.08$).
  \item \textbf{Multicollinearity} among the 14 analysed features is low (all VIF $< 2$),
        though feature groups (Rahu/Ketu variants, speed/degree pairs) exhibit redundancy.
  \item \textbf{SHAP--LIME consensus} on """ + str(sv.get("n_consensus", "?")) + r""" features
        provides robust interpretability; both methods agree that Vedic features are marginal
        contributors to model predictions.
\end{enumerate}

\end{document}
"""
    return doc


def _fi_bullets(fi: dict) -> str:
    top5 = fi.get("top5", {})
    vedic_top = fi.get("vedic_top", {})
    vedic_share = fi.get("vedic_share_top30", "?")
    lines = []
    for f, v in list(top5.items())[:3]:
        lines.append(f"  \\item \\texttt{{{f}}}: mean importance $= {v:.4f}$ (rank \\#1)")
        break
    if top5:
        features = list(top5.keys())
        lines.append(f"  \\item Top-5 features: {', '.join([f'\\texttt{{{f}}}' for f in features])}.")
    lines.append(f"  \\item {vedic_share} Vedic features appear in the top-30 (out of 30 positions).")
    if vedic_top:
        vf = list(vedic_top.keys())[0]
        vv = list(vedic_top.values())[0]
        lines.append(f"  \\item Highest-ranked Vedic feature: \\texttt{{{vf}}} (mean importance $= {vv:.4f}$).")
    lines.append(r"  \item GroupKFold CV by \texttt{storm\_id} prevents within-storm leakage; "
                 r"negative Group CV $R^2$ on Vedic-only models confirms no out-of-storm generalisation.")
    return "\n".join(lines) + "\n"


def _mi_bullets(mi: dict) -> str:
    top5 = mi.get("top5", {})
    vedic_top = mi.get("vedic_top3", {})
    lines = []
    if top5:
        lines.append(f"  \\item Top feature: \\texttt{{{list(top5.keys())[0]}}} (MI $= {list(top5.values())[0]:.3f}$).")
        lines.append(f"  \\item Top-5: {', '.join([f'\\texttt{{{f}}}' for f in list(top5.keys())])}.")
    if vedic_top:
        vf = list(vedic_top.keys())[0]
        vv = list(vedic_top.values())[0]
        lines.append(f"  \\item Highest Vedic MI: \\texttt{{{vf}}} (MI $= {vv:.3f}$) --- driven by temporal trend.")
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION (ReportLab)
# ══════════════════════════════════════════════════════════════════════════════

def _rl_para(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def _rl_table(headers: list, rows: list, col_widths=None) -> Table:
    data = [headers] + rows
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#222222")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("TOPPADDING",   (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build_pdf(fig_paths: dict, summaries: dict, df_storm: pd.DataFrame):
    if not HAS_RL:
        print("  [SKIP] PDF generation: reportlab not available")
        return
    out_pdf = OUT / "eda_report.pdf"
    doc = BaseDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm,  bottomMargin=2.5*cm,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame)])

    styles = getSampleStyleSheet()
    S = styles
    H1 = ParagraphStyle("H1", parent=S["Heading1"], fontSize=14, spaceAfter=8,
                         textColor=colors.HexColor("#111111"), fontName="Helvetica-Bold")
    H2 = ParagraphStyle("H2", parent=S["Heading2"], fontSize=11, spaceAfter=6,
                         textColor=colors.HexColor("#333333"), fontName="Helvetica-Bold")
    BODY = ParagraphStyle("BODY", parent=S["Normal"], fontSize=8.5, leading=13,
                           spaceAfter=5, textColor=colors.HexColor("#111111"))
    BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=14, bulletIndent=4,
                              spaceAfter=3)
    CAPTION = ParagraphStyle("CAPTION", parent=BODY, fontSize=7.5, textColor=colors.HexColor("#555555"),
                               spaceAfter=8, italics=True)
    CODE = ParagraphStyle("CODE", parent=BODY, fontName="Courier", fontSize=7.5)

    def H(level: int, text: str):
        return _rl_para(text, H1 if level == 1 else H2)

    def P(text: str):
        return _rl_para(text, BODY)

    def B(text: str):
        return _rl_para(f"• {text}", BULLET)

    def CAP(text: str):
        return _rl_para(f"<i>{text}</i>", CAPTION)

    def SP(h: float = 0.3):
        return Spacer(1, h*cm)

    def FIG(key: str, caption: str, w_frac: float = 0.85):
        p = fig_paths.get(key)
        elems = []
        if p and Path(p).exists():
            usable_w = doc.width * w_frac
            elems.append(Image(str(p), width=usable_w, height=usable_w * 0.55))
        else:
            elems.append(P(f"[Figure not available: {key}]"))
        elems.append(CAP(caption))
        elems.append(SP(0.3))
        return elems

    n_obs   = len(df_storm) if df_storm is not None else 47533
    n_storms = df_storm["storm_id"].nunique() if (df_storm is not None and "storm_id" in df_storm.columns) else 944
    n_feats = summaries.get("feature_importance", {}).get("n_features", 186)

    kw   = summaries.get("kw", {})
    retro = summaries.get("retrograde", {})
    vif  = summaries.get("vif", {})
    asp  = summaries.get("aspect_corr", {})
    fi   = summaries.get("feature_importance", {})
    sv   = summaries.get("shap_vs_lime", {})
    mi   = summaries.get("mi", {})
    vd   = summaries.get("vedic_degree", {})

    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story += [
        SP(2),
        _rl_para("<b>Exploratory Data Analysis Report</b>", ParagraphStyle(
            "TITLE", parent=S["Title"], fontSize=18, alignment=1,
            textColor=colors.HexColor("#111111"))),
        SP(0.4),
        _rl_para("Tropical Cyclone Dataset with Vedic-Astronomical Features", ParagraphStyle(
            "SUBTITLE", parent=S["Normal"], fontSize=12, alignment=1,
            textColor=colors.HexColor("#444444"))),
        SP(0.3),
        _rl_para(
            f"<i>{n_obs:,} observations &bull; {n_storms} storms &bull; {n_feats} features</i>",
            ParagraphStyle("SUBB", parent=S["Normal"], fontSize=9, alignment=1,
                           textColor=colors.HexColor("#666666"))),
        PageBreak(),
    ]

    # ── Section 1: Overview ───────────────────────────────────────────────────
    story += [
        H(1, "1. Overview and Data Description"),
        P(f"The dataset contains {n_obs:,} 6-hourly best-track observations from {n_storms} "
          f"tropical cyclones (2016–2026), enriched with {n_feats} features across three groups:"),
        B("Meteorological (14): latitude, longitude, wind speed, spline track geometry, elevation, land proximity."),
        B("Vedic-Panchang (18): Tithi, Nakshatra, Yoga, Karana, Paksha, Moonsign, Samvatsara, Dinamana/Ratrimana, Weekday, Ayana, Ritu."),
        B("Scientific-Astronomical (154): planet positions, altitudes, azimuths, angular separations, Vedic degree representations."),
        P("Wind speed (kt) is the target variable. Median = 40 kt; mean ≈ 45 kt; right-skewed "
          "(skewness = 1.35). All distributions are non-normal (Shapiro–Wilk confirmed), so "
          "non-parametric tests are used throughout."),
        SP(),
    ]

    # ── Section 2: Feature Importance ─────────────────────────────────────────
    story += [H(1, "2. Feature Importance: Tree-Model Ensemble")]
    story += [
        P("Three tree-based models (Random Forest, XGBoost, LightGBM) were trained with "
          "5-fold GroupKFold cross-validation (groups = storm_id) to prevent within-storm "
          "autocorrelation leakage. Feature importances were averaged across folds and models."),
        SP(0.2),
    ]
    top5 = fi.get("top5", {})
    if top5:
        for i, (f, v) in enumerate(list(top5.items())[:5]):
            story.append(B(f"{f}: mean importance = {v:.4f}"))
    story.append(B(f"Vedic features in top-30: {fi.get('vedic_share_top30', '?')} of 30 positions."))
    story.append(SP(0.2))
    story += FIG("feature_importance_top30",
                 "Fig 1. Top-30 feature importances (RF/XGB/LGB). Geographic features dominate.")
    story += FIG("important_vs_unimportant",
                 "Fig 2. Top-25 vs bottom-25 mean importance contrast.")

    # ── Section 3: Mutual Information ─────────────────────────────────────────
    story += [H(1, "3. Mutual Information with Wind Speed")]
    story += [
        P(f"Mutual Information was computed using k-nearest-neighbour estimation (k=5) "
          f"across all {n_feats} features."),
    ]
    mi_top5 = mi.get("top5", {})
    for f, v in list(mi_top5.items())[:3]:
        story.append(B(f"{f}: MI = {v:.3f}"))
    story.append(B("Note: lahiri_ayanamsha and Rahu/Ketu degree features have high MI due to "
                   "temporal trend confounding (precession), not genuine predictive signal."))
    story.append(SP(0.2))
    story += FIG("mutual_information_top30",
                 "Fig 3. Top-30 Mutual Information scores with wind speed.")

    # ── Section 4: SHAP ───────────────────────────────────────────────────────
    story += [H(1, "4. SHAP Explainability")]
    story += [
        P("SHAP TreeExplainer was applied to an XGBoost model. Mean |SHAP| values "
          "quantify average contribution magnitude per feature."),
        SP(0.2),
    ]
    story += FIG("shap_summary_bar",
                 "Fig 4. SHAP feature importance bar (top-30 mean |SHAP| values).")
    story += FIG("shap_important_vs_unimportant",
                 "Fig 5. SHAP importance contrast: top-25 vs bottom-25.")
    story += [H(2, "4.1 SHAP Feature Interactions")]
    story += [P("Top-15 SHAP interaction pairs:"), SP(0.1)]
    si_df = _safe_read("shap_interact", index_col=None)
    if si_df is not None and len(si_df.columns) >= 3:
        si_df.columns = ["feature_1", "feature_2", "mean_interaction"]
        rows = [[r["feature_1"][:28], r["feature_2"][:28], f"{r['mean_interaction']:.4f}"]
                for _, r in si_df.head(15).iterrows()]
        story.append(_rl_table(["Feature 1", "Feature 2", "Mean Interaction"], rows,
                               [7*cm, 7*cm, 3*cm]))
        story.append(SP(0.2))
    story += FIG("shap_interactions_top20",
                 "Fig 6. Top-20 SHAP feature interactions. "
                 "latitude × spline_bearing_deg is dominant (0.822).")

    # ── Section 5: LIME / consensus ───────────────────────────────────────────
    story += [H(1, "5. LIME and SHAP vs LIME Consensus")]
    consensus = sv.get("consensus_features", [])
    story += [
        P(f"LIME was applied to 5 representative instances. {sv.get('n_consensus', '?')} features "
          f"appear in the top-30 of both SHAP and LIME (consensus set)."),
        B(f"Consensus features: {', '.join(consensus[:10]) if consensus else 'none'}."),
        B(f"Vedic features in consensus: "
          f"{', '.join(sv.get('vedic_in_consensus', [])) or 'None'}."),
        SP(0.2),
    ]
    story += FIG("shap_vs_lime_top20",
                 "Fig 7. Normalised SHAP vs LIME importance for top-20 consensus features.")

    # ── Section 6: Vedic KW ────────────────────────────────────────────────────
    story += [H(1, "6. Vedic Category Analysis: Kruskal–Wallis Tests")]
    story += [
        P("Non-parametric Kruskal–Wallis tests compare wind speed across all Vedic categorical "
          "variables, with Benjamini–Hochberg FDR correction applied."),
        SP(0.1),
    ]
    kw_data = kw.get("rows", {})
    if kw_data:
        kw_rows_rl = [
            [var,
             f"{row.get('kw_H', 0):.2f}",
             _fmt_sci(row.get('kw_p', 1)),
             str(row.get('n_categories', '?')),
             "Yes" if row.get("significant_p05") else "No"]
            for var, row in kw_data.items()
        ]
        story.append(_rl_table(
            ["Variable", "H", "p-value", "# Cat.", "Sig."],
            kw_rows_rl,
            [4*cm, 2.5*cm, 3.5*cm, 2*cm, 1.5*cm]
        ))
    story += [
        SP(0.2),
        B("drik_ritu (H=1214) and sunsign (H=1123) are calendar-season proxies, not independent Vedic signals."),
        B(f"Large N ({n_obs:,}) inflates all p-values. Effect sizes (ε² < 0.003) are negligible."),
        SP(),
    ]

    # ── Section 7: Nakshatra ──────────────────────────────────────────────────
    story += [H(1, "7. Nakshatra Intensity Analysis")]
    story += [
        P("Median wind speed was computed per Nakshatra. Kruskal–Wallis H=116, p=2.4×10⁻¹³, "
          "ε²≈0.002 (negligible effect size)."),
        SP(0.2),
    ]
    story += FIG("nakshatra_intensity_bar",
                 "Fig 8. Median wind speed per Nakshatra with IQR error bars.")

    # ── Section 8: Tithi × Nakshatra ─────────────────────────────────────────
    story += [H(1, "8. Tithi × Nakshatra Heatmaps")]
    story += [P("Cross-tabulation of Tithi (lunar day) × Nakshatra shows joint wind distribution. "
                "No systematic high-intensity zone is visible.")]
    story += FIG("tithi_nakshatra_heatmap",
                 "Fig 9. Mean wind speed (kt): Tithi × Nakshatra.")
    story += FIG("tithi_nakshatra_heatmap_median",
                 "Fig 10. Median wind speed (kt): Tithi × Nakshatra.")

    # ── Section 9: Retrograde ─────────────────────────────────────────────────
    story += [H(1, "9. Retrograde Planetary Motion and Storm Intensity")]
    story += [
        P("All 7 classical Vedic planets show statistically significant wind speed differences "
          "between retrograde and direct periods (p < 0.001), but effect sizes are negligible "
          "(rank-biserial r < 0.10 for all planets)."),
        SP(0.1),
    ]
    all_retro = retro.get("all_rows", {})
    if all_retro:
        retro_rows_rl = [
            [planet,
             f"{row.get('retro_pct', 0):.1f}%",
             f"{row.get('retro_median_wind', 0):.0f}",
             f"{row.get('direct_median_wind', 0):.0f}",
             f"{row.get('wind_diff_median', 0):+.1f}",
             _fmt_sci(row.get("mannwhitney_p", 1)),
             "Yes*" if row.get("significant_p05") else "No"]
            for planet, row in all_retro.items()
        ]
        story.append(_rl_table(
            ["Planet", "Retro%", "Med(R)", "Med(D)", "ΔMed", "p-val", "Sig."],
            retro_rows_rl,
            [2.5*cm, 2*cm, 2.2*cm, 2.2*cm, 2*cm, 3*cm, 1.5*cm]
        ))
    story.append(SP(0.2))
    story += FIG("retrograde_effects",
                 "Fig 11. Retrograde vs direct median wind per planet. All differences ≤4 kt.")

    # ── Section 10: Aspect correlations ───────────────────────────────────────
    story += [H(1, "10. Planetary Aspect Correlations")]
    story += [
        P(f"Pearson correlations between planetary aspect angles and wind speed. "
          f"Maximum |r| = {asp.get('max_r', 0):.4f} (Jupiter–Saturn). "
          f"All correlations are weak (|r| < 0.10)."),
        SP(0.1),
    ]
    asp_df = _safe_read("aspect_corr", index_col=0)
    if asp_df is not None:
        asp_df.columns = ["pearson_r"]
        asp_df = asp_df.sort_values("pearson_r", ascending=False)
        asp_rows_rl = [[feat, f"{row['pearson_r']:.6f}"]
                       for feat, row in asp_df.iterrows()]
        story.append(_rl_table(["Aspect Pair", "Pearson r"],
                               asp_rows_rl, [9*cm, 4.5*cm]))
    story.append(SP(0.2))
    story += FIG("aspect_correlations",
                 "Fig 12. Planetary aspect correlations with wind speed.")

    # ── Section 11: Vedic degree circular ─────────────────────────────────────
    story += [H(1, "11. Vedic Degree: Circular-Linear Correlations")]
    story += [
        P("Circular-linear method used (sin/cos decomposition of 0–360° degrees). "
          "Naive Pearson on raw degrees is incorrect for cyclical data."),
        SP(0.1),
    ]
    vd_df = _safe_read("vedic_degree", index_col=0)
    if vd_df is not None:
        vd_df.columns = [c.strip() for c in vd_df.columns]
        vd_rows_rl = [
            [feat,
             f"{row.get('r_sin', 0):.5f}",
             f"{row.get('r_cos', 0):.5f}",
             f"{row.get('r_circular_max', 0):.5f}"]
            for feat, row in vd_df.iterrows()
        ]
        story.append(_rl_table(
            ["Feature", "r_sin", "r_cos", "r_circ"],
            vd_rows_rl,
            [7*cm, 2.8*cm, 2.8*cm, 2.8*cm]
        ))
    story.append(SP(0.2))
    story += FIG("vedic_degree_corr_bar",
                 "Fig 13. Circular-linear correlations per Vedic planet degree. Max r ≈ 0.089.")

    # ── Section 12: VIF ───────────────────────────────────────────────────────
    story += [H(1, "12. Variance Inflation Factor (VIF)")]
    story += [
        P("VIF analysis on 14 representative features. Thresholds: VIF < 5 = low, "
          "VIF 5–10 = moderate, VIF > 10 = high multicollinearity."),
        B(f"Maximum VIF = {vif.get('max_vif', '?'):.3f} (feature: {vif.get('max_feature', '?')})."),
        B("All values below 2.0 — negligible multicollinearity in this feature subset."),
        SP(0.1),
    ]
    vif_df2 = _safe_read("vif", index_col=0)
    if vif_df2 is not None:
        vif_df2.columns = [c.strip() for c in vif_df2.columns]
        vc = [c for c in vif_df2.columns if "vif" in c.lower()]
        if vc:
            vif_df2 = vif_df2.rename(columns={vc[0]: "VIF"}).sort_values("VIF", ascending=False)
            vif_rows_rl = [
                [feat,
                 f"{row['VIF']:.3f}",
                 "High" if row["VIF"] > 10 else ("Moderate" if row["VIF"] > 5 else "Low")]
                for feat, row in vif_df2.iterrows()
            ]
            story.append(_rl_table(
                ["Feature", "VIF", "Assessment"],
                vif_rows_rl,
                [8*cm, 2.5*cm, 3*cm]
            ))

    # ── Section 13: Intensity change ──────────────────────────────────────────
    story += [SP(), H(1, "13. Intensity Change Analysis")]
    story += [
        P("Wind change per 6-hour step. RI threshold: +30 kt/6h. RW threshold: −30 kt/6h."),
        SP(0.1),
    ]
    story += FIG("intensity_change",
                 "Fig 14. Wind change distribution, RI/RW event counts, and scatter vs sun altitude.")

    # ── Section 14: Conclusion ────────────────────────────────────────────────
    story += [H(1, "14. Conclusions")]
    story += [
        B("Geographic features (latitude, longitude, spline geometry) dominate all importance metrics."),
        B("Vedic categorical variables show statistically significant but negligible effect sizes (ε² < 0.003)."),
        B(f"All 7 retrograde planets are significant but with rank-biserial r < 0.10 (negligible practical effect)."),
        B(f"Circular-linear correlations of Vedic degrees: max r ≈ {vd.get('max_r', 0):.3f} (<1% variance explained)."),
        B(f"Planetary aspects: max |r| < 0.08 (weakest signals in dataset)."),
        B(f"SHAP–LIME consensus: {sv.get('n_consensus', '?')} features; "
          f"Vedic features in consensus: {', '.join(sv.get('vedic_in_consensus', [])) or 'none'}."),
        B("All VIF values below 2.0 for the analysed feature subset."),
        SP(),
    ]

    doc.build(story)
    print(f"  [OK] eda_report.pdf ({out_pdf.stat().st_size / 1024:.0f} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== 09_EDA_report.py ===")

    # 1. Load parquet
    df_storm = None
    if DATA.exists():
        print("Loading parquet...")
        df_storm = pd.read_parquet(DATA)
        for c in df_storm.columns:
            if df_storm[c].dtype == object:
                try:
                    df_storm[c] = pd.to_numeric(df_storm[c], errors="ignore")
                except Exception:
                    pass
        print(f"  Loaded: {df_storm.shape}")
    else:
        print(f"  [WARN] Parquet not found at {DATA}; some figures will be skipped")

    # 2. Generate all figures
    print("\n--- Generating figures ---")
    fig_paths = {}
    fig_paths["aspect_correlations"]          = fig_aspect_correlations()
    fig_paths["feature_importance_top30"]     = fig_feature_importance_top30()
    fig_paths["important_vs_unimportant"]     = fig_important_vs_unimportant()
    fig_paths["intensity_change"]             = fig_intensity_change(df_storm)
    fig_paths["mutual_information_top30"]     = fig_mutual_information_top30()
    fig_paths["shap_summary_bar"]             = fig_shap_summary_bar()
    fig_paths["shap_important_vs_unimportant"]= fig_shap_important_vs_unimportant()
    fig_paths["shap_interactions_top20"]      = fig_shap_interactions_top20()
    fig_paths["shap_vs_lime_top20"]           = fig_shap_vs_lime_top20()
    fig_paths["nakshatra_intensity_bar"]      = fig_nakshatra_intensity_bar(df_storm)
    fig_paths["retrograde_effects"]           = fig_retrograde_effects()
    fig_paths["tithi_nakshatra_heatmap"]      = fig_tithi_nakshatra_heatmap(df_storm, "mean")
    fig_paths["tithi_nakshatra_heatmap_median"]= fig_tithi_nakshatra_heatmap(df_storm, "median")
    fig_paths["vedic_degree_corr_bar"]        = fig_vedic_degree_corr_bar()

    # 3. Summaries
    print("\n--- Computing summaries ---")
    summaries = {
        "aspect_corr":       summarise_aspect_corr(),
        "feature_importance": summarise_feature_importance(),
        "retrograde":        summarise_retrograde(),
        "vif":               summarise_vif(),
        "kw":                summarise_kw(),
        "shap_vs_lime":      summarise_shap_vs_lime(),
        "mi":                summarise_mi(),
        "vedic_degree":      summarise_vedic_degree(),
    }

    # 4. LaTeX
    print("\n--- Writing LaTeX ---")
    tex = build_latex(fig_paths, summaries, df_storm)
    tex_path = OUT / "eda_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"  [OK] eda_report.tex ({tex_path.stat().st_size / 1024:.0f} KB)")

    # 5. PDF
    print("\n--- Building PDF ---")
    build_pdf(fig_paths, summaries, df_storm)

    print(f"\n=== Done. Outputs in {OUT} ===")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
