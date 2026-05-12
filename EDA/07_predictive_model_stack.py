"""
07_predictive_model_stack.py
============================
Production-style predictive modeling for storm intensification.

What this script does:
1. Builds a next-step wind-change target from track sequence.
2. Trains block-wise models:
   - Meteo only
   - Meteo + Panchangam
   - Full (Meteo + Panchangam + Scientific)
3. Trains a stacked residual model:
   - Stage A: Meteo baseline
   - Stage B: Residual model on Panchangam + Scientific
4. Evaluates with chronological holdout and forward-chaining validation.
5. Builds RI classification target with adaptive threshold.
6. Estimates group-level contribution by permutation (RMSE lift when shuffled).

Outputs are written to EDA/output/predictive_model/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from pandas import CategoricalDtype
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    f1_score,
)

DATA_PATH = Path(__file__).resolve().parent / "output" / "storm_data_all.parquet"
OUT_DIR = Path(__file__).resolve().parent / "output" / "predictive_model"
OUT_DIR.mkdir(exist_ok=True)


@dataclass
class BlockFeatures:
    meteo: List[str]
    panchang: List[str]
    scientific: List[str]


def safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def discover_blocks(df: pd.DataFrame) -> BlockFeatures:
    meteo_candidates = [
        "wind",
        "pressure",
        "longitude",
        "latitude",
        "hour_utc",
        "day_of_year",
        "day_of_week",
        "month",
        "movement_speed",
        "displacement",
        # LEAKAGE WARNING: spline_* features encode FUTURE track geometry (the smoothed
        # forecast path). Using spline_bearing_deg, spline_curvature, spline_end_lon/lat,
        # spline_displacement, spline_length_deg to predict next_wind_delta is problematic
        # because these spline features describe where the storm is forecasted to go —
        # information that implies knowledge of the future trajectory. They are kept here
        # in the meteo block for completeness but are EXCLUDED from the scientific block
        # to prevent double-counting. Any model improvement from these features likely
        # reflects this implicit future knowledge rather than learned physical relationships.
        # For unbiased evaluation, consider re-running models without these spline features.
        "spline_curvature",
        "spline_bearing_deg",
        "spline_displacement",
        "spline_length_deg",
    ]

    panchang_candidates = [
        "tithi",
        "nakshatra",
        "yoga",
        "karana",
        "weekday",
        "paksha",
        "moonsign",
        "sunsign",
        "drik_ritu",
        "vedic_ritu",
        "drik_ayana",
        "vedic_ayana",
        "anandadi_yoga",
        "tamil_yoga",
        "samvatsara",
        "rahu_kalam",
        "yamaganda",
        "gulikai_kalam",
    ]

    scientific_prefixes = (
        "sun_",
        "moon_",
        "mercury_",
        "venus_",
        "mars_",
        "jupiter_",
        "saturn_",
        "uranus_",
        "neptune_",
        "vedic_",
        "lunar_node_",
        "cone_",
        # NOTE: spline_* features are intentionally NOT included in the scientific block
        # because they are already in the meteo block and carry leakage risk (future track
        # geometry). Including "spline_" here would double-count them.
    )

    meteo = [c for c in meteo_candidates if c in df.columns]
    panchang = [c for c in panchang_candidates if c in df.columns]

    blocked = set(
        [
            "storm_id",
            "track_id",
            "track_num",
            "date_utc",
            "date_local",
            "time_local",
            "wind_change",
            "pressure_change",
            "next_wind_delta",
            "ri_target",
        ]
        + meteo
        + panchang
    )

    scientific = [
        c
        for c in df.columns
        if c not in blocked and c.startswith(scientific_prefixes)
    ]

    return BlockFeatures(meteo=meteo, panchang=panchang, scientific=scientific)


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["date_utc"] = pd.to_datetime(work["date_utc"], errors="coerce")

    sort_cols = [c for c in ["storm_id", "track_num", "date_utc"] if c in work.columns]
    work = work.sort_values(sort_cols).reset_index(drop=True)

    work["next_wind"] = work.groupby("storm_id")["wind"].shift(-1)
    work["next_wind_delta"] = work["next_wind"] - work["wind"]

    # Drop rows without target (typically final track in each storm)
    work = work.dropna(subset=["next_wind_delta", "date_utc", "storm_id"]).reset_index(drop=True)
    return work


def encode_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df[features].copy()
    test = test_df[features].copy()

    for col in features:
        if is_object_dtype(train[col]) or is_string_dtype(train[col]) or isinstance(train[col].dtype, CategoricalDtype):
            train[col] = train[col].astype(str)
            test[col] = test[col].astype(str)

            mapping = {v: i for i, v in enumerate(train[col].dropna().unique())}
            train[col] = train[col].map(mapping).fillna(-1)
            test[col] = test[col].map(mapping).fillna(-1)
        else:
            train[col] = safe_float(train[col])
            test[col] = safe_float(test[col])

        median = train[col].median()
        if pd.isna(median):
            median = 0.0
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)

    return train, test


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def fit_regressor(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=180,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_depth=18,
    )
    model.fit(X, y)
    return model


def choose_ri_threshold(train_y: pd.Series, test_y: pd.Series) -> float:
    # Choose the strictest threshold that still has enough positives in both splits.
    # NOTE: This selects from [20, 15, 10, 5] kt — all NEXT-STEP deltas, NOT 24h deltas.
    # This differs from 06_hypothesis_testing.py which uses 30 kt over a 24h rolling window
    # (closer to the WMO standard of 35 kt/24h). Cross-script RI comparisons are invalid
    # because these definitions are fundamentally different:
    #   - Here: raw next-track wind_delta (can be 1-6 hours apart)
    #   - 06_hypothesis_testing: 24-hour rolling window sum
    # The adaptive selection also means the threshold changes with each dataset split,
    # making results non-reproducible if the data changes. The selected threshold is
    # logged in MODEL_REPORT.md for reference.
    for t in [20, 15, 10, 5]:
        tr_pos = int((train_y >= t).sum())
        te_pos = int((test_y >= t).sum())
        if tr_pos >= 100 and te_pos >= 20:
            return float(t)
    return 5.0


def forward_cv_boundaries(dates: pd.Series, folds: int = 5) -> List[pd.Timestamp]:
    uniq = np.sort(pd.to_datetime(dates).dropna().unique())
    if len(uniq) < 50:
        return []
    qs = np.linspace(0.2, 0.9, folds)
    return [pd.Timestamp(uniq[int((len(uniq) - 1) * q)]) for q in qs]


def group_permutation_importance(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups: Dict[str, List[str]],
    repeats: int = 5,
) -> pd.DataFrame:
    base_pred = model.predict(X_test)
    base_rmse = np.sqrt(mean_squared_error(y_test, base_pred))

    rows = []
    rng = np.random.default_rng(42)

    for group_name, cols in groups.items():
        cols = [c for c in cols if c in X_test.columns]
        if not cols:
            continue

        lifts = []
        for _ in range(repeats):
            shuffled = X_test.copy()
            idx = rng.permutation(len(shuffled))
            shuffled.loc[:, cols] = shuffled[cols].to_numpy()[idx]
            pred = model.predict(shuffled)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            lifts.append(float(rmse - base_rmse))

        rows.append(
            {
                "group": group_name,
                "rmse_lift_mean": float(np.mean(lifts)),
                "rmse_lift_std": float(np.std(lifts)),
            }
        )

    out = pd.DataFrame(rows).sort_values("rmse_lift_mean", ascending=False)
    return out


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df = prepare_target(df)

    blocks = discover_blocks(df)

    base_cols = [c for c in ["storm_id", "date_utc", "next_wind_delta"] if c in df.columns]
    use_cols = list(dict.fromkeys(base_cols + blocks.meteo + blocks.panchang + blocks.scientific))
    work = df[use_cols].copy()

    split_date = work["date_utc"].quantile(0.8, interpolation="nearest")
    train_df = work[work["date_utc"] <= split_date].copy()
    test_df = work[work["date_utc"] > split_date].copy()

    y_train = train_df["next_wind_delta"]
    y_test = test_df["next_wind_delta"]

    meteo_feats = blocks.meteo
    meteo_panchang_feats = list(dict.fromkeys(blocks.meteo + blocks.panchang))
    full_feats = list(dict.fromkeys(blocks.meteo + blocks.panchang + blocks.scientific))
    residual_feats = list(dict.fromkeys(blocks.panchang + blocks.scientific))

    # Prepare encoded sets
    Xtr_m, Xte_m = encode_train_test(train_df, test_df, meteo_feats)
    Xtr_mp, Xte_mp = encode_train_test(train_df, test_df, meteo_panchang_feats)
    Xtr_f, Xte_f = encode_train_test(train_df, test_df, full_feats)
    Xtr_r, Xte_r = encode_train_test(train_df, test_df, residual_feats)

    # Block models
    m_meteo = fit_regressor(Xtr_m, y_train)
    p_meteo = m_meteo.predict(Xte_m)

    m_mp = fit_regressor(Xtr_mp, y_train)
    p_mp = m_mp.predict(Xte_mp)

    m_full = fit_regressor(Xtr_f, y_train)
    p_full = m_full.predict(Xte_f)

    # Stacked residual model
    p1_train = m_meteo.predict(Xtr_m)
    residual_train = y_train - p1_train
    m_residual = fit_regressor(Xtr_r, residual_train)
    p_stacked = p_meteo + m_residual.predict(Xte_r)

    reg_rows = []
    for name, pred in [
        ("meteo_only", p_meteo),
        ("meteo_panchang", p_mp),
        ("full_all_blocks", p_full),
        ("stacked_residual", p_stacked),
    ]:
        m = evaluate_regression(y_test, pred)
        reg_rows.append({"model": name, **m})

    reg_df = pd.DataFrame(reg_rows).sort_values("rmse")
    reg_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    # Forward-chaining validation (lightweight)
    fold_rows = []
    boundaries = forward_cv_boundaries(work["date_utc"], folds=5)
    for i, b in enumerate(boundaries, start=1):
        train_fold = work[work["date_utc"] < b]
        valid_fold = work[(work["date_utc"] >= b) & (work["date_utc"] < b + pd.Timedelta(days=120))]
        if len(train_fold) < 5000 or len(valid_fold) < 500:
            continue

        ytr = train_fold["next_wind_delta"]
        yva = valid_fold["next_wind_delta"]

        xtr_m, xva_m = encode_train_test(train_fold, valid_fold, meteo_feats)
        xtr_f, xva_f = encode_train_test(train_fold, valid_fold, full_feats)
        xtr_r, xva_r = encode_train_test(train_fold, valid_fold, residual_feats)

        mm = fit_regressor(xtr_m, ytr)
        pf_m = mm.predict(xva_m)

        mf = fit_regressor(xtr_f, ytr)
        pf_f = mf.predict(xva_f)

        res_y = ytr - mm.predict(xtr_m)
        mr = fit_regressor(xtr_r, res_y)
        pf_s = pf_m + mr.predict(xva_r)

        fold_rows.extend(
            [
                {
                    "fold": i,
                    "boundary": b,
                    "model": "meteo_only",
                    "rmse": evaluate_regression(yva, pf_m)["rmse"],
                },
                {
                    "fold": i,
                    "boundary": b,
                    "model": "full_all_blocks",
                    "rmse": evaluate_regression(yva, pf_f)["rmse"],
                },
                {
                    "fold": i,
                    "boundary": b,
                    "model": "stacked_residual",
                    "rmse": evaluate_regression(yva, pf_s)["rmse"],
                },
            ]
        )

    cv_df = pd.DataFrame(fold_rows)
    if not cv_df.empty:
        cv_df.to_csv(OUT_DIR / "forward_cv_results.csv", index=False)

    # RI classification with adaptive threshold
    threshold = choose_ri_threshold(y_train, y_test)
    y_train_cls = (y_train >= threshold).astype(int)
    y_test_cls = (y_test >= threshold).astype(int)

    cls_rows = []
    if len(np.unique(y_train_cls)) >= 2 and len(np.unique(y_test_cls)) >= 2:
        clf_base = RandomForestClassifier(
            n_estimators=220,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf_base.fit(Xtr_m, y_train_cls)
        prob_base = clf_base.predict_proba(Xte_m)[:, 1]
        pred_base = (prob_base >= 0.5).astype(int)

        clf_full = RandomForestClassifier(
            n_estimators=220,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf_full.fit(Xtr_f, y_train_cls)
        prob_full = clf_full.predict_proba(Xte_f)[:, 1]
        pred_full = (prob_full >= 0.5).astype(int)

        cls_rows.extend(
            [
                {
                    "model": "ri_meteo_only",
                    "threshold": threshold,
                    "roc_auc": float(roc_auc_score(y_test_cls, prob_base)),
                    "pr_auc": float(average_precision_score(y_test_cls, prob_base)),
                    "f1": float(f1_score(y_test_cls, pred_base, zero_division=0)),
                },
                {
                    "model": "ri_full_all_blocks",
                    "threshold": threshold,
                    "roc_auc": float(roc_auc_score(y_test_cls, prob_full)),
                    "pr_auc": float(average_precision_score(y_test_cls, prob_full)),
                    "f1": float(f1_score(y_test_cls, pred_full, zero_division=0)),
                },
            ]
        )

    cls_df = pd.DataFrame(cls_rows)
    if not cls_df.empty:
        cls_df.to_csv(OUT_DIR / "classification_metrics.csv", index=False)

    # Group contribution analysis on full model
    group_map = {
        "meteo": [c for c in meteo_feats if c in Xte_f.columns],
        "panchang": [c for c in blocks.panchang if c in Xte_f.columns],
        "scientific": [c for c in blocks.scientific if c in Xte_f.columns],
    }
    group_imp = group_permutation_importance(m_full, Xte_f, y_test, group_map)
    group_imp.to_csv(OUT_DIR / "group_importance.csv", index=False)

    # Top individual features from full model
    feat_imp = pd.DataFrame(
        {
            "feature": Xtr_f.columns,
            "importance": m_full.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feat_imp.to_csv(OUT_DIR / "top_features.csv", index=False)

    # Lightweight report
    best_reg = reg_df.iloc[0].to_dict()
    lines = []
    lines.append("# Predictive Modeling Report")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Rows used: {len(work)}")
    lines.append(f"- Train rows: {len(train_df)}")
    lines.append(f"- Test rows: {len(test_df)}")
    lines.append(f"- Split cutoff: {split_date}")
    lines.append("")
    lines.append("## Feature Blocks")
    lines.append(f"- Meteo features: {len(blocks.meteo)}")
    lines.append(f"- Panchangam features: {len(blocks.panchang)}")
    lines.append(f"- Scientific features: {len(blocks.scientific)}")
    lines.append("")
    lines.append("## Regression Summary")
    lines.append(f"- Best model: {best_reg['model']}")
    lines.append(f"- Best RMSE: {best_reg['rmse']:.4f}")
    lines.append(f"- Best MAE: {best_reg['mae']:.4f}")
    lines.append("- Full table: `model_comparison.csv`")
    # Interpret whether adding scientific (astronomical) features helped
    if best_reg['model'] == "meteo_panchang":
        lines.append("- **KEY FINDING**: Best model is `meteo_panchang`, NOT `full`.")
        lines.append("  Adding scientific (astronomical/Vedic planetary) features to meteo+panchang")
        lines.append("  did NOT improve prediction. The `full` model performs worse, indicating")
        lines.append("  that astronomical features add noise rather than signal for next-step wind.")
    elif best_reg['model'] == "meteo":
        lines.append("- **KEY FINDING**: Best model is `meteo` only — neither Panchang nor")
        lines.append("  astronomical features improved over pure meteorological predictors.")
    elif best_reg['model'] == "full":
        lines.append("- **KEY FINDING**: Best model is `full` (meteo + panchang + scientific).")
        lines.append("  This suggests some astronomical/Vedic features added modest predictive value.")
    lines.append("")
    lines.append("## Classification Target")
    lines.append(f"- RI threshold selected for next-step delta: >= {threshold}")
    lines.append(f"- NOTE: This is a NEXT-STEP wind delta, NOT a 24-hour delta.")
    lines.append(f"  It differs from 06_hypothesis_testing.py which uses 30 kt/24h rolling.")
    lines.append(f"  Cross-script RI comparisons are NOT valid.")
    if cls_df.empty:
        lines.append("- Classification metrics skipped due to class sparsity in split.")
    else:
        lines.append("- Metrics table: `classification_metrics.csv`")
    lines.append("")
    lines.append("## Spline Feature Leakage Warning")
    lines.append("- spline_bearing_deg, spline_curvature, spline_displacement, spline_length_deg")
    lines.append("  are included in the METEO block. These encode forecast track geometry and")
    lines.append("  may carry implicit information about the future trajectory (leakage risk).")
    lines.append("  See discover_blocks() in 07_predictive_model_stack.py for full comment.")
    lines.append("")
    lines.append("## Group Contribution")
    lines.append("- Group-level RMSE lift table: `group_importance.csv`")
    lines.append("- Feature ranking: `top_features.csv`")

    with open(OUT_DIR / "MODEL_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved outputs to:", OUT_DIR)
    print("Best regression model:", best_reg["model"], "RMSE=", round(float(best_reg["rmse"]), 4))


if __name__ == "__main__":
    main()
