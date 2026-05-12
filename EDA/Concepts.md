# Prologue

Hello !

* This isn’t just another long EDA document — it’s a practical roadmap that shows how professionals actually **think through a dataset from chaos to clarity**.
* It walks you from raw, messy data to understanding **what matters, what doesn’t, and why patterns exist** instead of relying on random visualizations.
* You’ll learn how to systematically uncover **hidden structure, feature relationships, non-linear effects, and real drivers of outcomes**.
* It connects statistics, machine learning, and visualization into **one continuous reasoning process**, not isolated techniques.
* By following it, datasets stop feeling overwhelming because you gain a **repeatable mental workflow** for analysis.
* It shows how to detect redundancy, anomalies, trends, and natural groupings that most beginners completely miss.
* The later sections reveal how models make decisions using **SHAP and LIME**, turning predictions into understandable explanations.
* Reading this once carefully can shift you from *plotting data* to actually **understanding systems hidden inside data**.
* Think of it as developing **X-ray vision for datasets** — after this, you won’t explore data blindly again.
* If you want analysis to feel structured, confident, and purposeful rather than trial-and-error, this is worth reading end to end.

```
if you read this once end-to-end, you’ll instantly spot the hidden patterns most people miss—and you’ll never “guess” your way through EDA again.
```

## 📊 Introduction — A Complete Framework for Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is the **foundation of all data science, machine learning, and statistical modeling work**. Before building predictive models or drawing conclusions, analysts must first understand how data behaves, how variables interact, and what underlying structure exists within the dataset.

The material presented here forms a **comprehensive, production-grade EDA ecosystem** — not just a collection of plots or statistics, but a **systematic analytical workflow** that transforms raw data into interpretable knowledge.

---

### 🎯 Purpose of This Framework

This framework is designed to answer three essential analytical questions:

1. **What patterns exist in the data?**
2. **Which variables truly matter?**
3. **How is the data internally structured?**

Rather than treating EDA as an informal preliminary step, this approach treats it as a **structured scientific investigation** combining statistics, machine learning, visualization, and explainability.

---

### 🧠 Philosophy Behind the Workflow

Modern datasets are typically:

* High-dimensional
* Noisy and incomplete
* Non-linear in relationships
* Temporally or spatially dependent
* Containing hidden group structures

Because of this complexity, understanding data requires moving through **progressive layers of analysis**, beginning with basic inspection and advancing toward deeper structural interpretation.

The workflow therefore follows a logical progression:

```
Raw Data
   ↓
Data Understanding
   ↓
Cleaning & Feature Preparation
   ↓
Statistical Structure Discovery
   ↓
Dimensional Understanding
   ↓
Importance & Dependency Analysis
   ↓
Pattern & Group Discovery
   ↓
Temporal / Interaction Analysis
   ↓
Explainability & Interpretation
```

---

### 🔬 Scope of the Covered Content

This content integrates multiple analytical domains into one unified pipeline:

#### ✅ Data Inspection & Cleaning

Understanding dataset composition, quality, completeness, and usability.

#### ✅ Feature Engineering & Scaling

Preparing variables so statistical and machine learning methods behave correctly.

#### ✅ Dimensionality Reduction

Revealing latent structure using techniques such as PCA.

#### ✅ Feature Importance & Dependency Analysis

Identifying meaningful predictors using model-based and information-theoretic approaches.

#### ✅ Statistical Testing

Validating whether observed differences are statistically significant rather than random.

#### ✅ Clustering & Segmentation

Discovering natural groupings without predefined labels.

#### ✅ Correlation & Multicollinearity Analysis

Detecting redundancy and stabilizing future models.

#### ✅ Time-Series & Trend Analysis

Understanding dynamics, persistence, seasonality, and change behavior.

#### ✅ Interaction Effects

Studying combined influence between variables rather than isolated effects.

#### ✅ Anomaly & Structural Detection

Finding rare or structurally different observations.

#### ✅ Explainable AI (SHAP & LIME)

Interpreting model predictions both globally and locally.

#### ✅ Automated Reporting & Reproducibility

Saving analytical outputs for auditing, collaboration, and model reference.

---

### ⚙️ Engineering Perspective

The included reusable Python functions convert EDA into a **modular analytical pipeline**:

* Domain-independent
* Dataset-agnostic
* Reusable across projects
* Suitable for research or production environments

This transforms EDA from manual exploration into a **repeatable analytical system**.

---

### 🧩 Educational Objective

Beyond analysis, this framework also serves as a **teaching architecture** for onboarding analysts or interns. It introduces concepts in increasing analytical depth:

1. Describe data
2. Diagnose quality issues
3. Understand distributions
4. Discover relationships
5. Reduce complexity
6. Detect structure
7. Explain outcomes

By following this progression, learners move from **data viewing → data reasoning**.

---

### 🌍 Big Picture

Taken together, this content represents a transition from traditional EDA toward **Advanced Analytical Exploration**, where datasets are examined across multiple dimensions:

| Dimension    | Focus                      |
| ------------ | -------------------------- |
| Statistical  | distributions, tests       |
| Structural   | PCA, clustering            |
| Temporal     | trends, autocorrelation    |
| Multivariate | correlations, interactions |
| Predictive   | feature importance         |
| Explainable  | SHAP, LIME                 |
| Operational  | reproducibility            |

---

### ⭐ In Essence

This entire framework establishes EDA as:

> **A structured process for discovering how a system behaves before attempting to model or predict it.**

It converts raw observations into:

* understanding,
* validated insights,
* and interpretable analytical structure.

# Learning Workflow

## Below is the **clean learning workflow** :

## **Step 1 — Load and Inspect the Dataset**

**Goal:** Understand what data you have.

**Actions**

* Load dataset (`csv`, `parquet`, database, etc.)
* Check:

  * number of rows & columns
  * data types
  * missing values
  * duplicate rows
* Identify:

  * numerical features
  * categorical features
  * target variable (if prediction problem)

**Key Questions**

* What am I predicting?
* Which columns are IDs or useless?
* Which columns contain real information?

---

## **Step 2 — Clean Feature Selection**

**Goal:** Prepare usable variables.

**Actions**

* Keep columns with sufficient data availability
* Remove:

  * IDs
  * leakage variables
  * derived duplicates
  * constant columns
* Handle missing values:

  * drop
  * impute
  * or flag

**Outcome**
👉 Clean feature matrix ready for analysis.

---

## **Step 3 — Standardize Numerical Features**

**Goal:** Make variables comparable.

Many advanced techniques depend on scale.

**Actions**

* Apply feature scaling:

  * StandardScaler
  * MinMaxScaler

Used before:

* PCA
* Clustering
* Distance-based models

---

## **Step 4 — Dimensionality Reduction (PCA)**

**Goal:** Understand hidden structure.

**Why PCA?**

* Detect redundancy
* Reduce feature count
* Visualize high-dimensional data

**Actions**

1. Scale numeric data
2. Apply PCA
3. Plot:

   * explained variance
   * cumulative variance
4. Identify number of important components
5. Examine feature loadings

**Learn**

* Which variables move together?
* How many true dimensions exist?

---

## **Step 5 — Feature Importance Analysis**

**Goal:** Find what actually matters.

Use multiple models because importance differs by algorithm.

**Typical Models**

* Random Forest
* Gradient Boosting
* XGBoost / LightGBM

**Actions**

1. Train models
2. Extract feature importance
3. Compare across models
4. Average importance scores

**Outcome**
✅ Important variables
✅ Unimportant variables

---

## **Step 6 — Mutual Information Analysis**

**Goal:** Detect **non-linear relationships**.

Correlation only detects linear patterns.

**Actions**

* Compute Mutual Information between features and target
* Rank features
* Plot top contributors

**Learn**

* Hidden dependencies missed by correlation.

---

## **Step 7 — Clustering Analysis**

**Goal:** Discover natural groups in data.

(No labels required)

### Process

1. Select meaningful features
2. Scale data
3. Determine cluster count

   * Elbow method
4. Apply clustering

   * KMeans
   * DBSCAN
5. Visualize clusters (often via PCA)

**Outcome**

* Data segments
* Behavioral groups
* Regimes/patterns

---

## **Step 8 — Group Comparison Analysis**

**Goal:** Test if categories affect outcomes.

Example:

* region vs sales
* category vs price
* class vs performance

**Actions**

1. Group data by category
2. Compare target distributions
3. Apply statistical test:

   * ANOVA
   * t-test
4. Visualize using boxplots

**Learn**
👉 Whether categories significantly influence results.

---

## **Step 9 — Feature Correlation & Redundancy Detection**

**Goal:** Remove duplicated information.

**Actions**

1. Compute correlation matrix
2. Cluster correlated features
3. Detect highly correlated pairs (e.g., |r| > 0.9)

**Outcome**

* Reduce multicollinearity
* Simpler models
* Better generalization

---

## **Step 10 — Change / Trend Analysis**

*(Especially useful for time or ordered data)*

**Goal:** Understand rate of change.

**Actions**

* Compute differences over time
* Plot distributions
* Detect extreme events
* Compare change vs other variables

**Learn**

* Stability
* sudden shifts
* anomalies

---

## **Step 11 — Interaction Effects**

**Goal:** Study relationships between variables.

Instead of:

```
Feature → Target
```

Understand:

```
Feature A + Feature B → Target
```

**Actions**

* Create interaction features
* Compare joint effects
* Analyze correlations or importance

**Outcome**
Hidden combined effects revealed.

---

## **Step 12 — Save EDA Outputs**

Professional EDA always saves results.

Save:

* plots
* summaries
* rankings
* statistics tables

Why?
✅ reproducibility
✅ reporting
✅ model reference

---

# 🧠 Mental Model of Advanced EDA

```
Understand Data
        ↓
Clean Features
        ↓
Reduce Dimensions
        ↓
Find Important Variables
        ↓
Discover Groups
        ↓
Test Statistical Effects
        ↓
Remove Redundancy
        ↓
Study Dynamics & Interactions
```

---

# ⭐ Golden Rule for Beginners

EDA answers **three fundamental questions**:

1. **What patterns exist?**
2. **What variables matter most?**
3. **How is the data structured internally?**

## ✅ Custom EDA Functions (Your Pipeline Structure)

These are the main EDA stages you implemented:

1. `load`
2. `descriptive_stats`
3. `missing_analysis`
4. `distribution_plots`
5. `trajectory_plots`
6. `correlation_analysis`
7. `categorical_analysis`
8. `time_series_plots`
9. `wind_pressure_analysis`
10. `storm_summary`
11. `pairplot_key_features`
12. `main`

---

## ✅ External Library Functions Used (Grouped for Teaching)

### 📦 Data Loading & Handling — **pandas / numpy**

* `pd.read_parquet`
* `DataFrame.describe`
* `DataFrame.to_csv`
* `DataFrame.isnull`
* `DataFrame.sum`
* `DataFrame.sort_values`
* `DataFrame.groupby`
* `DataFrame.dropna`
* `DataFrame.copy`
* `DataFrame.astype`
* `DataFrame.apply`
* `DataFrame.value_counts`
* `DataFrame.corr`
* `DataFrame.corrwith`
* `DataFrame.select_dtypes`
* `DataFrame.round`
* `DataFrame.agg`
* `DataFrame.hist`
* `DataFrame.plot`
* `DataFrame.unique`

NumPy:

* `np.linspace`
* `np.ceil`

---

### 📊 Visualization — **matplotlib**

* `matplotlib.use`
* `plt.subplots`
* `plt.close`
* `plt.colorbar`
* `plt.cm.tab10`
* `plt.cm.tab20`

Axes methods:

* `ax.plot`
* `ax.scatter`
* `ax.hist`
* `ax.boxplot`
* `ax.barh`
* `ax.fill_between`
* `ax.axvline`
* `ax.set_title`
* `ax.set_xlabel`
* `ax.set_ylabel`
* `ax.set_xticklabels`
* `ax.set_yticks`
* `ax.set_yticklabels`
* `ax.legend`
* `ax.grid`

Figure methods:

* `fig.tight_layout`
* `fig.savefig`
* `fig.suptitle`

---

### 🎨 Statistical Visualization — **seaborn**

* `sns.set_theme`
* `sns.heatmap`
* `sns.pairplot`

---

### 📈 Statistical Analysis — **scipy**

* `scipy.stats.gaussian_kde`
* `scipy.stats.linregress`

---

### 📁 File System Handling

* `Path.resolve`
* `Path.mkdir`

---

### ⚙️ Utility / System

* `warnings.filterwarnings`
* `print`

---

## ✅ Recommended Teaching Order for Intern (EDA Flow)

You effectively built this sequence:

1. **Load data**
2. **Descriptive statistics**
3. **Missing value analysis**
4. **Univariate distributions**
5. **Spatial / trajectory analysis**
6. **Correlation analysis**
7. **Categorical analysis**
8. **Time-series behavior**
9. **Feature relationships**
10. **Entity-level summaries**
11. **Multivariate visualization**

This is actually a **production-grade EDA template**, not just beginner EDA.

---

If you want, I can next give you a **universal EDA teaching framework** distilled from this script that works for *any dataset (tabular / geo / time-series)* — ideal for onboarding interns.

### EDA structures used commonly :

Below are **only the reusable Python functions** corresponding to the **generic EDA teaching steps**, extracted and rewritten so they work for **any dataset** (domain-independent).

No project-specific logic, no storm/astronomy assumptions — just **pure reusable EDA functions**.

---

# ✅ 1. Load Dataset

```python
import pandas as pd

def load_data(path):
    df = pd.read_parquet(path) if str(path).endswith("parquet") else pd.read_csv(path)
    print("Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isna().sum())
    return df
```

---

# ✅ 2. Get Numeric Features

```python
import numpy as np

def get_numeric_features(df, min_non_null=0.5):
    num_cols = df.select_dtypes(include=[np.number]).columns
    valid_cols = [
        c for c in num_cols
        if df[c].notna().sum() > len(df) * min_non_null
    ]
    return valid_cols
```

---

# ✅ 3. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

def scale_features(df, feature_cols):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return X_scaled, scaler
```

---

# ✅ 4. PCA Analysis

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def pca_analysis(X_scaled):
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8,5))
    plt.plot(cumvar, marker="o")
    plt.axhline(0.95, linestyle="--")
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Components")
    plt.ylabel("Variance Explained")
    plt.show()

    return pca, X_pca
```

---

# ✅ 5. Feature Importance (Tree Models)

```python
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def feature_importance_rf(X, y, feature_names):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    importance.head(20).plot.barh(figsize=(8,6))
    plt.gca().invert_yaxis()
    plt.title("Random Forest Feature Importance")
    plt.show()

    return importance
```

---

# ✅ 6. Mutual Information

```python
from sklearn.feature_selection import mutual_info_regression

def mutual_information_analysis(X, y, feature_names):
    mi = mutual_info_regression(X, y, random_state=42)

    mi_series = pd.Series(mi, index=feature_names)\
        .sort_values(ascending=False)

    mi_series.head(20).plot.barh(figsize=(8,6))
    plt.gca().invert_yaxis()
    plt.title("Mutual Information")
    plt.show()

    return mi_series
```

---

# ✅ 7. Clustering Analysis

```python
from sklearn.cluster import KMeans

def clustering_analysis(X_scaled, k=4):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    plt.scatter(X_scaled[:,0], X_scaled[:,1],
                c=labels, cmap="tab10", s=10)
    plt.title("KMeans Clusters")
    plt.show()

    return labels
```

---

# ✅ 8. Category vs Target (ANOVA)

```python
from scipy import stats

def categorical_anova(df, category_col, target_col):
    groups = [
        g[target_col].dropna().values
        for _, g in df.groupby(category_col)
        if len(g) > 2
    ]

    if len(groups) < 2:
        return None

    f_stat, p_val = stats.f_oneway(*groups)

    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_val:.6f}")

    return f_stat, p_val
```

---

# ✅ 9. Correlation & Redundancy Detection

```python
import seaborn as sns

def correlation_analysis(df, feature_cols, threshold=0.9):
    corr = df[feature_cols].corr().abs()

    sns.clustermap(corr, cmap="coolwarm", figsize=(10,10))
    plt.show()

    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    high_corr = [
        (c1, c2, upper.loc[c1, c2])
        for c1 in upper.columns
        for c2 in upper.index
        if upper.loc[c1, c2] > threshold
    ]

    return high_corr
```

---

# ✅ 10. Change / Trend Analysis

```python
def change_analysis(df, column):
    delta = df[column].diff()

    plt.hist(delta.dropna(), bins=30)
    plt.axvline(0, linestyle="--")
    plt.title(f"Change Distribution: {column}")
    plt.show()

    return delta
```

---

# ✅ 11. Interaction Effect Exploration

```python
def interaction_effect(df, feature_a, feature_b, target):
    interaction = df[feature_a] * df[feature_b]

    plt.scatter(interaction, df[target], alpha=0.4)
    plt.xlabel(f"{feature_a} * {feature_b}")
    plt.ylabel(target)
    plt.title("Interaction Effect")
    plt.show()

    return interaction

```

## 1) Missingness heatmap (quick view)

```python
def missing_map(df, n=50):
    plt.imshow(df.isna().iloc[:n].T, aspect="auto")
    plt.yticks(range(df.shape[1]), df.columns); plt.title("Missingness"); plt.show()
```

## 2) Missingness % per column

```python
def missing_pct(df):
    return (df.isna().mean().sort_values(ascending=False) * 100)
```

## 3) Duplicate rows count

```python
def duplicates_info(df):
    return {"duplicates": int(df.duplicated().sum()), "rows": len(df)}
```

## 4) Cardinality of categoricals

```python
def categorical_cardinality(df):
    cat = df.select_dtypes(exclude=[np.number]).columns
    return df[cat].nunique().sort_values(ascending=False)
```

## 5) Rare category detection

```python
def rare_categories(df, col, min_frac=0.01):
    vc = df[col].value_counts(normalize=True)
    return vc[vc < min_frac]
```

## 6) Outlier flags (IQR rule)

```python
def outlier_iqr_flags(df, col, k=1.5):
    q1,q3 = df[col].quantile([0.25,0.75]); iqr=q3-q1
    return (df[col] < q1-k*iqr) | (df[col] > q3+k*iqr)
```

## 7) Winsorize (clip outliers)

```python
def winsorize_series(s, p=0.01):
    lo,hi = s.quantile([p,1-p])
    return s.clip(lo, hi)
```

## 8) Skewness check (numeric)

```python
def skewness_report(df, cols):
    return df[cols].skew(numeric_only=True).sort_values(ascending=False)
```

## 9) Log transform helper

```python
def log1p_transform(df, col):
    return np.log1p(df[col].clip(lower=0))
```

## 10) Standard Z-score anomalies

```python
def zscore_anomalies(df, col, z=3):
    v = df[col].dropna(); zs = (v - v.mean()) / v.std(ddof=0)
    return zs.abs() > z
```

## 11) Pairwise correlation with target

```python
def corr_with_target(df, feature_cols, target):
    return df[feature_cols].corrwith(df[target]).sort_values(ascending=False)
```

## 12) Partial correlation (control one variable)

```python
def partial_corr(df, x, y, z):
    rx = df[x].corr(df[z]); ry = df[y].corr(df[z]); rxy = df[x].corr(df[y])
    return (rxy - rx*ry) / np.sqrt((1-rx**2)*(1-ry**2))
```

## 13) Permutation importance (any fitted model)

```python
from sklearn.inspection import permutation_importance
def perm_importance(model, X, y, feature_names):
    r = permutation_importance(model, X, y, n_repeats=5, random_state=42)
    return pd.Series(r.importances_mean, index=feature_names).sort_values(ascending=False)
```

## 14) SHAP (concept hook; minimal)

```python
import shap
def shap_summary(model, X):
    expl = shap.Explainer(model, X); sv = expl(X)
    shap.summary_plot(sv, X, show=True)
```

## 15) VIF (multicollinearity)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
def vif_table(X_df):
    return pd.Series([variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])],
                     index=X_df.columns).sort_values(ascending=False)
```

## 16) KS test (train vs test drift for a feature)

```python
from scipy.stats import ks_2samp
def ks_drift(train, test, col):
    return ks_2samp(train[col].dropna(), test[col].dropna()).pvalue
```

## 17) PSI (population stability index)

```python
def psi(a, b, bins=10):
    qa = pd.qcut(a.dropna(), bins, duplicates="drop"); edges = qa.cat.categories
    pa = qa.value_counts(normalize=True); pb = pd.cut(b.dropna(), edges).value_counts(normalize=True)
    pb = pb.reindex(pa.index, fill_value=1e-6); pa = pa.clip(1e-6)
    return float(((pa - pb) * np.log(pa / pb)).sum())
```

## 18) Time-series rolling mean plot

```python
def rolling_mean_plot(df, time_col, value_col, win=7):
    s = df.sort_values(time_col).set_index(time_col)[value_col]
    s.rolling(win).mean().plot(); plt.title("Rolling Mean"); plt.show()
```

## 19) Seasonal decomposition (time series)

```python
from statsmodels.tsa.seasonal import seasonal_decompose
def decompose_ts(df, time_col, value_col, period):
    s = df.sort_values(time_col).set_index(time_col)[value_col].asfreq("D")
    return seasonal_decompose(s, period=period, model="additive")
```

## 20) Autocorrelation / partial autocorrelation

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def acf_pacf(df, time_col, value_col, lags=30):
    s = df.sort_values(time_col).set_index(time_col)[value_col].dropna()
    plot_acf(s, lags=lags); plt.show(); plot_pacf(s, lags=lags); plt.show()
```

## SHAP and LIME - Feature importance

Below is what this script is doing conceptually (SHAP + LIME for **regression**) and how each function implements those concepts — without tying it to any particular dataset.

---

## What SHAP and LIME are doing here

### SHAP (global + local explanations for tree models)

* **Goal:** attribute a model’s prediction to its input features using Shapley values (game theory).
* **Interpretation:** each feature gets a signed contribution (positive pushes prediction up, negative pushes down) relative to a **base value** (expected prediction).
* **Why it fits here:** you’re using a **tree-based model** (XGBoost), so SHAP can use fast exact/approx algorithms via `TreeExplainer`.

### LIME (local surrogate explanations)

* **Goal:** explain a *single prediction* by training a simple interpretable model (usually linear) on **perturbations** around that instance.
* **Interpretation:** weights from the local surrogate show which feature conditions increase/decrease the prediction near that point.
* **Why it fits here:** it’s model-agnostic — you only need `model.predict`.

---

## Module-level setup (what it enables)

* `matplotlib.use("Agg")`: renders plots in non-interactive environments (servers, pipelines) so `savefig()` works reliably.
* `OUT.mkdir(..., exist_ok=True)`: guarantees a stable output folder so every analysis step can write artifacts.
* `sns.set_theme(...)`: sets consistent plot styling (whitegrid, font scale).

---

## Function-by-function: concept + implementation

### `load()`

**Concept:** a single entry point that produces the working dataframe.

**Implementation:** reads a persisted table and prints shape so you can sanity-check that the pipeline is running on expected rows/cols.

---

### `get_features(df)`

**Concept:** define a “feature space” suitable for explainability:

* SHAP/LIME assume your features are actual model inputs.
* Excluding targets/leakage prevents explanations from being dominated by “cheating” variables.
* Dropping sparse columns reduces unstable explanations from heavy missingness.

**Implementation details:**

* `df.select_dtypes(include=[np.number])`: SHAP/LIME are simplest on numeric matrices.
* `exclude = [...]`: removes targets and obvious leakage variables.
* `df[c].notna().sum() > len(df) * 0.8`: keeps columns with at least 80% non-null.

---

### `prepare_data(df, features, target="wind")`

**Concept:** build the exact training matrix used for explanations.

* Explanations are only meaningful if they’re computed on the same `X` the model sees.
* A strict `dropna()` gives consistent arrays for scikit-learn and for SHAP/LIME.

**Implementation:**

* `df_clean = df[features + [target]].dropna()`
* `X = ...values`, `y = ...values`

---

## 1) `shap_analysis(df, features)`

This is the “global + local SHAP” block.

### A) Train a tree model (XGBoost regressor)

**Concept:** SHAP needs a trained model; TreeExplainer is optimized for tree ensembles.

**Implementation:**

* `train_test_split(... test_size=0.2, random_state=42)`
* `xgb.XGBRegressor(...)` with `n_estimators`, `max_depth`, `learning_rate`
* `model.score(...)` prints R² to verify the model learned something (otherwise explanations are noise)

### B) Compute SHAP values

**Concept:** SHAP values quantify each feature’s contribution for each prediction.

**Implementation:**

* `explainer = shap.TreeExplainer(model)`
* `shap_values = explainer.shap_values(X_test)`

  * shape is typically `(n_samples, n_features)` for regression

### C) SHAP summary plots

**Concept:** global importance + distribution:

* **Beeswarm:** shows distribution of SHAP values per feature (importance + directionality + spread).
* **Bar plot:** ranks by mean absolute SHAP value.

**Implementation:**

* `shap.summary_plot(... show=False, max_display=30)`
* saved to PNG files

### D) SHAP importance table

**Concept:** convert per-instance contributions into a global importance metric:

* `mean(|SHAP|)` per feature is a standard global importance.

**Implementation:**

* `shap_df = pd.DataFrame(shap_values, columns=features)`
* `shap_importance = shap_df.abs().mean().sort_values(ascending=False)`
* saved as CSV

### E) Top vs bottom features

**Concept:** visualize the separation between influential and negligible features (under this model).

**Implementation:**

* takes `head(25)` and `tail(25)` of `shap_importance`
* horizontal bar plots, two panels

### F) Dependence plots (manual scatter)

**Concept:** show how feature values relate to contribution magnitude and sign.

* For each top feature, plot `feature value` vs `SHAP value`.

**Implementation:**

* picks `top6`
* scatter: `ax.scatter(X_test[:, feat_idx], shap_values[:, feat_idx], ...)`
* draws `y=0` reference line for sign changes

### G) Local explanations as readable text

**Concept:** “force plot”-style logic, but exported as text:

* base value + top positive contributors + top negative contributors for selected instances.

**Implementation:**

* selects 5 indices across the test set
* for each:

  * `predicted = model.predict(...)`
  * `sv = shap_values[idx]`
  * `top_pos = np.argsort(sv)[-5:][::-1]` (largest positives)
  * `top_neg = np.argsort(sv)[:5]` (most negative)
* writes a human-readable report

Returns the trained model + split arrays so later stages reuse them.

---

## 2) `shap_interaction(model, X_test, features)`

**Concept:** SHAP interactions estimate how *pairs* of features jointly affect predictions beyond their independent effects.

**Why subsample:** interaction tensors are expensive:

* interaction values are `(n_samples, n_features, n_features)`

**Implementation:**

* `X_sample = X_test[:n_sample]`, with `n_sample = min(50, len(X_test))`
* `explainer.shap_interaction_values(X_sample)`
* `mean_interaction = np.abs(...).mean(axis=0)` gives a global interaction matrix
* enumerates pairs `(i, j)` and ranks them
* saves:

  * `shap_top_interactions.csv`
  * bar plot of top 20 interaction pairs

Wrapped in `try/except` because some setups or versions can fail for interaction computation.

---

## 3) `lime_analysis(... )`

This is “local surrogate explanations + aggregation”.

### A) Create a LIME tabular explainer (regression mode)

**Concept:** LIME builds a local neighborhood by perturbing features and fits a simple interpretable surrogate.

**Implementation:**

* `lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=features, mode="regression")`

### B) Pick representative instances

**Concept:** explain multiple points across the output range rather than one cherry-picked example.

**Implementation:**

* uses percentiles (10, 25, 50, 75, 90)
* for each percentile value, chooses closest instance by absolute difference

### C) Explain each instance

**Concept:** local linear surrogate weights approximate feature influence near that instance.

**Implementation:**

* `exp = explainer.explain_instance(X_test[idx], model.predict, num_features=len(features), num_samples=1000)`

  * `num_samples` controls neighborhood size (stability vs compute)
* `exp.as_list()` returns items like:

  * `"feature <= threshold"` and a signed weight

### D) Aggregate LIME importance

**Concept:** LIME is local; aggregation approximates a *global-ish* ranking:

* mean absolute LIME weight per feature across explained instances.

**Implementation details:**

* feature name cleaning:

  * LIME uses condition strings; the script maps them back to the base feature names
* stores `abs(weight)` per feature across instances
* `lime_agg = pd.Series({k: np.mean(v) ...}).sort_values(...)`
* saves CSV + plot

### E) Per-instance plots and text report

**Concept:** provide “human consumable” explanations:

* per-instance bar plot of weights (signed)
* plus a text file listing top contributions

**Implementation:**

* sorts by `abs(weight)` for display
* custom barh plot (positive/negative colored)
* saves one PNG per explained instance
* writes `lime_explanations.txt`

---

## 4) `compare_shap_lime()`

This is the “agreement and disagreement” layer.

### A) Normalize importances

**Concept:** SHAP and LIME are on different scales; normalization enables comparison.

**Implementation:**

* reads both CSVs
* divides each by its max → values in `[0, 1]`
* outer join and fill missing with 0

### B) Combined ranking

**Concept:** produce a consensus-ish ordering by ranking each method then averaging ranks.

**Implementation:**

* `combined["mean_rank"] = (combined.rank(ascending=False)).mean(axis=1)`
* lower mean rank = more consistently important
* saves `shap_vs_lime_comparison.csv`

### C) Scatter plot (SHAP vs LIME)

**Concept:** quick diagnostic:

* points near diagonal → agreement
* far from diagonal → method disagreement / local-vs-global effects / instability

**Implementation:**

* scatter of normalized SHAP vs normalized LIME
* labels top ~15 features by mean rank
* diagonal reference line

### D) Side-by-side bar for top 20

**Concept:** show relative strengths among top features under each method.

**Implementation:**

* barh with offsets (`xi ± width/2`)

### E) “Consensus top features”

**Concept:** robust set:

* features appearing in the top-k of both methods.

**Implementation:**

* top 30 sets from each
* intersection → consensus
* also writes SHAP-only and LIME-only lists

---

## Practical interpretation differences (what the outputs *mean*)

* **SHAP summary/bar:** “globally important features under this trained tree ensemble.”
* **SHAP dependence:** “how feature values relate to contribution size and sign (potential nonlinearity).”
* **SHAP interactions:** “pairs whose combined effect matters.”
* **LIME per-instance:** “what mattered *for this one prediction*, in a local linear approximation.”
* **LIME aggregated:** “rough global signal from a handful of local explanations (not a true global method).”
* **Comparison outputs:** “agreement = more confidence; disagreement = investigate locality, correlations, nonlinearities, and interaction effects.”

---

## Notes on how the concepts are implemented (and what to watch)

* LIME’s `num_samples=1000` trades compute for explanation stability.
* LIME condition strings are heuristically mapped back to feature names; this is why the “cleaning” loop exists.
* SHAP interaction values can be heavy; subsampling is the right conceptual move.
* Both methods can be distorted by strong collinearity; SHAP splits credit across correlated features, and LIME’s local linear fit can swap importance among correlated predictors.

If you want, I can rewrite the module docstring + add short in-code comments (only conceptual) so the script reads like a tutorial without changing behavior.

## Correlation Analysis

Below is a **pure conceptual breakdown** of what this script is doing — focusing only on the **analytical ideas, statistical concepts, and computational methods**, not on any specific dataset.

---

# 🧠 Core Purpose of the Script

This program performs a **full-spectrum Exploratory Data Analysis (EDA)** pipeline combining:

* Time-series analysis
* Statistical inference
* Multivariate analysis
* Feature diagnostics
* Pattern discovery
* Anomaly detection
* Cyclical & spatial reasoning

It transforms raw observations into **interpretable statistical structure**.

---

# 1. Storm Lifecycle Phase Analysis → *Progress Normalization*

### Underlying Concept

Many phenomena evolve through **stages**.
Instead of absolute time, the script converts progression into a **normalized lifecycle scale (0 → 1)**.

### Key Ideas

* Min–Max normalization
* Phase segmentation using binning
* Comparative aggregation across entities

### Functions Used

* `pd.cut()` → converts continuous progress into categorical phases
* `groupby().agg()` → summarizes behavior per phase
* Scatter + trend visualization

✅ Concept learned:

> Compare evolving systems independent of duration.

---

# 2. Cross-Tabulation Analysis → *Categorical Interaction Study*

### Concept

Examines relationships between **multiple categorical variables** and a numeric outcome.

### Methods

* Pivot tables
* Frequency tables
* Independence testing

### Functions

* `pivot_table()` → multidimensional aggregation
* `pd.crosstab()` → contingency matrix
* `stats.chi2_contingency()` → Chi-square test

✅ Core Question:

> Are categories statistically associated with outcome differences?

---

# 3. Anomaly Detection → *Outlier Discovery in Feature Space*

### Concept

Detect observations that behave differently from the majority in **multidimensional space**.

### Algorithm

**Isolation Forest**

* Random partitioning isolates rare points faster.
* Anomalies require fewer splits.

### Pipeline

1. Feature scaling
2. Model fitting
3. Anomaly scoring
4. Dimensional projection

### Functions

* `StandardScaler`
* `IsolationForest`
* `decision_function()`
* `PCA`

✅ Concept:

> Outliers are structurally separable observations.

---

# 4. Autocorrelation & Lag Analysis → *Temporal Memory*

### Concept

Measures how current values depend on past values.

### Autocorrelation (ACF)

Checks persistence across time delays.

[
Corr(X_t, X_{t-k})
]

### Cross-Correlation

Measures delayed interaction between two signals.

### Functions

* `acf()` → autocorrelation coefficients
* `np.correlate()` → lag relationships

✅ Insight:

> Systems often retain memory or delayed influence.

---

# 5. Day/Night (Binary Regime) Analysis → *Group Comparison*

### Concept

Split data into two regimes and test mean differences.

### Statistical Test

**Independent t-test**

[
H_0: \mu_1 = \mu_2
]

### Functions

* Boolean classification
* `stats.ttest_ind()`
* Distribution comparison

✅ Concept:

> Determine whether environmental regimes change behavior.

---

# 6. Retrograde / Directional Motion Analysis → *Sign-Based State Detection*

### Concept

Transform continuous motion into **state variables**.

Example logic:

```
negative speed → reverse state
positive speed → forward state
```

Then compare outcomes between states.

### Operations

* Logical masking
* Conditional aggregation
* Difference-of-means analysis

✅ Concept:

> Directional state changes may influence system response.

---

# 7. Hemisphere & Basin Analysis → *Spatial Segmentation*

### Concept

Partition observations geographically.

### Techniques

* Rule-based spatial classification
* Regional aggregation
* Geographic comparison

### Functions

* `np.where()`
* Row-wise classification (`apply`)
* Spatial scatter plots

✅ Concept:

> Location-dependent behavior analysis.

---

# 8. Statistical Normality Testing → *Distribution Diagnostics*

### Why Important

Many statistical models assume normal distributions.

### Tests Used

#### Shapiro–Wilk

Best for smaller samples.

#### D’Agostino–Pearson

Uses skewness + kurtosis.

### Additional Metrics

* Skewness → asymmetry
* Kurtosis → tail heaviness

### Functions

* `stats.shapiro()`
* `stats.normaltest()`
* Q-Q plots (`probplot`)

✅ Concept:

> Validate assumptions before modeling.

---

# 9. Circular / Angular Feature Analysis → *Periodic Geometry*

### Concept

Some variables lie on a **circle (0–360°)**.

Linear statistics fail here.

### Approach

* Convert degrees → radians
* Polar coordinate visualization
* Circular correlation exploration

### Functions

* `np.deg2rad()`
* Polar matplotlib projections

✅ Concept:

> Angular variables require circular reasoning.

---

# 10. Variance Inflation Factor (VIF) → *Multicollinearity Detection*

### Problem

Predictors may contain redundant information.

### Definition

[
VIF = \frac{1}{1 - R^2}
]

High VIF ⇒ feature predictable from others.

### Methods

* Correlation matrix inversion
* Regression-based fallback

### Functions

* `np.corrcoef()`
* Matrix inverse (`inv`)
* `LinearRegression`

✅ Concept:

> Remove redundant predictors to stabilize models.

---

# 11. Spectral / Periodicity Analysis → *Frequency Domain Analysis*

### Concept

Transforms signal from:

```
time domain → frequency domain
```

Detects repeating cycles.

### Method

**Periodogram**

Outputs power at each frequency.

### Function

* `periodogram()`

✅ Concept:

> Hidden oscillations become visible in frequency space.

---

# 12. Automated Summary Reporting → *Meta-EDA*

### Concept

Programmatic interpretation layer.

Creates:

* descriptive statistics
* feature counts
* missingness audit

### Functions

* aggregation
* formatted text generation
* file writing

✅ Concept:

> Convert analysis into reproducible documentation.

---

# 🔬 Overall Analytical Architecture

This script integrates **five major analytical dimensions**:

| Dimension    | Techniques               |
| ------------ | ------------------------ |
| Temporal     | lifecycle, ACF, spectral |
| Statistical  | tests, distributions     |
| Spatial      | regional segmentation    |
| Multivariate | PCA, VIF                 |
| Structural   | anomaly detection        |

---

# 🧩 Big Conceptual Takeaway

The workflow follows a mature analytical progression:

```
Structure →
Distribution →
Dependence →
Anomalies →
Redundancy →
Periodicity →
Interpretation
```

In essence, it answers:

✅ How things evolve
✅ How variables relate
✅ What is unusual
✅ What repeats
✅ What is redundant
✅ What assumptions hold

---

## Summary

This document presents a **production-grade Exploratory Data Analysis (EDA) workflow** designed to work across **any tabular dataset** (including time-series, geo/spatial, and mixed-type data). It follows a structured learning path: first establishing what data exists and whether it is usable, then progressively applying more advanced methods to uncover structure, relationships, and actionable signals.

The workflow starts with **dataset inspection and feature hygiene**—checking shape, types, missingness, duplicates, and removing non-informative or risky columns (IDs, leakage, constants). It then builds a reliable analysis foundation through **numerical scaling**, enabling techniques that depend on distance and variance.

From there, the pipeline moves into “advanced EDA” layers:

* **Dimensionality reduction (PCA)** to reveal redundancy, compress information, and visualize hidden structure.
* **Feature importance (tree models)** and **mutual information** to identify what matters—capturing both linear and non-linear signals.
* **Clustering** to discover natural groupings without labels and understand segmentation behavior.
* **Group comparison and statistical testing** (e.g., t-tests / ANOVA / chi-square) to determine whether categorical regimes meaningfully change outcomes.
* **Correlation + redundancy detection** (including VIF) to reduce multicollinearity and simplify modeling.
* **Change/trend and time-dependence tools** (differences, rolling behavior, ACF/PACF, spectral analysis) to quantify dynamics and periodic patterns.
* **Anomaly detection** (e.g., Isolation Forest + PCA visualization) to surface rare or structurally unusual observations.
* **Interaction effects** to expose combined feature influences that single-variable analysis can miss.

To support professional usage, the workflow emphasizes **reproducibility** by saving plots, rankings, and summary tables as artifacts. It also includes an explainability layer using **SHAP and LIME**: SHAP provides consistent global+local attribution (especially strong for tree models), while LIME offers local surrogate explanations; comparing both highlights agreement (confidence) and disagreement (signals to investigate, often due to correlation or locality).

Overall, the document defines a reusable EDA mental model:

**Understand → Clean → Standardize → Reduce dimensions → Rank drivers → Discover groups → Test effects → Remove redundancy → Study dynamics & interactions → Save outputs**

This turns raw data into interpretable structure and prepares a stable foundation for modeling, reporting, and onboarding new analysts.
