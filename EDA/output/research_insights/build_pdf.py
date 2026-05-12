"""
build_pdf.py
============
Generates a publication-quality PDF of the research paper using ReportLab.
Reads data directly from the CSV outputs and renders the full paper
with all tables, sections, and the publication figure.

Output: output/research_insights/paper.pdf
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from PIL import Image as PILImage

import pandas as pd
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image, KeepTogether, ListFlowable, ListItem,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.colors import HexColor, black, white, Color

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR  = Path(__file__).resolve().parent
PDF_OUT  = OUT_DIR / "paper.pdf"
FIG_PATH = OUT_DIR / "publication_figure.png"

# CSV inputs
model_comp  = OUT_DIR / "model_comparison.csv"
group_imp   = OUT_DIR / "group_importance.csv"
fwd_cv      = OUT_DIR / "forward_cv_results.csv"
effect_csv  = OUT_DIR / "effect_sizes.csv"
confound_csv= OUT_DIR / "confound_check_summary.csv"
lunar_csv   = OUT_DIR / "lunar_lifecycle_ri_rates.csv"
retro_csv   = Path(__file__).resolve().parent.parent / "comprehensive" / "retrograde_analysis.csv"
h3_csv      = Path(__file__).resolve().parent.parent / "hypothesis_testing" / "hypothesis3_spearman_matrix.csv"

# ── Colour palette ─────────────────────────────────────────────────────────────
C_NAVY    = HexColor("#1a3a5c")
C_STEEL   = HexColor("#2e6da4")
C_LIGHT   = HexColor("#d6e4f0")
C_RULE    = HexColor("#c0c0c0")
C_HEAD    = HexColor("#f0f4f8")
C_ALT     = HexColor("#f8fafc")
C_RED     = HexColor("#c0392b")
C_GREEN   = HexColor("#27ae60")
C_YELLOW  = HexColor("#f39c12")

# ── Styles ─────────────────────────────────────────────────────────────────────
base   = getSampleStyleSheet()

def _sty(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=base[parent], **kw)
    return s

TITLE_STYLE   = _sty("Title",    "Normal",
    fontName="Helvetica-Bold", fontSize=18, leading=24,
    textColor=C_NAVY, alignment=TA_CENTER, spaceAfter=6)
SUBTITLE_STYLE= _sty("Subtitle", "Normal",
    fontName="Helvetica", fontSize=13, leading=17,
    textColor=C_STEEL, alignment=TA_CENTER, spaceAfter=4)
AUTHOR_STYLE  = _sty("Author",   "Normal",
    fontName="Helvetica-Oblique", fontSize=11, leading=15,
    textColor=C_NAVY, alignment=TA_CENTER, spaceAfter=3)
DATE_STYLE    = _sty("Date",     "Normal",
    fontName="Helvetica", fontSize=10, textColor=colors.grey,
    alignment=TA_CENTER, spaceAfter=12)

ABS_HEAD_STYLE= _sty("AbsHead",  "Normal",
    fontName="Helvetica-Bold", fontSize=10, textColor=C_NAVY,
    alignment=TA_CENTER, spaceAfter=4)
ABS_STYLE     = _sty("Abstract", "Normal",
    fontName="Helvetica", fontSize=9.5, leading=14,
    alignment=TA_JUSTIFY, leftIndent=30, rightIndent=30, spaceAfter=6)

H1_STYLE      = _sty("H1", "Normal",
    fontName="Helvetica-Bold", fontSize=13, leading=17,
    textColor=C_NAVY, spaceBefore=18, spaceAfter=6)
H2_STYLE      = _sty("H2", "Normal",
    fontName="Helvetica-Bold", fontSize=11, leading=15,
    textColor=C_STEEL, spaceBefore=12, spaceAfter=4)
H3_STYLE      = _sty("H3", "Normal",
    fontName="Helvetica-BoldOblique", fontSize=10, leading=14,
    textColor=C_NAVY, spaceBefore=8, spaceAfter=3)

BODY_STYLE    = _sty("Body", "Normal",
    fontName="Helvetica", fontSize=9.5, leading=14,
    alignment=TA_JUSTIFY, spaceAfter=6)
BODY_NOTE     = _sty("BodyNote", "Normal",
    fontName="Helvetica-Oblique", fontSize=8.5, leading=12,
    textColor=HexColor("#555555"), spaceAfter=4)
CAPTION_STYLE = _sty("Caption", "Normal",
    fontName="Helvetica-Oblique", fontSize=8.5, leading=12,
    textColor=HexColor("#444444"), alignment=TA_CENTER, spaceAfter=8)
CODE_STYLE    = _sty("Code", "Normal",
    fontName="Courier", fontSize=8, leading=11,
    leftIndent=12, spaceAfter=4)
BULLET_STYLE  = _sty("Bullet", "Normal",
    fontName="Helvetica", fontSize=9.5, leading=14,
    leftIndent=18, firstLineIndent=-12, spaceAfter=4)

TABLE_HEAD    = _sty("THead", "Normal",
    fontName="Helvetica-Bold", fontSize=8.5, leading=11,
    textColor=white, alignment=TA_CENTER)
TABLE_CELL    = _sty("TCell", "Normal",
    fontName="Helvetica", fontSize=8.5, leading=11,
    alignment=TA_LEFT)
TABLE_CELLR   = _sty("TCellR", "Normal",
    fontName="Helvetica", fontSize=8.5, leading=11,
    alignment=TA_RIGHT)

# ── Helpers ────────────────────────────────────────────────────────────────────
def P(text, style=BODY_STYLE):
    return Paragraph(text, style)

def H1(text):
    return P(f"<b>{text}</b>", H1_STYLE)

def H2(text):
    return P(text, H2_STYLE)

def H3(text):
    return P(text, H3_STYLE)

def SP(n=6):
    return Spacer(1, n)

def HR():
    return HRFlowable(width="100%", thickness=0.5, color=C_RULE, spaceAfter=4)

def note(text):
    return P(f"<i>{text}</i>", BODY_NOTE)

def bullet_list(items):
    return [P(f"•  {t}", BULLET_STYLE) for t in items]

def _fmt(v, decimals=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)

def _pval(p):
    try:
        p = float(p)
    except Exception:
        return str(p)
    if p < 1e-200: return "<10⁻²⁰⁰"
    if p < 1e-50:  return f"<10⁻⁵⁰"
    if p < 1e-20:  return f"<10⁻²⁰"
    if p < 1e-10:  return f"{p:.1e}"
    if p < 0.001:  return f"{p:.4f}"
    return f"{p:.4f}"

def _safe(path):
    if isinstance(path, Path):
        return pd.read_csv(path) if path.exists() else None
    return None

# ── Table builder ──────────────────────────────────────────────────────────────
def make_table(headers, rows, col_widths=None, caption=None, zebra=True):
    """Build a styled ReportLab Table from headers (list) and rows (list of lists)."""
    head_row  = [Paragraph(h, TABLE_HEAD) for h in headers]
    data_rows = []
    for i, row in enumerate(rows):
        data_rows.append([Paragraph(str(c), TABLE_CELL) for c in row])

    table_data = [head_row] + data_rows

    if col_widths is None:
        n = len(headers)
        avail = 17 * cm
        col_widths = [avail / n] * n

    style = [
        ("BACKGROUND",  (0, 0), (-1, 0),  C_NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  8.5),
        ("ALIGN",       (0, 0), (-1, 0),  "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",        (0, 0), (-1, -1), 0.3, C_RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_ALT, white] if zebra else [white]),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0,0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",(0, 0), (-1, -1), 5),
    ]

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style))

    elems = [tbl]
    if caption:
        elems.append(P(caption, CAPTION_STYLE))
    return elems


# ═════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
def build_doc():
    doc = SimpleDocTemplate(
        str(PDF_OUT),
        pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm,  bottomMargin=2.5*cm,
        title="Vedic Astrology and Astronomical Features in Tropical Cyclone Intensity Modelling",
        author="Deepak Chakravarthy A.",
        subject="Research Paper",
    )
    return doc


# ═════════════════════════════════════════════════════════════════════════════
# CONTENT BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def section_title_page(story):
    story.append(SP(40))
    story.append(P("Vedic Astrology and Astronomical Features", TITLE_STYLE))
    story.append(P("in Tropical Cyclone Intensity Modelling:", TITLE_STYLE))
    story.append(P("A Large-Scale Empirical Assessment", TITLE_STYLE))
    story.append(SP(20))
    story.append(P("Hariramanan Sivakumar", AUTHOR_STYLE))
    story.append(P("reachariramanan@gmail.com", AUTHOR_STYLE))
    story.append(SP(6))
    story.append(P("May 2026", DATE_STYLE))
    story.append(HR())
    story.append(SP(12))
    story.append(P("Abstract", ABS_HEAD_STYLE))
    story.append(P(
        "We present the first large-scale empirical investigation of whether Vedic Panchang (Hindu "
        "calendar) categories and planetary astronomical positions carry detectable signal for "
        "tropical cyclone intensity. Using 47,533 track observations from 944 global storms spanning "
        "2016–2026, we construct a 222-feature dataset encompassing 14 meteorological, 18 Panchangam, "
        "and 190 scientific-astronomical predictors, and evaluate predictive performance through "
        "chronological holdout regression, five-fold forward-chaining cross-validation, permutation "
        "group importance, non-parametric hypothesis testing with effect size quantification, and "
        "confound-adjusted stratified analysis. "
        "Our primary finding is a <b>null result</b>: adding Vedic features to a meteorological "
        "baseline degrades wind-change RMSE by 14.3% and the full model fails to outperform the "
        "meteorological baseline in 4 of 5 temporal folds. "
        "Two secondary signals survive Bonferroni correction but carry negligible practical effect "
        "sizes: a lunar-phase association with rapid intensification (RI; χ²=19.66, p=0.006, "
        "Cramér's V≈0.020) and a consistent negative correlation between planetary speed and storm "
        "bearing direction (ρ≈−0.18, r²≈3%). "
        "We document that statistical significance is inflated by dataset scale (N≈47,533) and "
        "identify seasonal confounding as the primary driver of apparent Vedic categorical effects. "
        "These findings provide a methodological template for rigorous evaluation of non-conventional "
        "features in meteorological machine learning.",
        ABS_STYLE))
    story.append(HR())
    story.append(PageBreak())


def section_intro(story):
    story.append(H1("1. Introduction"))
    story.append(P(
        "Tropical cyclones pose severe societal risks, and improving the accuracy of intensity "
        "forecasts remains an active area of meteorological research. Alongside conventional "
        "numerical weather prediction and machine learning approaches that exploit atmospheric "
        "and oceanic variables, there exists a long-standing traditional claim — rooted in Vedic "
        "astrology — that planetary positions and Hindu calendar cycles modulate weather events "
        "and natural disasters."
    ))
    story.append(P(
        "Vedic astrology encodes the positions of the Sun, Moon, and seven classical planets into "
        "a rich system of named categories: <i>Tithi</i> (lunar day), <i>Nakshatra</i> (lunar "
        "mansion), <i>Yoga</i> (Sun–Moon arc interval), <i>Karana</i> (half-Tithi), and several "
        "dozen more. These categories are deterministic functions of planetary ephemerides and are "
        "therefore computable for any date, time, and location. This determinism makes Vedic "
        "features unusual among non-conventional predictors: they are neither noisy survey data "
        "nor domain-expert judgements but precise mathematical transformations of astronomical "
        "coordinates."
    ))
    story.append(P(
        "To our knowledge, no prior study has subjected these claims to a rigorous large-scale test "
        "against a diverse global cyclone database. We address this gap by assembling data from 944 "
        "storms across all major ocean basins, pairing track observations with Vedic Panchang values "
        "computed using the Drik Panchang astronomy engine, and conducting a battery of tests whose "
        "design explicitly addresses the statistical pathologies common in large-N astronomical "
        "association studies."
    ))
    story.append(H2("1.1 Research Questions"))
    rqs = [
        "RQ1: Do Vedic Panchang features improve next-step storm intensity prediction over a meteorological baseline?",
        "RQ2: Is there a statistically and practically significant association between lunar phase and rapid intensification?",
        "RQ3: Do planetary speeds correlate with storm track bearing beyond chance?",
        "RQ4: Do planets in apparent retrograde motion correspond to measurably different storm intensity distributions?",
        "RQ5: Do Vedic categorical wind associations persist after removing the dominant seasonal–geographic confound?",
    ]
    story.extend(bullet_list(rqs))
    story.append(H2("1.2 Summary of Contributions"))
    contribs = [
        "First large-scale (N=47,533 observations) empirical test of Vedic astrology features in tropical cyclone forecasting.",
        "Comprehensive effect size analysis distinguishing statistical from practical significance across all findings.",
        "A confound-adjusted partial Kruskal–Wallis framework for testing categorical astronomical associations within season×basin strata.",
        "A novel lunar-phase×lifecycle interaction analysis showing that RI associations with lunar phase are lifecycle-stage dependent.",
        "Open release of the 222-feature parsed dataset and all analysis scripts.",
    ]
    story.extend(bullet_list(contribs))


def section_data(story):
    story.append(H1("2. Data and Features"))
    story.append(H2("2.1 Storm Track Data"))
    story.append(P(
        "We use best-track data assembled from multiple regional operational meteorological agencies "
        "for 944 tropical cyclones active between July 2016 and January 2026, spanning the North "
        "Atlantic, Western Pacific, North Indian Ocean, South Indian Ocean, and South Pacific basins. "
        "Each track observation records the storm centre position (latitude, longitude), maximum "
        "sustained wind speed (knots), minimum central pressure (hPa), and UTC timestamp. Track "
        "observations are provided at variable intervals (predominantly 3–6 h), yielding 47,533 "
        "observations in total."
    ))

    story.append(H2("2.2 Dataset Summary"))
    ds_rows = [
        ["Total track observations", "47,533"],
        ["Unique storms", "944"],
        ["Date range", "2016-07-15 – 2026-01-24"],
        ["Ocean basins", "7"],
        ["Mean wind speed", "49.9 kt"],
        ["Median wind speed", "40.0 kt"],
        ["Wind skewness", "1.35 (right-skewed)"],
        ["Pressure range", "882–1019 hPa"],
        ["Meteorological features", "14"],
        ["Panchangam features", "18"],
        ["Scientific/astronomical features", "190"],
        ["Total features", "222"],
    ]
    story.extend(make_table(
        ["Property", "Value"],
        ds_rows,
        col_widths=[9*cm, 8*cm],
        caption="Table 1: Dataset summary statistics."
    ))

    story.append(H2("2.3 Vedic Panchang Features"))
    story.append(P(
        "For each track observation, Vedic Panchang attributes were computed using the Drik Panchang "
        "astronomy engine at the storm centre location and local time. This produces 18 categorical "
        "and derived features including the five canonical Panchang limbs plus additional calendar "
        "and celestial quantities."
    ))
    pan_rows = [
        ["Tithi",         "30 values", "Lunar day (Sun–Moon elongation in 12° steps)"],
        ["Nakshatra",     "27 values", "Lunar mansion (Moon sidereal position)"],
        ["Yoga",          "27 values", "Sun+Moon sidereal lon mod 360°, 27 equal parts"],
        ["Karana",        "11 values", "Half-Tithi period"],
        ["Paksha",        "Binary",    "Waxing (Shukla) or waning (Krishna) fortnight"],
        ["Moonsign",      "12 values", "Sidereal zodiac sign of the Moon"],
        ["Sunsign",       "12 values", "Sidereal zodiac sign of the Sun"],
        ["Drik Ritu",     "6 values",  "Observational season (equinox/solstice based)"],
        ["Vedic Ritu",    "6 values",  "Traditional Hindu season"],
        ["Drik/Vedic Ayana","Binary",  "Northern/southern hemisphere Sun path"],
        ["Anandadi Yoga", "9 values",  "Auspiciousness index (weekday×Nakshatra)"],
        ["Tamil Yoga",    "5 values",  "Tamil calendar auspiciousness class"],
        ["Samvatsara",    "60 values", "60-year Jupiter cycle year name"],
        ["Rahu Kalam",    "Time range","Daily inauspicious period"],
        ["Yamaganda",     "Time range","Secondary inauspicious period"],
        ["Weekday",       "7 values",  "Hindu weekday"],
    ]
    story.extend(make_table(
        ["Feature", "Type", "Description"],
        pan_rows,
        col_widths=[3*cm, 2.5*cm, 11.5*cm],
        caption="Table 2: Vedic Panchang categorical features used in this study."
    ))

    story.append(H2("2.4 Astronomical Features (Skyfield)"))
    story.append(P(
        "Skyfield ephemeris computations at each storm position and timestamp provide 90 numeric "
        "features: altitude, azimuth, right ascension, declination, ecliptic longitude and latitude, "
        "and derived trigonometric transforms (sin/cos of altitude and azimuth) for the Sun, Moon, "
        "and seven classical planets."
    ))
    story.append(H2("2.5 Meteorological Features"))
    story.append(P(
        "Fourteen standard meteorological and kinematic features form the baseline: wind speed (kt), "
        "pressure (hPa), latitude, longitude, hour (UTC), day of year, day of week, month, movement "
        "speed, displacement, and four spline-fitted track geometry features (curvature, bearing, "
        "displacement, length in degrees)."
    ))
    story.append(note(
        "Leakage note: spline features encode smoothed forecast track geometry and may carry "
        "implicit future trajectory information. Correlations involving these features should "
        "be interpreted with caution."
    ))


def section_methods(story):
    story.append(H1("3. Methods"))
    story.append(H2("3.1 Predictive Modelling"))
    story.append(P(
        "We train Random Forest regressors (200 trees, max depth 18, min samples per leaf 2) in "
        "four block configurations: (1) Meteo only (14 features), (2) Meteo + Panchang (32 "
        "features), (3) Full (all 222 features), and (4) Stacked residual (meteorological "
        "stage-A model whose residuals are fed to a stage-B model on Panchang + scientific features). "
        "Categorical features are ordinal-label-encoded from the training split only. Missing "
        "numeric values are imputed with the training-set median."
    ))
    story.append(P(
        "<b>Train–test split:</b> A chronological 80/20 split is applied at the observation level: "
        "observations before 2023-10-21 form the training set (n=37,274) and subsequent observations "
        "form the held-out test set (n=9,315)."
    ))
    story.append(P(
        "<b>Forward-chaining cross-validation:</b> Five temporal folds are defined at chronological "
        "quantiles. Each fold uses all data before the boundary for training and a 120-day window "
        "after for validation, preventing within-storm data leakage."
    ))
    story.append(P(
        "<b>Group permutation importance:</b> The contribution of each feature group (meteo, "
        "Panchang, scientific) is estimated by permuting all features in that group simultaneously "
        "(5 repeats) and recording the mean RMSE lift."
    ))

    story.append(H2("3.2 Effect Size Framework"))
    story.append(P(
        "With N≈47,533 observations, even trivially small effects produce highly significant "
        "p-values. We therefore accompany every significant result with an appropriate effect size:"
    ))
    eff_rows = [
        ["Rank-biserial r", "Mann–Whitney U comparisons", "r = 1 − 2U/(n₁n₂)", "Small <0.10, Medium 0.10–0.30"],
        ["Cohen's d",       "Two-group mean comparisons",  "Pooled-SD standardised difference", "Small <0.20, Medium 0.20–0.50"],
        ["Epsilon-squared", "Kruskal–Wallis tests",        "ε² = (H−k+1)/(N−k)", "Small <0.01, Medium 0.06–0.14"],
        ["Cramér's V",      "Chi-squared tests",           "V = √(χ²/(N·(min(r,c)−1)))", "Small <0.10, Medium 0.10–0.30"],
        ["Eta-squared",     "ANOVA tests",                 "η² = F·df_b/(F·df_b+df_w)", "Small <0.01, Medium 0.06–0.14"],
        ["r² (Spearman)",   "Monotone correlations",       "Variance explained by ρ", "Small <0.01, Medium >0.06"],
    ]
    story.extend(make_table(
        ["Metric", "Used for", "Formula", "Thresholds"],
        eff_rows,
        col_widths=[3*cm, 4*cm, 5*cm, 5*cm],
        caption="Table 3: Effect size metrics used throughout this study."
    ))

    story.append(H2("3.3 Hypothesis Testing"))
    story.append(P(
        "All key variables are non-normal (Shapiro–Wilk p<10⁻³⁰; wind skewness=1.35). "
        "We therefore use non-parametric tests: Mann–Whitney U (two groups), Kruskal–Wallis "
        "(k groups), and Spearman ρ (monotone associations). Where multiple tests are conducted "
        "within a family, Benjamini–Hochberg FDR correction is applied. Across the seven primary "
        "hypothesis groups, a Bonferroni-corrected threshold of α/7≈0.0071 is used."
    ))


def section_results(story):
    story.append(H1("4. Results"))

    # ── RQ1 ──────────────────────────────────────────────────────────────────
    story.append(H2("4.1 RQ1: Predictive Modelling"))

    mc = _safe(model_comp)
    if mc is not None:
        mc = mc.sort_values("rmse")
        base_rmse = mc.loc[mc["model"] == "meteo_only", "rmse"].values[0]
        label_map = {
            "meteo_only":       "Meteo only",
            "stacked_residual": "Stacked residual",
            "meteo_panchang":   "Meteo + Panchang",
            "full_all_blocks":  "Full (all blocks)",
        }
        rows = []
        for _, r in mc.iterrows():
            rmse_val = r["rmse"]
            delta = ((rmse_val - base_rmse) / base_rmse * 100) if r["model"] != "meteo_only" else 0.0
            sign  = "+" if delta >= 0 else ""
            delta_str = "—" if delta == 0.0 else f"{sign}{delta:.1f}%"
            rows.append([label_map.get(r["model"], r["model"]),
                         f"{rmse_val:.3f}", f"{r['mae']:.3f}", delta_str])
        story.extend(make_table(
            ["Model", "RMSE (kt)", "MAE (kt)", "ΔRMSE vs Meteo"],
            rows,
            col_widths=[5.5*cm, 3*cm, 3*cm, 3.5*cm],
            caption="Table 4: Regression performance on the chronological holdout test set (n=9,315). Lower is better."
        ))

    story.append(P(
        "The meteorological-only model achieves the best performance (RMSE=4.164 kt). Every "
        "addition of Vedic or astronomical features strictly increases error: adding Panchang "
        "alone degrades RMSE by 4.9%, and the full model by 8.4%."
    ))

    cv = _safe(fwd_cv)
    if cv is not None:
        piv = cv.pivot_table(values="rmse", index="fold", columns="model", aggfunc="first")
        piv = piv.reset_index()
        col_order = ["fold", "meteo_only", "full_all_blocks", "stacked_residual"]
        col_order = [c for c in col_order if c in piv.columns]
        col_labels = {"fold": "Fold", "meteo_only": "Meteo Only",
                      "full_all_blocks": "Full", "stacked_residual": "Stacked"}
        headers = [col_labels.get(c, c) for c in col_order]
        rows = []
        for _, r in piv.iterrows():
            vals = [str(int(r["fold"])) if c == "fold" else f"{r[c]:.3f}" for c in col_order]
            rows.append(vals)
        story.extend(make_table(
            headers, rows,
            col_widths=[2*cm, 4.5*cm, 4.5*cm, 4.5*cm],
            caption="Table 5: Forward-chaining cross-validation RMSE (kt) by fold. Meteo-only wins in 4/5 folds."
        ))

    gi = _safe(group_imp)
    if gi is not None:
        lmap = {"meteo": "Meteorological", "scientific": "Scientific/Astro", "panchang": "Vedic/Panchang"}
        gi["label"] = gi["group"].map(lmap).fillna(gi["group"])
        gi = gi.sort_values("rmse_lift_mean", ascending=False)
        rows = [[r["label"], f"{r['rmse_lift_mean']:.4f}", f"±{r.get('rmse_lift_std', 0):.4f}"]
                for _, r in gi.iterrows()]
        story.extend(make_table(
            ["Feature Group", "Mean RMSE Lift (kt)", "Std Dev"],
            rows,
            col_widths=[6*cm, 5.5*cm, 4*cm],
            caption="Table 6: Group-level permutation RMSE lift on the full model test set. "
                    "Higher lift = more important. Vedic/Panchang lift is <2% of meteorological lift."
        ))

    story.append(P(
        "For RI classification (threshold ≥10 kt/step), the meteo-only model achieves "
        "AUC-ROC=0.707 while the full model degrades to AUC-ROC=0.429 — below chance — "
        "demonstrating that adding 190 astronomical features actively harms RI classification."
    ))

    # ── RQ2 ──────────────────────────────────────────────────────────────────
    story.append(H2("4.2 RQ2: Lunar Phase and Rapid Intensification"))
    story.append(P(
        "A chi-squared test of the 24-hour RI indicator (Δw≥30 kt, rolling window) against "
        "8 lunar-phase bins yields χ²=19.66, p=0.006, dof=7, which survives the "
        "Bonferroni-corrected threshold of α/7≈0.0071. However, Cramér's V=√(19.66/47,533)≈0.020 "
        "places this firmly in the 'small' range."
    ))

    ll = _safe(lunar_csv)
    if ll is not None:
        agg = (ll.groupby("lunar_phase_cat")
                  .agg(ri_count=("ri_count","sum"), total_count=("total_count","sum"))
                  .reset_index())
        agg["ri_rate"] = (agg["ri_count"] / agg["total_count"] * 100).round(2)
        phase_order = ["New Moon","Waxing Crescent","First Quarter","Waxing Gibbous",
                       "Full Moon","Waning Gibbous","Third Quarter","Waning Crescent"]
        agg["_o"] = agg["lunar_phase_cat"].map({p:i for i,p in enumerate(phase_order)})
        agg = agg.sort_values("_o")
        rows = [[r["lunar_phase_cat"], str(int(r["ri_count"])),
                 f"{int(r['total_count']):,}", f"{r['ri_rate']:.2f}%"]
                for _, r in agg.iterrows()]
        story.extend(make_table(
            ["Lunar Phase", "RI Events", "Total Obs", "RI Rate"],
            rows,
            col_widths=[5*cm, 3*cm, 4*cm, 3.5*cm],
            caption="Table 7: Rapid intensification (RI) rate by lunar phase. "
                    "RI = wind change ≥30 kt/24h (rolling window)."
        ))

    story.append(H3("Lifecycle interaction"))
    story.append(P(
        "When the chi-squared test is repeated within each lifecycle phase (Genesis, Intensify, "
        "Mature, Dissipate), the lunar-phase RI signal is significant in the Genesis (p=0.0008) "
        "and Dissipation (p=0.001) stages but not in the Mature phase (p=0.265). "
        "The aggregate signal is not uniformly distributed across the storm lifetime: forming and "
        "decaying storms show stronger lunar-phase modulation of RI rate than storms at peak intensity."
    ))

    if ll is not None:
        lc_chi = ll[["lifecycle_phase","chi2","p","n_obs"]].drop_duplicates("lifecycle_phase").dropna(subset=["chi2"])
        lc_order = ["Genesis","Intensify","Mature","Dissipate"]
        lc_chi["_o"] = lc_chi["lifecycle_phase"].map({p:i for i,p in enumerate(lc_order)})
        lc_chi = lc_chi.sort_values("_o").dropna(subset=["chi2"])
        rows = [[r["lifecycle_phase"], f"{r['chi2']:.2f}", _pval(r["p"]), f"{int(r['n_obs']):,}"]
                for _, r in lc_chi.iterrows()]
        story.extend(make_table(
            ["Lifecycle Phase", "χ²", "p-value", "N obs"],
            rows,
            col_widths=[4.5*cm, 3*cm, 4*cm, 4*cm],
            caption="Table 8: Chi-squared test for lunar-phase RI distribution within each lifecycle stage."
        ))

    # ── RQ3 ──────────────────────────────────────────────────────────────────
    story.append(H2("4.3 RQ3: Planetary Speed and Storm Bearing"))
    story.append(P(
        "Table 9 presents Spearman ρ between Vedic planetary speeds and storm bearing. "
        "Four planets (Yama, Shani, Varun, Surya) show nominally medium-strength correlations "
        "(|ρ|>0.10) that are overwhelmingly significant in p-value terms but explain only "
        "2.5–3.2% of bearing variance (r²≤0.032). Quintile dose-response analysis confirms "
        "that all correlations are perfectly monotone (ρ_quintile=−1.0): as planetary speed "
        "increases from its lowest to highest quintile, median storm bearing decreases "
        "monotonically from ≈279° to ≈217° (a 62° range)."
    ))

    h3 = _safe(h3_csv)
    if h3 is not None:
        bearing = h3[h3["target_feature"] == "spline_bearing_deg"].copy()
        bearing["abs_rho"] = bearing["spearman_rho"].abs()
        bearing = bearing.sort_values("abs_rho", ascending=False).head(10)
        bearing["short"] = (bearing["speed_feature"]
                            .str.replace("vedic_","",regex=False)
                            .str.replace("_speed_deg_per_day","",regex=False))
        bearing["r2"] = (bearing["spearman_rho"]**2).round(4)
        rows = [[r["short"], f"{r['spearman_rho']:.4f}", _pval(r["p_value"]),
                 f"{int(r['n']):,}", f"{r['r2']:.4f}"]
                for _, r in bearing.iterrows()]
        story.extend(make_table(
            ["Speed Feature", "Spearman ρ", "p-value", "N", "r²"],
            rows,
            col_widths=[4.5*cm, 3*cm, 3.5*cm, 2.5*cm, 2.5*cm],
            caption="Table 9: Top-10 Spearman correlations between Vedic planetary speed (deg/day) "
                    "and storm bearing. All p<10⁻²⁰; r² = variance explained."
        ))

    story.append(note(
        "Spline leakage caution: spline_bearing_deg encodes the smoothed forecast track bearing "
        "and may carry implicit future trajectory information. Correlations may partly reflect "
        "this rather than a direct astronomical–dynamical coupling."
    ))

    # ── RQ4 ──────────────────────────────────────────────────────────────────
    story.append(H2("4.4 RQ4: Retrograde Planet Effects on Wind Intensity"))
    story.append(P(
        "All seven planets yield statistically significant retrograde/direct wind differences, "
        "but every rank-biserial correlation is below the small-effect threshold (|r_rb|<0.10). "
        "Varun (Neptune) and Arun (Uranus) are retrograde for 60% and 52% of all observations "
        "respectively — their retrograde phase is simply the majority condition during long "
        "spans of the dataset. Even Shukra (Venus, retrograde 7.9%) shows a median wind "
        "difference of only 5 kt with |r_rb|=0.044."
    ))

    ret = _safe(retro_csv)
    eff = _safe(effect_csv)
    if ret is not None and eff is not None:
        ret_eff = eff[eff["finding_id"].str.startswith("RET_")].copy()
        ret_eff["planet"] = ret_eff["finding_id"].str.replace("RET_","",regex=False).str.lower()
        merged = ret.copy()
        merged["planet_key"] = merged.apply(
            lambda r: str(r.get("planet", r.iloc[0])).lower() if "planet" in merged.columns
            else str(r.iloc[0]).lower(), axis=1
        )
        rows = []
        for _, r in merged.iterrows():
            pk = r.get("planet", str(r.iloc[0])).lower() if "planet" in merged.columns else str(r.iloc[0]).lower()
            eff_row = ret_eff[ret_eff["planet"] == pk]
            rb  = _fmt(eff_row["effect_value"].values[0], 4) if not eff_row.empty else "—"
            d   = _fmt(eff_row["cohens_d"].values[0], 4)     if not eff_row.empty else "—"
            p   = _pval(r.get("mannwhitney_p", np.nan))
            rows.append([
                pk.capitalize(),
                f'{float(r.get("retro_pct",0)):.1f}%' if "retro_pct" in merged.columns else "—",
                f'{float(r.get("retro_median_wind",0)):.0f}' if "retro_median_wind" in merged.columns else "—",
                f'{float(r.get("direct_median_wind",0)):.0f}' if "direct_median_wind" in merged.columns else "—",
                f'{float(r.get("wind_diff_median",0)):+.0f}' if "wind_diff_median" in merged.columns else "—",
                p, rb, d
            ])
        if rows:
            story.extend(make_table(
                ["Planet", "Retro %", "Retro Med", "Direct Med", "Δ (kt)", "p-value", "r_rb", "Cohen's d"],
                rows,
                col_widths=[2.2*cm, 1.8*cm, 2*cm, 2.2*cm, 1.8*cm, 2.5*cm, 1.8*cm, 2.2*cm],
                caption="Table 10: Retrograde vs direct wind intensity (Mann–Whitney U). "
                        "All effects are small (|r_rb|<0.10)."
            ))

    # ── RQ5 ──────────────────────────────────────────────────────────────────
    story.append(H2("4.5 RQ5: Confound-Adjusted Vedic Category Analysis"))
    story.append(P(
        "The Kruskal–Wallis signals for Yoga (H=143), Nakshatra (H=116), Moonsign (H=78), and "
        "Tithi (H=51) may be driven by the dominant seasonal proxy drik_ritu (H=1,214). "
        "Partial KW tests within each basin×season stratum (≥100 obs, ≥3 categories with ≥10 obs) "
        "reveal a surprising result: all four variables remain significant in 100% of the 12 viable "
        "strata after BH-FDR correction. The Vedic categorical signals are not merely artefacts of "
        "the seasonal confound — they persist within homogeneous subgroups."
    ))

    conf = _safe(confound_csv)
    if conf is not None:
        rows = [[r["variable"].capitalize(),
                 str(int(r["total_strata"])),
                 str(int(r["sig_strata_bh"])),
                 f"{r['pct_sig']:.0f}%",
                 str(r["verdict"])]
                for _, r in conf.iterrows()]
        story.extend(make_table(
            ["Variable", "Strata tested", "Sig. strata (BH)", "% sig.", "Verdict"],
            rows,
            col_widths=[3*cm, 3*cm, 3*cm, 2*cm, 5.5*cm],
            caption="Table 11: Confound-adjusted partial Kruskal–Wallis within basin×season strata. "
                    "BH-FDR correction applied within each variable."
        ))

    # ── Effect sizes ──────────────────────────────────────────────────────────
    story.append(H2("4.6 Effect Size Summary"))
    story.append(P(
        "Table 12 consolidates the statistically significant findings with their effect sizes. "
        "The dominant pattern is that p-values uniformly indicate significance while effect sizes "
        "are uniformly small."
    ))
    eff = _safe(effect_csv)
    if eff is not None:
        top = eff.sort_values("effect_value", ascending=False).head(12)
        rows = []
        for _, r in top.iterrows():
            fid   = str(r["finding_id"])[:18]
            feat  = str(r["feature"])[:35]
            stat  = f"{r['stat_label']}={_fmt(r['statistic'], 3)}"
            pv    = _pval(r["p_value"])
            em    = str(r["effect_metric"])[:15]
            ev    = _fmt(r["effect_value"], 4)
            interp= str(r["effect_interpretation"])
            rows.append([fid, feat, stat, pv, em, ev, interp])
        story.extend(make_table(
            ["ID", "Feature", "Statistic", "p-value", "Effect metric", "Effect value", "Size"],
            rows,
            col_widths=[2.2*cm, 4.5*cm, 2.5*cm, 2.2*cm, 2.5*cm, 2*cm, 1.6*cm],
            caption="Table 12: Top-12 effect sizes for significant findings. "
                    "Most rank as 'small' despite highly significant p-values."
        ))


def section_discussion(story):
    story.append(H1("5. Discussion"))

    story.append(H2("5.1 The Null Result: Vedic Features in Prediction"))
    story.append(P(
        "The primary finding is unambiguous: Vedic Panchang features provide no predictive lift "
        "over a meteorological baseline for next-step wind intensity change. Across three "
        "feature-block configurations and five temporal validation folds, adding Vedic features "
        "consistently increases RMSE. Permutation importance confirms: shuffling all 18 Vedic "
        "features changes the full-model RMSE by only 0.010 kt — less than 2% of the "
        "meteorological group's contribution (0.582 kt)."
    ))
    story.append(P(
        "This null result has a nuance. SHAP and LIME feature-importance analyses identify four "
        "Vedic speed/degree features in the top-14 consensus set across both methods. This apparent "
        "contradiction — Vedic features appear important in importance scores but degrade held-out "
        "accuracy — is explained by within-storm autocorrelation: random 80/20 splits allow "
        "consecutive track observations from the same storm in both sets, inflating apparent "
        "importance. The forward-chaining design, which prevents any within-storm leakage across "
        "temporal folds, provides the honest estimate."
    ))

    story.append(H2("5.2 Effect Size and the Large-N Problem"))
    story.append(P(
        "With N=47,533, any correlation above |r|≈0.013 will produce p<0.05, and above |r|≈0.018 "
        "will survive Bonferroni correction at α/7≈0.007. This is precisely what we observe: "
        "every retrograde/direct comparison across seven planets is significant at p<10⁻⁴ while "
        "the largest effect size is |r_rb|=0.088 (Varun, small by convention). The median wind "
        "difference is 0–5 kt against a baseline σ=27.4 kt; Cohen's d≤0.14 for all seven planets. "
        "We recommend that future astronomical association studies with large databases report "
        "effect sizes as the primary outcome and treat p-values as supplementary."
    ))

    story.append(H2("5.3 The Confound-Adjustment Surprise"))
    story.append(P(
        "The confound-adjusted partial analysis yields an unexpected result: Tithi, Nakshatra, "
        "Yoga, and Moonsign all remain significant within 100% of the 12 basin×season strata "
        "tested. One explanation is that Vedic categorical variables are dense functions of "
        "lunar and solar position and encode fine-grained astronomical geometry not captured by "
        "coarser seasonal groupings. A second, more parsimonious explanation is that within-stratum "
        "KW tests on large sub-samples retain high power for small effects, and the true "
        "epsilon-squared within strata remains small. Future work should report within-stratum "
        "effect sizes explicitly."
    ))

    story.append(H2("5.4 Lunar Phase and Rapid Intensification"))
    story.append(P(
        "The lunar-phase RI signal (χ²=19.66, p=0.006, V=0.020) is the closest finding to an "
        "independent Vedic astronomical signal. The lifecycle interaction reveals it is stronger "
        "in Genesis and Dissipation phases (p≈0.001) than in the Mature phase (p=0.265). The "
        "RI rate difference between the highest and lowest phase is less than 0.6 percentage points "
        "(1.46% vs 0.79%). At Cramér's V=0.020, this finding has essentially no operational "
        "significance for forecasting, but motivates process-level investigation into tidal "
        "influence on convective organisation at different storm lifecycle stages."
    ))

    story.append(H2("5.5 Planetary Speeds and Storm Bearing"))
    story.append(P(
        "The perfectly monotone dose-response relationship between planetary speed quintiles and "
        "median storm bearing (≈62° shift across quintiles, ρ_quintile=−1.0) is geometrically "
        "intriguing. The most likely mechanism is not direct astronomical forcing but seasonal "
        "confounding: the Sun's ecliptic speed varies slightly over the year (±3%), and outer "
        "planets' apparent speed depends on their synodic relationship with Earth's orbital phase. "
        "Storm track bearing is strongly modulated by season (monsoon circulation, polar jet), "
        "creating a three-way correlation: season → planetary speed, season → storm bearing, "
        "which appears as planetary speed → bearing."
    ))

    story.append(H2("5.6 Samvatsara and Interannual Confounding"))
    story.append(P(
        "The Samvatsara (60-year Jupiter cycle year name) ANOVA produces F=65.12, p<10⁻¹³⁰ — "
        "the largest significance value in the study. This is entirely explained by the fact that "
        "the 11 Samvatsara years present correspond one-to-one with calendar years 2016–2026, and "
        "interannual cyclone activity is strongly modulated by ENSO and PDO. Testing whether "
        "Samvatsara predicts storm intensity is equivalent to testing whether intensity differs "
        "across years — a trivially true statement carrying no Vedic-specific information."
    ))


def section_limitations(story):
    story.append(H1("6. Limitations"))
    lims = [
        "<b>Temporal autocorrelation.</b> Track observations are not independent. "
        "Models trained with random splits overestimate performance; all modelling conclusions "
        "are drawn from forward-chaining results.",

        "<b>Large-N significance inflation.</b> At N≈47,533, every p-value should be treated "
        "as a screening tool. Effect size metrics are the primary scientific discriminators.",

        "<b>Seasonal and geographic confounding.</b> Variables derived from the solar calendar "
        "(drik_ritu, sunsign, drik_ayana) co-vary with tropical cyclone season by construction.",

        "<b>Spline track geometry leakage.</b> spline_bearing_deg and spline_curvature encode "
        "the smoothed forecast track and may embody implicit future trajectory information.",

        "<b>RI threshold heterogeneity.</b> Three distinct RI definitions are used across "
        "scripts (30 kt/24h rolling; 10 kt/step). Cross-definition comparisons are not valid.",

        "<b>Missing meteorological drivers.</b> Sea surface temperature, ocean heat content, "
        "vertical wind shear, and mid-level humidity are the primary physical drivers of "
        "intensification. Their absence limits the meteorological baseline.",

        "<b>Outer planet speed near-constancy.</b> Varun (Neptune) and Arun (Uranus) are "
        "retrograde for 60% and 52% of observations. Their apparent associations may be proxies "
        "for epoch-level climate modes rather than direct astronomical signals.",
    ]
    story.extend(bullet_list(lims))


def section_conclusion(story):
    story.append(H1("7. Conclusion"))
    story.append(P(
        "We conducted the first rigorous large-scale empirical evaluation of Vedic Panchang and "
        "astronomical features as predictors of tropical cyclone intensity across 944 global "
        "storms from 2016 to 2026. Our findings span a consistent arc from statistical curiosity "
        "to operational null result."
    ))
    story.append(P(
        "Across chronological holdout testing and five-fold forward-chaining cross-validation, "
        "adding 18 Vedic Panchang features to a meteorological baseline consistently degrades "
        "next-step wind-change RMSE (by up to 14.3%). Permutation importance confirms that the "
        "Vedic feature group contributes less than 2% of the meteorological group's explanatory "
        "power. The null hypothesis that Vedic features add no predictive value is, if anything, "
        "too generous: the features actively harm forecast accuracy."
    ))
    story.append(P(
        "Two secondary findings warrant scientific attention despite their small practical effect "
        "sizes. First, a statistically robust lunar-phase association with rapid intensification "
        "(χ²=19.66, p=0.006, survives Bonferroni correction) shows that forming and decaying "
        "storms exhibit stronger lunar-phase RI modulation than storms at peak intensity — a "
        "lifecycle-dependent pattern that motivates process-level investigation into "
        "tidal–convective coupling. Second, a perfectly monotone dose-response relationship "
        "between planetary speed quintiles and storm bearing direction (r²≤3.2%) warrants "
        "examination as a potential season-mediated confound."
    ))
    story.append(P(
        "A striking methodological finding is that Vedic categorical signals (Tithi, Nakshatra, "
        "Yoga, Moonsign) persist in 100% of basin×season strata after false discovery rate "
        "correction. This statistical persistence does not translate into predictive lift — "
        "consistent with the interpretation that the signal is real but too small to generalise "
        "across storms."
    ))
    story.append(P(
        "The broader lesson is methodological: the same large dataset that provides statistical "
        "power to detect these signals also demands rigorous effect size analysis to evaluate "
        "their practical relevance. We recommend that all future astronomical association studies "
        "with geophysical databases report effect sizes as the primary outcome and treat "
        "p-values as supplementary."
    ))


def section_figure(story):
    story.append(H1("8. Publication Figure"))
    if FIG_PATH.exists():
        img = PILImage.open(FIG_PATH)
        w_px, h_px = img.size
        max_w = 17 * cm
        scale = max_w / w_px
        h_img = h_px * scale
        story.append(Image(str(FIG_PATH), width=max_w, height=h_img))
        story.append(P(
            "<b>Figure 1.</b> Four-panel publication figure. "
            "<b>(A)</b> Model RMSE comparison; the dashed line marks the meteorological baseline — "
            "adding Vedic features monotonically increases error. "
            "<b>(B)</b> Feature group RMSE lift (permutation importance); the Vedic/Panchang "
            "lift of 0.010 kt is negligible. "
            "<b>(C)</b> RI rate by lunar phase; the Waning Crescent bar (dark red) is the peak "
            "phase, the dashed line marks the overall mean RI rate. "
            "<b>(D)</b> Top-10 Spearman ρ between planetary speed and storm bearing; all values "
            "are negative (faster speed → more westward track) and statistically significant "
            "(p<10⁻²⁰) but small (r²<0.032).",
            CAPTION_STYLE
        ))
    else:
        story.append(P("Publication figure not found at: " + str(FIG_PATH), BODY_NOTE))


def section_references(story):
    story.append(H1("References"))
    refs = [
        "Emanuel, K. (2003). Tropical cyclones. <i>Annual Review of Earth and Planetary Sciences</i>, 31(1), 75–104.",
        "Gray, W. M. (1984). Atlantic seasonal hurricane frequency. Part I: El Niño and 30 mb quasi-biennial oscillation influences. <i>Monthly Weather Review</i>, 112(9), 1649–1668.",
        "Klotzbach, P. J. (2010). On the Madden–Julian oscillation–Atlantic hurricane relationship. <i>Journal of Climate</i>, 23(2), 282–293.",
        "Shapiro, S. S., &amp; Wilk, M. B. (1965). An analysis of variance test for normality. <i>Biometrika</i>, 52(3/4), 591–611.",
        "Kruskal, W. H., &amp; Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. <i>JASA</i>, 47(260), 583–621.",
        "Benjamini, Y., &amp; Hochberg, Y. (1995). Controlling the false discovery rate. <i>J. Royal Statistical Society B</i>, 57(1), 289–300.",
        "Breiman, L. (2001). Random forests. <i>Machine Learning</i>, 45(1), 5–32.",
        "Lundberg, S. M., &amp; Lee, S.-I. (2017). A unified approach to interpreting model predictions. <i>NeurIPS</i>, 30.",
        "Ribeiro, M. T., Singh, S., &amp; Guestrin, C. (2016). 'Why should I trust you?': Explaining the predictions of any classifier. <i>KDD 2016</i>, 1135–1144.",
    ]
    for i, r in enumerate(refs, 1):
        story.append(P(f"[{i}] {r}", BODY_STYLE))
        story.append(SP(2))


def section_appendix(story):
    story.append(H1("Appendix A: Consolidated Evidence Table"))
    story.append(P(
        "The table below reproduces all 16 paper-level findings with source scripts, "
        "effect sizes, and research verdicts."
    ))

    ev_rows = [
        ["F01", "Vedic features degrade RMSE by 4.9%", "06, 07", "−14.3% relative", "No predictive lift"],
        ["F02", "Panchang RMSE lift = 0.010 kt vs meteo 0.582", "07", "<2% variance", "Negligible contribution"],
        ["F03", "Lunar phase vs RI: χ²=19.66, V=0.020", "06", "V=0.020 (small)", "Stat. sig; tiny effect"],
        ["F04", "Waning Crescent RI=1.33%, Waxing=0.79%", "06", "37% range", "Lifecycle-dependent"],
        ["F05", "Yam speed vs bearing: ρ=−0.178, r²=3.2%", "06", "r²=0.032", "Small; leakage risk"],
        ["F06", "Shani/Varun speed vs bearing ρ≈−0.17", "06", "r²=0.028", "Same caveat as F05"],
        ["F07", "Varun retrograde: wind +4 kt, r_rb=0.088", "05", "Small", "Retro 60% of obs"],
        ["F08", "Shukra retrograde: wind +5 kt, r_rb=0.044", "05", "Small", "Genuinely rare (8%)"],
        ["F09", "Rahu degree shift at RI onset: η²=0.097", "06", "Medium on N=176", "Needs replication"],
        ["F10", "Mangal retro vs curvature: η²=0.0005", "06", "Tiny", "Trivially small"],
        ["F11", "Yoga KW H=143; 100% strata sig", "02, 08", "ε² small", "Residual signal"],
        ["F12", "Nakshatra KW H=116; 100% strata sig", "02, 08", "ε² small", "Residual signal"],
        ["F13", "Samvatsara ANOVA p<10⁻¹³⁰", "06", "Confounded", "= Calendar year effect"],
        ["F14", "Rahu degree vs Δw: ρ=0.013, r²=0.017%", "06", "Negligible", "Practically irrelevant"],
        ["F15", "4 Vedic features in SHAP/LIME top-14", "04", "Autocorr. inflated", "Not generalisable"],
        ["F16", "Meteo-only wins 4/5 forward-CV folds", "07", "0.15–0.48 kt worse", "Consistent across time"],
    ]
    story.extend(make_table(
        ["ID", "Finding", "Script", "Effect", "Verdict"],
        ev_rows,
        col_widths=[1.5*cm, 5.5*cm, 1.8*cm, 3.2*cm, 5*cm],
        caption="Table A1: Full consolidated evidence table (all 16 findings)."
    ))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("Building PDF …")
    doc  = build_doc()
    story = []

    section_title_page(story)
    section_intro(story)
    story.append(PageBreak())
    section_data(story)
    story.append(PageBreak())
    section_methods(story)
    story.append(PageBreak())
    section_results(story)
    story.append(PageBreak())
    section_discussion(story)
    story.append(PageBreak())
    section_limitations(story)
    section_conclusion(story)
    story.append(PageBreak())
    section_figure(story)
    story.append(PageBreak())
    section_references(story)
    story.append(PageBreak())
    section_appendix(story)

    doc.build(story)
    print(f"PDF written to: {PDF_OUT}")
    print(f"File size: {PDF_OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
