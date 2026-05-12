"""
01_data_transform.py
====================
Flatten all storm JSON files (2024, 2025, 2026) into a single tabular DataFrame
and save as Parquet. Extracts: meteorological core, astronomical (Skyfield),
Vedic planetary positions, Panchang categorical variables, and time correction data.
"""

import json
import glob
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

COMPILED_DIR = Path(__file__).resolve().parent.parent / "Compiled_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CELESTIAL_BODIES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
BODY_FIELDS = ["ra_hours", "dec_deg", "ecliptic_lon_deg", "ecliptic_lat_deg",
               "altitude_deg", "azimuth_deg", "sin_alt", "cos_alt", "sin_az", "cos_az"]

# Vedic planet name cleaning map
VEDIC_PLANET_CLEAN = {
    "Lagna": "lagna",
    "Surya": "surya",
    "Chandra": "chandra",
    "Mangal": "mangal",
    "Budha": "budha",
    "Guru": "guru",
    "Shukra": "shukra",
    "Shani": "shani",
    "Arun": "arun",
    "Varun": "varun",
    "Yam": "yam",
    "Rahu": "rahu",
    "Ketu": "ketu",
    "Spashth Rahu": "spashth_rahu",
    "Spashth Ketu": "spashth_ketu",
}

VEDIC_NUMERIC_FIELDS = ["full_degree", "padam", "speed_deg_per_day", "right_ascension", "declination"]


# ─────────────────────────────────────────────────────────────────────────────
# Spline & Cone feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def _spline_features(spline: list, current_lon: float | None, current_lat: float | None) -> dict:
    """
    Extract scalar features from a per-track spline (list of [lon, lat]).

    spline[0]  = current/smoothed position (aligns with track coordinates)
    spline[-1] = next observation position (where the storm is heading)

    Features
    --------
    spline_point_count  : number of interpolation knots
    spline_length_deg   : total Euclidean path length (degrees)
    spline_end_lon/lat  : forecast next position
    spline_displacement : straight-line dist from spline[0] to spline[-1]
    spline_bearing_deg  : compass bearing (0=N, 90=E) from start to end
    spline_curvature    : path_length / displacement  (1=straight, >1=curved)
    """
    out = {
        "spline_point_count": None,
        "spline_length_deg": None,
        "spline_end_lon": None,
        "spline_end_lat": None,
        "spline_displacement": None,
        "spline_bearing_deg": None,
        "spline_curvature": None,
    }
    if not spline or len(spline) < 2:
        return out

    out["spline_point_count"] = len(spline)

    # Total path length (Euclidean in degree-space)
    length = 0.0
    for i in range(1, len(spline)):
        dx = spline[i][0] - spline[i - 1][0]
        dy = spline[i][1] - spline[i - 1][1]
        length += (dx ** 2 + dy ** 2) ** 0.5
    out["spline_length_deg"] = round(length, 6)

    # End position
    out["spline_end_lon"] = spline[-1][0]
    out["spline_end_lat"] = spline[-1][1]

    # Displacement: start → end
    dx = spline[-1][0] - spline[0][0]
    dy = spline[-1][1] - spline[0][1]
    disp = (dx ** 2 + dy ** 2) ** 0.5
    out["spline_displacement"] = round(disp, 6)

    # Bearing (meteorological convention: 0=North, clockwise)
    import math
    bearing = math.degrees(math.atan2(dx, dy)) % 360
    out["spline_bearing_deg"] = round(bearing, 2)

    # Curvature ratio
    if disp > 1e-9:
        out["spline_curvature"] = round(length / disp, 4)
    else:
        out["spline_curvature"] = None  # stationary

    return out


def _cone_features(cone: list) -> dict:
    """
    Extract scalar features from a storm-level cone polygon (list of [lon, lat]).

    Features
    --------
    cone_point_count : polygon vertex count
    cone_area_deg2   : signed area via shoelace formula (abs value)
    cone_lon_min/max : bounding box
    cone_lat_min/max : bounding box
    cone_lon_spread  : longitude extent
    cone_lat_spread  : latitude extent
    """
    out = {
        "cone_point_count": None,
        "cone_area_deg2": None,
        "cone_lon_min": None,
        "cone_lon_max": None,
        "cone_lat_min": None,
        "cone_lat_max": None,
        "cone_lon_spread": None,
        "cone_lat_spread": None,
    }
    if not cone or len(cone) < 3:
        return out

    out["cone_point_count"] = len(cone)
    lons = [p[0] for p in cone]
    lats = [p[1] for p in cone]

    # Shoelace area
    n = len(cone)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += cone[i][0] * cone[j][1]
        area -= cone[j][0] * cone[i][1]
    out["cone_area_deg2"] = round(abs(area) / 2.0, 4)

    out["cone_lon_min"] = min(lons)
    out["cone_lon_max"] = max(lons)
    out["cone_lat_min"] = min(lats)
    out["cone_lat_max"] = max(lats)
    out["cone_lon_spread"] = round(max(lons) - min(lons), 4)
    out["cone_lat_spread"] = round(max(lats) - min(lats), 4)
    return out


def clean_vedic_planet_name(raw_name: str) -> str:
    """Remove emoji modifiers (🔥, ↺) and map to clean snake_case name."""
    cleaned = re.sub(r'[🔥↺\s]+$', '', raw_name).strip()
    for key, val in VEDIC_PLANET_CLEAN.items():
        if cleaned == key:
            return val
    return cleaned.lower().replace(" ", "_")


def extract_first_panchang_value(val):
    """Extract the first meaningful value from Panchang fields (may be str or list)."""
    if isinstance(val, list):
        return val[0] if val else None
    return val


def flatten_track(storm_meta: dict, track_key: str, track_data: dict, agg_lookup: dict | None = None) -> dict:
    """Flatten a single track entry into a flat dictionary."""
    row = {}

    # --- Storm metadata ---
    row["storm_id"] = storm_meta.get("id")
    row["storm_name"] = storm_meta.get("name")
    row["storm_type"] = storm_meta.get("type")
    row["storm_description"] = storm_meta.get("description")
    row["season"] = storm_meta.get("season")
    row["storm_max_wind"] = storm_meta.get("max")
    row["storm_forecast"] = storm_meta.get("forecast")
    row["storm_active"] = storm_meta.get("active")
    row["agencies"] = storm_meta.get("agencies")
    row["place"] = storm_meta.get("place")
    row["year"] = storm_meta.get("year")

    analysis = track_data.get("analysis", {})
    sci = analysis.get("scientific_data", {})

    # Find the matching aggregate track entry (carries date, coordinates, wind, pressure, spline)
    agg: dict = {}
    if agg_lookup:
        agg = agg_lookup.get(row["storm_id"], {}).get(sci.get("datetime_utc", ""), {})

    # --- Track core ---
    row["track_id"] = track_key
    row["track_num"] = int(track_key.replace("track_", ""))
    row["date_utc"] = agg.get("date") or sci.get("datetime_utc") or track_data.get("date")
    sci_loc = sci.get("location", {})
    coords = agg.get("coordinates") or [sci_loc.get("longitude"), sci_loc.get("latitude")]
    row["longitude"] = coords[0] if coords else None
    row["latitude"] = coords[1] if coords else None
    row["wind"] = agg.get("wind") if agg.get("wind") is not None else track_data.get("wind")
    row["pressure"] = agg.get("pressure") if agg.get("pressure") is not None else track_data.get("pressure")

    # --- Spline features (smooth inter-observation path) ---
    spline = agg.get("spline") or track_data.get("spline")
    if not spline or not isinstance(spline, list) or len(spline) < 2:
        spline = None  # Invalidate bad spline data
    row.update(_spline_features(spline, row["longitude"], row["latitude"]))

    # --- Nearest location ---
    loc = analysis.get("nearest_location", {})
    row["nearest_city"] = loc.get("city")
    row["nearest_state"] = loc.get("state")
    row["nearest_country"] = loc.get("country")
    row["nearest_lat"] = _safe_float(loc.get("latitude"))
    row["nearest_lon"] = _safe_float(loc.get("longitude"))
    row["elevation"] = _safe_float(loc.get("elevation"))
    row["timezone"] = loc.get("olson_timezone")
    row["timezone_offset"] = _safe_float(loc.get("timezone_offset"))

    # --- Astronomical data (Skyfield) ---
    astro = analysis.get("scientific_data", {})
    bodies = astro.get("bodies", {})
    for body in CELESTIAL_BODIES:
        body_data = bodies.get(body, {})
        for field in BODY_FIELDS:
            col_name = f"{body}_{field}"
            row[col_name] = body_data.get(field)

    # Lunar nodes
    lunar_nodes = astro.get("lunar_node_events_in_month", [])
    for node in lunar_nodes:
        ntype = node.get("type", "")
        row[f"lunar_node_{ntype}_utc"] = node.get("time_utc")

    # --- Drik Panchang data ---
    dpd = analysis.get("drik_panchang_data", {})
    row["date_local"] = track_data.get("date_local")
    row["time_local"] = track_data.get("time_local")

    # Panchang categorical variables
    panchang = dpd.get("panchang") or {}

    # Sunrise/Moonrise
    srm = panchang.get("Sunrise and Moonrise", {})
    row["sunrise"] = srm.get("Sunrise")
    row["sunset"] = srm.get("Sunset")
    row["moonrise"] = srm.get("Moonrise")
    row["moonset"] = srm.get("Moonset")

    # Core Panchang
    panch = panchang.get("Panchangam", {})
    row["tithi"] = extract_first_panchang_value(panch.get("Tithi"))
    row["nakshatra"] = extract_first_panchang_value(panch.get("Nakshathram"))
    row["yoga"] = extract_first_panchang_value(panch.get("Yoga"))
    row["karana"] = extract_first_panchang_value(panch.get("Karana"))
    row["weekday"] = panch.get("Weekday")
    row["paksha"] = panch.get("Paksha")

    # Rashi and Nakshathram
    rn = panchang.get("Rashi and Nakshathram", {})
    row["moonsign"] = extract_first_panchang_value(rn.get("Moonsign"))
    row["sunsign"] = extract_first_panchang_value(rn.get("Sunsign"))

    # Ritu and Ayana
    ra = panchang.get("Ritu and Ayana", {})
    row["drik_ritu"] = ra.get("Drik Ritu")
    row["vedic_ritu"] = ra.get("Vedic Ritu")
    row["drik_ayana"] = ra.get("Drik Ayana")
    row["vedic_ayana"] = ra.get("Vedic Ayana")
    row["dinamana"] = ra.get("Dinamana")
    row["ratrimana"] = ra.get("Ratrimana")

    # Inauspicious timings
    inausp = panchang.get("Inauspicious Timings", {})
    row["rahu_kalam"] = inausp.get("Rahu Kalam")
    row["yamaganda"] = inausp.get("Yamaganda")
    row["gulikai_kalam"] = inausp.get("Gulikai Kalam")

    # Anandadi & Tamil Yoga
    at = panchang.get("Anandadi and Tamil Yoga", {})
    row["anandadi_yoga"] = extract_first_panchang_value(at.get("Anandadi Yoga"))
    row["tamil_yoga"] = extract_first_panchang_value(at.get("Tamil Yoga"))

    # Nivas and Shool
    ns = panchang.get("Nivas and Shool", {})
    row["disha_shool"] = ns.get("Disha Shool")
    row["agnivasa"] = ns.get("Agnivasa")

    # Other Calendars
    oc = panchang.get("Other Calendars and Epoch", {})
    row["lahiri_ayanamsha"] = _safe_float(oc.get("Lahiri Ayanamsha"))

    # --- Vedic Planetary positions ---
    planet_dict = dpd.get("planetary_positions") or {}
    for raw_name, pdata in planet_dict.items():
        if not isinstance(pdata, dict):
            continue
        clean_name = clean_vedic_planet_name(raw_name)
        for field in VEDIC_NUMERIC_FIELDS:
            col_name = f"vedic_{clean_name}_{field}"
            row[col_name] = pdata.get(field)
        # Also extract nakshatra as categorical
        nk = pdata.get("nakshatra", "")
        row[f"vedic_{clean_name}_nakshatra"] = nk.replace("🔥", "").strip() if isinstance(nk, str) else str(nk)

    # --- Cone features (broadcast from storm level) ---
    row.update(_cone_features(storm_meta.get("cone")))

    # --- Samvatsara ---
    row["samvatsara"] = get_samvatsara(row["year"])

    return row


def _safe_float(val):
    """Convert string to float safely."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_duration_to_minutes(dur_str) -> float | None:
    """Parse '11 Hours 14 Mins 45 Secs' to total minutes."""
    if not dur_str or not isinstance(dur_str, str):
        return None
    hours = mins = secs = 0
    m = re.search(r'(\d+)\s*Hour', dur_str)
    if m:
        hours = int(m.group(1))
    m = re.search(r'(\d+)\s*Min', dur_str)
    if m:
        mins = int(m.group(1))
    m = re.search(r'(\d+)\s*Sec', dur_str)
    if m:
        secs = int(m.group(1))
    return hours * 60 + mins + secs / 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Samvatsara Mapping
# ─────────────────────────────────────────────────────────────────────────────

SAMVATSARA_MAP = {
    (1999, 2000): "Pramadi",
    (2000, 2001): "Vikrama",
    (2001, 2002): "Vrisha",
    (2002, 2003): "Chitrabhanu",
    (2003, 2004): "Svabhanu",
    (2004, 2005): "Tarana",
    (2005, 2006): "Parthiva",
    (2006, 2007): "Vyaya",
    (2007, 2008): "Sarvajit",
    (2008, 2009): "Sarvadhari",
    (2009, 2010): "Virodhi",
    (2010, 2011): "Vikruti",
    (2011, 2012): "Khara",
    (2012, 2013): "Nandana",
    (2013, 2014): "Vijaya",
    (2014, 2015): "Jaya",
    (2015, 2016): "Manmatha",
    (2016, 2017): "Durmukhi",
    (2017, 2018): "Hevilambi",
    (2018, 2019): "Vilambi",
    (2019, 2020): "Vikari",
    (2020, 2021): "Sharvari",
    (2021, 2022): "Plava",
    (2022, 2023): "Shubhakritu",
    (2023, 2024): "Shobhakritu",
    (2024, 2025): "Krodhi",
    (2025, 2026): "Vishvavasu",
}

def get_samvatsara(year: int) -> str | None:
    """Map a year to its Samvatsara name."""
    if not year:
        return None
    for (start, end), name in SAMVATSARA_MAP.items():
        if start <= year < end:
            return name
    return None


def main():
    # Build aggregate lookup: {storm_id: {date_utc: track_entry}} for joining wind/pressure/coords/spline
    agg_data_path = COMPILED_DIR.parent / "all_storm_data.json"
    agg_lookup: dict = {}
    agg_meta: dict = {}
    if agg_data_path.exists():
        with open(agg_data_path, encoding="utf-8") as f:
            agg_storms = json.load(f)
        for s in agg_storms:
            sid = s.get("id")
            if not sid:
                continue
            agg_lookup[sid] = {t["date"]: t for t in s.get("track", []) if t.get("date")}
            agg_meta[sid] = {"active": s.get("active")}
        print(f"Loaded aggregate lookup: {len(agg_lookup)} storms")
    else:
        print(f"Warning: all_storm_data.json not found at {agg_data_path}")

    all_rows = []
    json_files = sorted(glob.glob(str(COMPILED_DIR / "*.json")))
    print(f"Found {len(json_files)} storm files in {COMPILED_DIR}")

    for fpath in json_files:
        with open(fpath, encoding="utf-8") as f:
            storm = json.load(f)

        year_match = re.search(r'-(\d{4})\.json$', fpath)
        file_year = int(year_match.group(1)) if year_match else None

        storm_meta = {k: v for k, v in storm.items() if k != "track"}
        # keep cone in storm_meta so flatten_track can extract cone features
        if file_year is not None:
            storm_meta["year"] = file_year
        # Fill active from aggregate if not present in compiled file
        sid = storm_meta.get("id")
        if "active" not in storm_meta and sid in agg_meta:
            storm_meta["active"] = agg_meta[sid].get("active")
        track_dict = storm.get("track", {})

        for track_key in sorted(track_dict.keys(), key=lambda x: int(x.replace("track_", ""))):
            track_data = track_dict[track_key]
            row = flatten_track(storm_meta, track_key, track_data, agg_lookup)
            all_rows.append(row)

    print(f"\nTotal storm files: {len(json_files)}")

    df = pd.DataFrame(all_rows)

    # --- Post-processing ---
    # Parse date
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)

    # Derive temporal features
    df["hour_utc"] = df["date_utc"].dt.hour
    df["day_of_week"] = df["date_utc"].dt.dayofweek
    df["day_of_year"] = df["date_utc"].dt.dayofyear
    df["month"] = df["date_utc"].dt.month

    # Parse dinamana/ratrimana to minutes
    df["dinamana_minutes"] = df["dinamana"].apply(parse_duration_to_minutes)
    df["ratrimana_minutes"] = df["ratrimana"].apply(parse_duration_to_minutes)

    # Compute intensity change (delta wind between consecutive tracks of same storm)
    df = df.sort_values(["storm_id", "track_num"]).reset_index(drop=True)
    df["wind_change"] = df.groupby("storm_id")["wind"].diff()
    df["pressure_change"] = df.groupby("storm_id")["pressure"].diff()

    # Compute distance moved (simple Euclidean proxy)
    df["lon_change"] = df.groupby("storm_id")["longitude"].diff()
    df["lat_change"] = df.groupby("storm_id")["latitude"].diff()
    df["displacement"] = (df["lon_change"].astype(float) ** 2 + df["lat_change"].astype(float) ** 2) ** 0.5

    # Time delta between tracks
    df["hours_elapsed"] = df.groupby("storm_id")["date_utc"].diff().dt.total_seconds() / 3600

    # Speed of movement (degrees per hour)
    df["movement_speed"] = df["displacement"] / df["hours_elapsed"].replace(0, np.nan)

    # --- Summary ---
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Storms: {df['storm_id'].nunique()}")
    print(f"Total tracks: {len(df)}")

    # Column categories
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    print(f"\nNumeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(cat_cols)}")
    print(f"Datetime columns: {len(dt_cols)}")

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    high_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
    print(f"\nColumns with missing data: {len(high_missing)}")
    if len(high_missing) > 0:
        print(high_missing.head(20).to_string())

    # Coerce all object columns to string for Parquet compatibility
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: str(x) if x is not None and not isinstance(x, str) else x)

    # Save Parquet + CSV
    parquet_path = OUTPUT_DIR / "storm_data_all.parquet"
    csv_path = OUTPUT_DIR / "storm_data_all.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {parquet_path}")
    print(f"Saved: {csv_path}")

    # Save Excel (tz-naive datetime for compatibility)
    df_excel = df.copy()
    for col in df_excel.select_dtypes(include=["datetimetz"]).columns:
        df_excel[col] = df_excel[col].dt.tz_convert("UTC").dt.tz_localize(None)
    excel_path = OUTPUT_DIR / "storm_data_all.xlsx"
    df_excel.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"Saved: {excel_path}")

    # Also save column metadata
    col_info = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null": df.notnull().sum().values,
        "null_count": df.isnull().sum().values,
        "null_pct": (df.isnull().sum() / len(df) * 100).round(1).values,
        "nunique": [df[c].nunique() for c in df.columns],
    })
    col_info.to_csv(OUTPUT_DIR / "column_metadata.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'column_metadata.csv'}")

    return df


if __name__ == "__main__":
    df = main()
