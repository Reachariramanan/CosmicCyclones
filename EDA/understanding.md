# Storm Data Understanding — 2026 Season

## Overview

This dataset contains **8 tropical cyclone events** from the 2026 season, each stored as a JSON file in `Compiled_data/2026/`. Every storm is tracked over time with meteorological observations enriched with **astronomical/astrological (Vedic Panchang)** data computed at the storm's geographic position and time.

---

## Storms in 2026

| Storm | Name | Type | Max Wind (kt) | Forecast | Tracks | Active |
|-------|------|------|---------------|----------|--------|--------|
| 16p-2026 | 16P | Cyclone | 6 | 7 | 12 | True |
| dudzai-2026 | Dudzai | Tropical Storm | 50 | 50 | 80 | True |
| ewetse-2026 | Ewetse | Tropical Storm | 5 | 6 | 9 | True |
| grant-2026 | Grant | Cyclone | 90 | 90 | 155 | N/A |
| iggy-2026 | Iggy | Cyclone | 4 | 4 | 25 | N/A |
| jenna-2026 | Jenna | Severe Cyclone | 12 | 12 | 39 | N/A |
| koji-2026 | Koji | Cyclone | 4 | 4 | 11 | N/A |
| nokaen-2026 | Nokaen (Ada) | Tropical Depression | 21 | 21 | 41 | True |

**Total track observations: ~372**

---

## JSON Schema (Per Storm File)

### Top-Level Fields
```
id              : str    — Unique storm identifier (e.g. "nokaen-2026")
name            : str    — Storm name (e.g. "Nokaen (Ada)")
title           : str    — Full title (e.g. "Tropical Depression Nokaen (Ada)")
description     : str    — Storm category description
season          : str    — Year ("2026")
type            : str    — Classification (Cyclone, Tropical Storm, Tropical Depression, Severe Cyclone)
max             : int    — Maximum observed wind speed (knots)
forecast        : int    — Forecast max wind
active          : bool   — Whether storm is still active (may be absent)
ja              : int    — Japan Meteorological Agency numbering (optional)
ph              : str    — Philippine local name (optional)
agencies        : str    — Tracking agency (e.g. "JTWC")
place           : str    — General geographic region
cone            : list   — [[lon, lat], ...] polygon of forecast cone
track           : dict   — {"track_1": {...}, "track_2": {...}, ...}
```

### Track Entry (track_N)
```
date            : str    — ISO 8601 UTC timestamp (e.g. "2026-01-14T12:00:00Z")
coordinates     : list   — [longitude, latitude]
wind            : int    — Wind speed in knots (nullable)
pressure        : int    — Atmospheric pressure in hPa (nullable, often missing)
analysis        : dict   — Contains all enrichment data
```

### Analysis → nearest_location
```
id              : str    — GeoNames location ID
city            : str    — Nearest city name
state           : str    — State/region
country         : str    — Country name
latitude        : str    — Location latitude (string)
longitude       : str    — Location longitude (string)
elevation       : str    — Elevation in meters (string)
olson_timezone  : str    — IANA timezone identifier
timezone_offset : str    — UTC offset in hours (string)
geoname_tag     : str    — GeoName feature tag
flag_url        : str    — Country flag image URL
```

### Analysis → astronomical_data
```
engine          : str    — "Skyfield"
bodies          : dict   — 9 celestial bodies (sun, moon, mercury, venus, mars, jupiter, saturn, uranus, neptune)
lunar_nodes     : list   — [{time_utc, type: "ascending"/"descending"}, ...]
```

**Per celestial body (10 numeric features each):**
```
ra_hours            : float — Right Ascension (hours, 0-24)
dec_deg             : float — Declination (degrees, -90 to +90)
ecliptic_lon_deg    : float — Ecliptic longitude (0-360°)
ecliptic_lat_deg    : float — Ecliptic latitude (degrees)
altitude_deg        : float — Altitude above horizon (-90 to +90°)
azimuth_deg         : float — Azimuth (0-360°)
sin_alt             : float — sin(altitude)
cos_alt             : float — cos(altitude)
sin_az              : float — sin(azimuth)
cos_az              : float — cos(azimuth)
```

**Total astronomical features: 9 bodies × 10 features = 90 numeric columns**

### Analysis → drik_panchang_data

#### Time Correction
```
date_local              : str — Local date (DD/MM/YYYY)
time_local              : str — Corrected local time
date_local_original     : str — Original local date
time_local_original     : str — Original local time before correction
time_correction:
    delta_lon_deg       : float — Longitude delta from reference
    correction_minutes  : float — Time correction applied
    direction           : str   — "add" or "subtract"
```

#### Panchang Cards (12 categories)
1. **Sunrise and Moonrise** — Sunrise, Sunset, Moonrise, Moonset times
2. **Panchang** — Tithi, Nakshatra, Yoga, Karana, Weekday, Paksha
3. **Lunar Month, Samvat and Brihaspati Samvatsara** — Vikram/Shaka/Gujarati Samvat, Chandramasa
4. **Rashi and Nakshatra** — Moonsign, Nakshatra Pada, Sunsign, Surya Nakshatra
5. **Ritu and Ayana** — Season (Ritu), Day/Night duration, Ayana (solstice period)
6. **Auspicious Timings** — Brahma Muhurta, Abhijit, Vijaya Muhurta, Amrit Kalam, etc.
7. **Inauspicious Timings** — Rahu Kalam, Yamaganda, Gulikai Kalam, Dur Muhurtam, etc.
8. **Anandadi and Tamil Yoga** — Anandadi Yoga, Tamil Yoga, Jeevanama, Netrama
9. **Nivas and Shool** — Disha Shool, Agnivasa, Chandra Vasa, Rahu Vasa, etc.
10. **Other Calendars and Epoch** — Kaliyuga, Ayanamsha, Julian Day, etc.
11. **Chandrabalam & Tarabalam** — Moon strength for different Rashis/Nakshatras
12. **Panchaka Rahita Muhurta and Udaya Lagna** — Time windows and rising signs

#### Tamil Panchang
- Same 12 card categories as above but with Tamil naming conventions

#### Planetary Positions (Vedic/Sidereal)
15 bodies: Lagna, Surya, Chandra, Mangal, Budha, Guru, Shukra, Shani, Arun, Varun, Yam, Rahu, Ketu, Spashth Rahu, Spashth Ketu

**Per body (9 features):**
```
longitude               : str   — Sidereal longitude in DMS format
nakshatra               : str   — Lunar mansion name
padam                   : int   — Nakshatra quarter (1-4)
nakshatra_lord_sub_lord : str   — Ruling planet pair
full_degree             : float — Full sidereal degree (0-360)
latitude                : str   — Celestial latitude in DMS
speed_deg_per_day       : float — Daily motion speed
right_ascension         : float — RA in degrees
declination             : float — Declination in degrees
```

---

## Data Characteristics

### Numeric Variables (Directly Usable for ML/EDA)
| Category | Count | Examples |
|----------|-------|---------|
| Core meteorological | 4 | wind, pressure, longitude, latitude |
| Astronomical (Skyfield) | 90 | sun_ra_hours, moon_altitude_deg, jupiter_azimuth_deg, ... |
| Vedic planetary positions | 45+ | surya_full_degree, chandra_speed_deg_per_day, ... |
| Time correction | 2 | delta_lon_deg, correction_minutes |
| Storm metadata | 2 | max, forecast |

### Categorical Variables
| Category | Examples |
|----------|---------|
| Storm identity | name, type, season, agencies |
| Location | city, state, country, timezone |
| Panchang (Vedic calendar) | Tithi, Nakshatra, Yoga, Karana, Weekday, Paksha, Moonsign, Sunsign, Ritu, Ayana |
| Tamil Panchang | Tamil variants of above |

### Temporal Variables
- `date` (track timestamp, UTC)
- `date_local` / `time_local` (location-corrected)
- Sunrise, Sunset, Moonrise, Moonset times
- Auspicious/Inauspicious timing windows

### Data Quality Notes
- **Pressure**: Frequently NULL (especially for weaker storms)
- **Wind**: Generally complete
- **Active field**: Missing for some storms (Grant, Iggy, Jenna, Koji)
- **Optional fields**: `ja`, `ph` present only for storms tracked by Japan/Philippines
- **Panchang errors**: Field exists but generally NULL (data is clean)

---

## Key Research Questions This Data Can Answer

1. **Do astronomical positions correlate with storm intensity (wind/pressure)?**
2. **Are certain Vedic Nakshatras / Tithis / Yogas associated with storm formation or intensification?**
3. **Which celestial body positions are most predictive of storm behavior?**
4. **Is there a relationship between lunar nodes and cyclone activity?**
5. **Do planetary speeds (retrograde vs direct motion) correlate with storm dynamics?**
6. **Can Vedic auspicious/inauspicious periods predict storm intensification windows?**
7. **Feature importance: Which of the 90+ astronomical variables actually matter?**

---

## EDA Plan

### Phase 1: Data Transformation
- Flatten nested JSON into tabular format
- Extract all numeric features from astronomical data, planetary positions
- Encode categorical Panchang variables
- Save as Parquet for efficient analysis

### Phase 2: Basic EDA
- Descriptive statistics (mean, std, min, max, quartiles)
- Missing value analysis
- Distribution plots for wind, pressure, coordinates
- Correlation matrix for astronomical features
- Storm trajectory visualization

### Phase 3: Advanced EDA
- Feature importance via Random Forest / Gradient Boosting
- SHAP (SHapley Additive exPlanations) for global/local feature importance
- LIME (Local Interpretable Model-agnostic Explanations) for instance-level interpretation
- PCA / dimensionality reduction on 90+ astronomical features
- Time series decomposition of storm intensity
- Cluster analysis of storm tracks
- Cross-correlation of planetary positions with intensity changes

### Phase 4: Comprehensive Analysis
- Interaction effects between planetary positions
- Vedic categorization analysis (Tithi × Nakshatra × intensity)
- Spatial analysis of storm tracks vs astronomical alignments
- Anomaly detection in astronomical feature space
- Mutual information and feature selection
