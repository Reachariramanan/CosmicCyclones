
# time_corrector.py

"""
Time Corrector for Storm Track Analysis — Deshaantara (Longitude) Correction
=============================================================================

When the actual storm coordinates differ from the nearest DrikPanchang geoname
location, the panchang is computed for the CITY's position, not the storm's.

Primary Correction — Longitude (Deshaantara):
    Earth's sidereal rotation period = 86164.0905 s (relative to stars).
    86164.0905 / 360 = 239.3447 s ≈ 3.989078 minutes per degree.
    For every degree of longitude the storm is EAST of the city, local solar
    time at the storm is ~3.989 minutes AHEAD of the city clock.

    corrected_time = city_local_time + (storm_lon − city_lon) × 3.989078 min

Secondary Note — Latitude:
    Latitude affects sunrise/sunset timing and lagna (ascendant), but it
    CANNOT be compensated with a simple time offset.  It is recorded as
    metadata for analytical reference.
"""

from datetime import datetime, timedelta
import math

# Earth sidereal rotation: 86164.0905 s / 360° = 239.3447 s = 3.989078 min per degree
MINUTES_PER_DEGREE_LON = 86164.0905 / 360 / 60  # ≈ 3.989078


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------

def compute_longitude_correction(storm_lon, city_lon):
    """
    Compute the Deshaantara time correction from the longitude gap.

    Parameters
    ----------
    storm_lon : float   Actual storm longitude (degrees, +E / −W)
    city_lon  : float   Nearest geoname city longitude (degrees, +E / −W)

    Returns
    -------
    dict
        delta_lon_deg        – longitude difference (storm − city)
        correction_minutes   – signed correction in minutes
        correction_timedelta – equivalent timedelta object
        direction            – human-readable 'add' or 'subtract'
    """
    delta_lon = storm_lon - city_lon

    # Wrap around the International Date Line → keep delta in [-180, +180]
    if delta_lon > 180:
        delta_lon -= 360
    elif delta_lon < -180:
        delta_lon += 360

    correction_minutes = delta_lon * MINUTES_PER_DEGREE_LON

    return {
        "delta_lon_deg":        round(delta_lon, 6),
        "correction_minutes":   round(correction_minutes, 2),
        "correction_timedelta": timedelta(minutes=correction_minutes),
        "direction":            "add" if correction_minutes >= 0 else "subtract",
    }


def compute_latitude_metadata(storm_lat, city_lat):
    """
    Record the latitude gap as metadata.

    Latitude differences change sunrise/sunset and lagna but cannot be
    compensated with a simple time offset.
    """
    delta_lat = storm_lat - city_lat
    return {
        "delta_lat_deg": round(delta_lat, 6),
        "note": (
            "Latitude difference affects sunrise/sunset and lagna (ascendant). "
            "A direct time correction is not applicable; "
            "provided for analytical reference only."
        ),
    }


# ---------------------------------------------------------------------------
# Time-string level correction
# ---------------------------------------------------------------------------

def correct_local_time(local_time_str, storm_lon, city_lon,
                       time_format="%H:%M:%S"):
    """
    Apply the longitude correction to a bare time string ("HH:MM:SS").

    Returns the corrected time string plus correction metadata.
    A non-zero ``date_shift_days`` means the corrected time crosses midnight.
    """
    correction = compute_longitude_correction(storm_lon, city_lon)

    base_time = datetime.strptime(local_time_str, time_format)
    corrected_time = base_time + correction["correction_timedelta"]

    date_shift = 0
    if corrected_time.day != base_time.day:
        date_shift = 1 if correction["correction_minutes"] > 0 else -1

    return {
        "original_time":   local_time_str,
        "corrected_time":  corrected_time.strftime(time_format),
        "date_shift_days": date_shift,
        "correction":      correction,
    }


def correct_local_datetime(local_date_str, local_time_str,
                           storm_lon, city_lon,
                           date_format="%d/%m/%Y",
                           time_format="%H:%M:%S"):
    """
    Apply the longitude correction to a date + time pair.

    Handles midnight crossover gracefully — the returned date is adjusted
    if the corrected time rolls into the next or previous day.
    """
    correction = compute_longitude_correction(storm_lon, city_lon)

    dt_str    = f"{local_date_str} {local_time_str}"
    dt_format = f"{date_format} {time_format}"

    base_dt      = datetime.strptime(dt_str, dt_format)
    corrected_dt = base_dt + correction["correction_timedelta"]

    return {
        "original_date":  local_date_str,
        "original_time":  local_time_str,
        "corrected_date": corrected_dt.strftime(date_format),
        "corrected_time": corrected_dt.strftime(time_format),
        "correction":     correction,
    }


# ---------------------------------------------------------------------------
# Master function — meant to be called from main_processor.py
# ---------------------------------------------------------------------------

def get_full_correction(storm_lat, storm_lon,
                        city_lat, city_lon,
                        local_date_str, local_time_str,
                        date_format="%d/%m/%Y",
                        time_format="%H:%M:%S"):
    """
    Compute the full time correction with all metadata.

    Typical integration in main_processor.process_single_track():

        from time_corrector import get_full_correction

        correction = get_full_correction(
            storm_lat=lat,          storm_lon=lon,
            city_lat=nearest["latitude"],
            city_lon=nearest["longitude"],
            local_date_str=drik_date,
            local_time_str=drik_time,
        )
        drik_date = correction["corrected_date"]
        drik_time = correction["corrected_time"]
        # … then call get_drik_data_cached(geoname_id, drik_date, drik_time)

    Returns
    -------
    dict  with storm/city coords, distance, original & corrected date/time,
          longitude correction detail, and latitude metadata.
    """
    lon_correction = compute_longitude_correction(storm_lon, city_lon)
    lat_metadata   = compute_latitude_metadata(storm_lat, city_lat)

    dt_str    = f"{local_date_str} {local_time_str}"
    dt_format = f"{date_format} {time_format}"

    base_dt      = datetime.strptime(dt_str, dt_format)
    corrected_dt = base_dt + lon_correction["correction_timedelta"]

    # Haversine for distance (duplicated here to keep the module self-contained)
    R = 6371  # km
    dlat = math.radians(city_lat - storm_lat)
    dlon = math.radians(city_lon - storm_lon)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(storm_lat)) *
         math.cos(math.radians(city_lat)) *
         math.sin(dlon / 2) ** 2)
    distance_km = 2 * R * math.asin(math.sqrt(a))

    return {
        "storm_coordinates": {"latitude": storm_lat, "longitude": storm_lon},
        "city_coordinates":  {"latitude": city_lat,  "longitude": city_lon},
        "distance_km":       round(distance_km, 2),
        "original_date":     local_date_str,
        "original_time":     local_time_str,
        "corrected_date":    corrected_dt.strftime(date_format),
        "corrected_time":    corrected_dt.strftime(time_format),
        "longitude_correction": {
            "delta_lon_deg":      lon_correction["delta_lon_deg"],
            "correction_minutes": lon_correction["correction_minutes"],
            "direction":          lon_correction["direction"],
        },
        "latitude_metadata": lat_metadata,
    }


# ---------------------------------------------------------------------------
# UTC → local time conversion  (mirrors main_processor.get_local_date_and_time)
# ---------------------------------------------------------------------------

def utc_to_local(iso_date, tz_string=None, offset_str=None):
    """
    Convert a UTC ISO timestamp to local date & time strings.

    Tries the named timezone first (handles DST), falls back to numeric
    offset, and ultimately returns UTC if nothing works.

    Parameters
    ----------
    iso_date   : str   e.g. "2026-01-19T12:00:00Z"
    tz_string  : str   IANA timezone name, e.g. "Asia/Kolkata"  (optional)
    offset_str : str   numeric UTC offset in hours, e.g. "5.5"  (optional)

    Returns
    -------
    (local_date_str, local_time_str)   e.g. ("19/01/2026", "17:30:00")
    """
    from zoneinfo import ZoneInfo

    dt_utc = datetime.strptime(iso_date, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=ZoneInfo("UTC")
    )

    if tz_string:
        try:
            dt_local = dt_utc.astimezone(ZoneInfo(tz_string))
            return dt_local.strftime("%d/%m/%Y"), dt_local.strftime("%H:%M:%S")
        except Exception:
            pass

    if offset_str:
        try:
            dt_local = dt_utc.replace(tzinfo=None) + timedelta(
                hours=float(offset_str)
            )
            return dt_local.strftime("%d/%m/%Y"), dt_local.strftime("%H:%M:%S")
        except ValueError:
            pass

    return dt_utc.strftime("%d/%m/%Y"), dt_utc.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# All-in-one helper — drop-in for main_processor.process_single_track
# ---------------------------------------------------------------------------

def correct_track_time(iso_date, storm_lat, storm_lon, nearest_location):
    """
    One-call convenience for process_single_track.

    Parameters
    ----------
    iso_date          : str   UTC ISO timestamp, e.g. "2026-01-19T12:00:00Z"
    storm_lat         : float Storm latitude
    storm_lon         : float Storm longitude
    nearest_location  : dict  A single entry from the geonames list
                              (must have latitude, longitude, timezone,
                               timezone_offset)

    Returns
    -------
    dict with keys:
        corrected_date  – date string ready for DrikPanchang
        corrected_time  – time string ready for DrikPanchang
        original_date   – city-based local date (before correction)
        original_time   – city-based local time (before correction)
        correction_detail – full correction payload from get_full_correction
    """
    city_lat = float(nearest_location["latitude"])
    city_lon = float(nearest_location["longitude"])

    # Step 1 — UTC → city local time
    local_date, local_time = utc_to_local(
        iso_date,
        tz_string=nearest_location.get("timezone"),
        offset_str=nearest_location.get("timezone_offset"),
    )

    # Step 2 — Apply Deshaantara (longitude) correction
    correction = get_full_correction(
        storm_lat=storm_lat,
        storm_lon=storm_lon,
        city_lat=city_lat,
        city_lon=city_lon,
        local_date_str=local_date,
        local_time_str=local_time,
    )

    return {
        "corrected_date":    correction["corrected_date"],
        "corrected_time":    correction["corrected_time"],
        "original_date":     correction["original_date"],
        "original_time":     correction["original_time"],
        "correction_detail": correction,
    }


# ---------------------------------------------------------------------------
# CLI — fully dynamic, accepts any event via command-line args
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Deshaantara (longitude) time corrector for storm tracks."
    )
    parser.add_argument(
        "--utc", required=True,
        help="UTC event time in ISO format, e.g. 2026-01-19T12:00:00Z",
    )
    parser.add_argument(
        "--lat", type=float, required=True,
        help="Storm latitude (degrees, +N / −S)",
    )
    parser.add_argument(
        "--lon", type=float, required=True,
        help="Storm longitude (degrees, +E / −W)",
    )

    # The user can supply city coords directly OR let us look them up
    parser.add_argument(
        "--city-lat", type=float, default=None,
        help="City latitude  (if omitted, uses --geo-file to find nearest)",
    )
    parser.add_argument(
        "--city-lon", type=float, default=None,
        help="City longitude (if omitted, uses --geo-file to find nearest)",
    )
    parser.add_argument(
        "--tz", default=None,
        help="IANA timezone name, e.g. Asia/Kolkata  (optional)",
    )
    parser.add_argument(
        "--tz-offset", default=None,
        help="Numeric UTC offset in hours, e.g. 5.5  (fallback if --tz missing)",
    )
    parser.add_argument(
        "--geo-file", default=None,
        help="Path to DrikPanchang_Locations.json for automatic nearest-city lookup",
    )

    args = parser.parse_args()

    # ── Resolve city coordinates ──────────────────────────────────────────
    if args.city_lat is not None and args.city_lon is not None:
        city_lat = args.city_lat
        city_lon = args.city_lon
        city_name = f"({city_lat}, {city_lon})"
        tz_string = args.tz
        tz_offset = args.tz_offset
    elif args.geo_file:
        from drik_location import load_geonames, find_nearest_location

        geonames = load_geonames(args.geo_file)
        nearest = find_nearest_location(args.lat, args.lon, geonames)
        if not nearest:
            print(f"ERROR: No geoname found near ({args.lat}, {args.lon})")
            raise SystemExit(1)

        city_lat = float(nearest["latitude"])
        city_lon = float(nearest["longitude"])
        city_name = (
            f"{nearest.get('city', '?')}, "
            f"{nearest.get('state', '')}, "
            f"{nearest.get('country', '')}"
        ).rstrip(", ")
        tz_string = nearest.get("timezone")
        tz_offset = nearest.get("timezone_offset")
    else:
        parser.error(
            "Provide either --city-lat/--city-lon OR --geo-file for auto-lookup."
        )

    # ── UTC → local time ─────────────────────────────────────────────────
    local_date, local_time = utc_to_local(args.utc, tz_string, tz_offset)

    # ── Apply correction ─────────────────────────────────────────────────
    result = get_full_correction(
        storm_lat=args.lat,
        storm_lon=args.lon,
        city_lat=city_lat,
        city_lon=city_lon,
        local_date_str=local_date,
        local_time_str=local_time,
    )

    # ── Print results ────────────────────────────────────────────────────
    print("=" * 65)
    print("  Time Corrector — Deshaantara Correction")
    print("=" * 65)
    print(f"  UTC event time       : {args.utc}")
    print(f"  Storm position       : {args.lat}°, {args.lon}°")
    print(f"  Nearest city         : {city_name} ({city_lat}°, {city_lon}°)")
    tz_label = tz_string or (f"UTC{'+' if float(tz_offset or 0) >= 0 else ''}{tz_offset}" if tz_offset else "UTC")
    print(f"  Timezone             : {tz_label}")
    print(f"  City local time      : {local_date}  {local_time}")
    print("-" * 65)
    print(f"  Longitude gap        : {result['longitude_correction']['delta_lon_deg']}°")
    print(f"  Time correction      : {result['longitude_correction']['correction_minutes']} min"
          f"  ({result['longitude_correction']['direction']})")
    print(f"  Corrected local time : {result['corrected_date']}  {result['corrected_time']}")
    print(f"  Lat gap (metadata)   : {result['latitude_metadata']['delta_lat_deg']}°")
    print(f"  Distance to city     : {result['distance_km']} km")
    print("-" * 65)
    print("\nFull correction payload:\n")
    print(json.dumps(result, indent=4, default=str))
