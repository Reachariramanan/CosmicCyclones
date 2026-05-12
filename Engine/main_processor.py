# main_processor.py

import json
import os
import threading
import subprocess
import time
import random
from collections import deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local modules
from skyfield_engine import compute_skyfield
from drik_location import find_nearest_location, load_geonames
from drik_scraper import fetch_drik_data
from time_corrector import correct_track_time

INPUT_FILE = os.path.join(os.path.dirname(__file__), "all_storm_data.json")
GEO_FILE   = os.path.join(os.path.dirname(__file__), "DrikPanchang_Locations.json")
OUTPUT_FOLDER = "Compiled_data"

MAX_WORKERS = 15

# Each fetch_drik_data() fires 3 requests in parallel per logical fetch.
# 2 concurrent logical fetches = 6 simultaneous connections — safe but fast.
DRIK_MAX_CONCURRENT = 3
drik_sema = threading.BoundedSemaphore(DRIK_MAX_CONCURRENT)

drik_cache = {}
fetching_events = {}
cache_lock = threading.Lock()

# ── Warp reset gate ───────────────────────────────────────────────────────
# Cleared during a warp reset; all Drik fetchers wait on it before starting.
warp_ok = threading.Event()
warp_ok.set()

_last_warp_reset     = 0.0
WARP_RESET_COOLDOWN_SEC = 60

# ── Pattern-based failure tracking ───────────────────────────────────────
# Only trigger a warp reset when >= FAILURE_THRESHOLD failures occur within
# FAILURE_WINDOW_SEC seconds (avoids resetting on isolated blips).
_failure_lock       = threading.Lock()
_failure_times      = deque()          # timestamps of recent Drik failures
FAILURE_WINDOW_SEC  = 30
FAILURE_THRESHOLD   = 3

# Short negative cache: prevents hammering the same key after repeated fails.
NEGATIVE_CACHE_TTL_SEC = 120
drik_negative_cache = {}  # cache_key -> last_fail_ts


def get_local_date_and_time(iso_date, tz_string, offset_str):
    """Converts Zulu time to local clock time."""
    try:
        dt_utc = datetime.strptime(iso_date, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=ZoneInfo("UTC"))
    except:
        return None, None

    if tz_string:
        try:
            dt_local = dt_utc.astimezone(ZoneInfo(tz_string))
            return dt_local.strftime("%d/%m/%Y"), dt_local.strftime("%H:%M:%S")
        except:
            pass

    if offset_str:
        try:
            dt_local = dt_utc.replace(tzinfo=None) + timedelta(hours=float(offset_str))
            return dt_local.strftime("%d/%m/%Y"), dt_local.strftime("%H:%M:%S")
        except:
            pass

    return dt_utc.strftime("%d/%m/%Y"), dt_utc.strftime("%H:%M:%S")


def _record_failure():
    """Record a Drik failure. Returns True when the pattern threshold is met."""
    now = time.time()
    with _failure_lock:
        # drop stale entries
        while _failure_times and now - _failure_times[0] > FAILURE_WINDOW_SEC:
            _failure_times.popleft()
        _failure_times.append(now)
        return len(_failure_times) >= FAILURE_THRESHOLD


def reset_connection():
    """
    Pause all Drik fetchers (via warp_ok gate + draining drik_sema), reset
    Warp, then resume.  The semaphore drain ensures no in-flight request is
    killed mid-flight by the VPN disconnect.
    """
    global _last_warp_reset

    now = time.time()
    with cache_lock:
        if now - _last_warp_reset < WARP_RESET_COOLDOWN_SEC:
            return
        _last_warp_reset = now

    # Block new fetches from starting
    warp_ok.clear()

    tqdm.write("\n--- Flagging detected. Resetting Connection ---")

    # The one in-flight request (if any) will get a connection error and retry
    # after warp_ok is set again — no semaphore drain needed (avoids deadlock).
    try:
        subprocess.run(["warp-cli", "disconnect"], capture_output=True)
        time.sleep(2)
        subprocess.run(["warp-cli", "connect"], capture_output=True)
        time.sleep(8)
        with _failure_lock:
            _failure_times.clear()   # reset the sliding window after recovery
    except Exception:
        tqdm.write("Warp-cli failed to reset.")
    finally:
        warp_ok.set()   # let fetchers proceed again


def _looks_like_transient_failure(data):
    """
    Decide if returned data is effectively a failure even if no exception was raised.
    This avoids relying on specific 'error' strings.
    """
    if not isinstance(data, dict):
        return True
    # If english_panchang exists but has no cards, your earlier logic treated this as bad
    ep = data.get("english_panchang")
    if isinstance(ep, dict) and ep and not ep.get("cards"):
        return True
    return False


def get_drik_data_cached(geoname_id, drik_date, drik_time):
    cache_key = (geoname_id, drik_date, drik_time)

    with cache_lock:
        # normal cache hit
        if cache_key in drik_cache:
            return drik_cache[cache_key]

        # negative cache hit (avoid hammering the same key repeatedly)
        last_fail = drik_negative_cache.get(cache_key)
        if last_fail and (time.time() - last_fail) < NEGATIVE_CACHE_TTL_SEC:
            return {
                "error": f"Skipped due to recent failure (TTL {NEGATIVE_CACHE_TTL_SEC}s)",
                "english_panchang": {},
                "tamil_panchang": {},
                "planetary_positions": {}
            }

        # single-flight: only one thread fetches per key
        if cache_key not in fetching_events:
            fetching_events[cache_key] = threading.Event()
            is_fetcher = True
        else:
            is_fetcher = False
            event = fetching_events[cache_key]

    if not is_fetcher:
        event.wait()
        with cache_lock:
            return drik_cache.get(cache_key)

    # fetcher path
    event = fetching_events[cache_key]
    data = None
    err = None

    # NEW: retry with exponential backoff for any transient network failure
    max_attempts = 5
    base_sleep = 1.5

    try:
        # Wait if a warp reset is in progress before touching the semaphore
        warp_ok.wait()
        with drik_sema:  # throttle Drik fetching to DRIK_MAX_CONCURRENT slots
            for attempt in range(1, max_attempts + 1):
                try:
                    data = fetch_drik_data(geoname_id, drik_date, drik_time)

                    if data and not _looks_like_transient_failure(data):
                        break  # success — exit retry loop

                    # Soft failure (empty cards etc.) counts as a failure
                    if _record_failure():
                        reset_connection()
                        warp_ok.wait()   # re-check gate after reset

                except Exception as e:
                    err = e
                    if _record_failure():
                        reset_connection()
                        warp_ok.wait()

                if attempt < max_attempts:
                    sleep_s = (base_sleep * (2 ** (attempt - 1))) + random.uniform(0.0, 0.8)
                    time.sleep(sleep_s)

            if not data or _looks_like_transient_failure(data):
                # store a minimal error payload (prevents repeated hammering)
                data = {
                    "error": f"Drik fetch failed after {max_attempts} attempts: {repr(err)}",
                    "english_panchang": {},
                    "tamil_panchang": {},
                    "planetary_positions": {}
                }
                with cache_lock:
                    drik_negative_cache[cache_key] = time.time()

    finally:
        # NEW: always release waiters to prevent deadlocks
        with cache_lock:
            drik_cache[cache_key] = data
            event.set()
            fetching_events.pop(cache_key, None)

    return data


def process_single_track(track, track_index, cyclone_id, geonames_list):
    try:
        lon, lat = track["coordinates"]
        iso_date = track["date"]

        skyfield_data = compute_skyfield(lat, lon, iso_date)
        nearest = find_nearest_location(lat, lon, geonames_list)
        if not nearest:
            return track_index, None, "No Location"

        time_fix = correct_track_time(iso_date, lat, lon, nearest)
        d_local  = time_fix["corrected_date"]
        t_local  = time_fix["corrected_time"]
        drik_data = get_drik_data_cached(nearest["id"], d_local, t_local)

        geoname_id = nearest["id"]
        track_data = {
            "date": iso_date,
            "coordinates": [lon, lat],
            "wind": track.get("wind"),
            "pressure": track.get("pressure"),
            "analysis": {
                "nearest_location": nearest,
                "astronomical_data": {
                    "engine": "Skyfield",
                    "bodies": skyfield_data.get("bodies", {}),
                    "lunar_nodes": skyfield_data.get("lunar_node_events_in_month", [])
                },
                "drik_panchang_data": {
                    "date_local": d_local,
                    "time_local": t_local,
                    "date_local_original": time_fix["original_date"],
                    "time_local_original": time_fix["original_time"],
                    "time_correction": time_fix["correction_detail"]["longitude_correction"],
                    "panchang": drik_data.get("english_panchang", {}),
                    "tamil_panchang": drik_data.get("tamil_panchang", {}),
                    "planetary_positions": drik_data.get("planetary_positions", {}),
                    "error": drik_data.get("error")  # NEW: preserve error if present
                },
                "source": {
                    "skyfield": "Skyfield (DE440.bsp)",
                    "drik_panchang_urls": {
                        "panchang": f"https://www.drikpanchang.com/panchang/day-panchang.html?geoname-id={geoname_id}&date={d_local}&time={t_local}",
                        "tamil_panchang": f"https://www.drikpanchang.com/tamil/tamil-day-panchangam.html?geoname-id={geoname_id}&date={d_local}&time={t_local}",
                        "planetary_positions": f"https://www.drikpanchang.com/planet/position/planetary-positions-sidereal.html?geoname-id={geoname_id}&date={d_local}&time={t_local}"
                    }
                }
            }
        }
        return track_index, (f"track_{track_index}", track_data), None
    except Exception as e:
        return track_index, None, str(e)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    geonames_list = load_geonames(GEO_FILE)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        cyclones = json.load(f)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for cyclone in tqdm(cyclones, desc="Overall Progress", position=0):
            i_id = cyclone.get("id", "unknown_id")
            season = str(cyclone.get("season", "Misc"))

            try:
                if not (2024 <= int(season) <= 2026):
                    continue
            except ValueError:
                continue

            season_dir = os.path.join(OUTPUT_FOLDER, season)
            os.makedirs(season_dir, exist_ok=True)

            # Resume: skip cyclones whose output file already exists
            if os.path.exists(os.path.join(season_dir, f"{i_id}.json")):
                continue

            output_data = {**cyclone, "track": {}}
            tracks = cyclone.get("track", [])

            futures = [
                executor.submit(process_single_track, t, i, i_id, geonames_list)
                for i, t in enumerate(tracks, 1)
            ]

            results_buffer = {}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f" > {i_id}",
                position=1,
                leave=False
            ):
                idx, res, err = future.result()
                if res:
                    results_buffer[idx] = res

            for idx in sorted(results_buffer.keys()):
                k, v = results_buffer[idx]
                output_data["track"][k] = v

            final_filename = f"{i_id}.json"
            final_path = os.path.join(season_dir, final_filename)

            counter = 1
            while os.path.exists(final_path):
                final_path = os.path.join(season_dir, f"{i_id}_{counter}.json")
                counter += 1

            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()