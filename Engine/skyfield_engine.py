# skyfield_engine.py

from skyfield.api import load, Topos
from skyfield import almanac
from skyfield.framelib import ecliptic_frame
from datetime import datetime
import math

ts = load.timescale()
eph = load('de440.bsp')

planets = {
    "sun": eph["sun"],
    "moon": eph["moon"],
    "mercury": eph["mercury"],
    "venus": eph["venus"],
    "mars": eph["mars barycenter"],
    "jupiter": eph["jupiter barycenter"],
    "saturn": eph["saturn barycenter"],
    "uranus": eph["uranus barycenter"],
    "neptune": eph["neptune barycenter"]
}


def compute_skyfield(lat, lon, iso_date):

    dt = datetime.strptime(iso_date, "%Y-%m-%dT%H:%M:%SZ")

    t = ts.utc(dt.year, dt.month, dt.day,
               dt.hour, dt.minute, dt.second)

    observer = eph['earth'] + Topos(
        latitude_degrees=lat,
        longitude_degrees=lon
    )

    bodies_data = {}

    for name, body in planets.items():
        astrometric = observer.at(t).observe(body).apparent()

        ra, dec, _ = astrometric.radec()
        alt, az, _ = astrometric.altaz()
        ecl_lat, ecl_lon, _ = astrometric.frame_latlon(ecliptic_frame)

        alt_rad = math.radians(alt.degrees)
        az_rad = math.radians(az.degrees)

        bodies_data[name] = {
            "ra_hours": ra.hours,
            "dec_deg": dec.degrees,
            "ecliptic_lon_deg": ecl_lon.degrees,
            "ecliptic_lat_deg": ecl_lat.degrees,
            "altitude_deg": alt.degrees,
            "azimuth_deg": az.degrees,
            "sin_alt": math.sin(alt_rad),
            "cos_alt": math.cos(alt_rad),
            "sin_az": math.sin(az_rad),
            "cos_az": math.cos(az_rad)
        }

    # Lunar node events for that month
    t0 = ts.utc(dt.year, dt.month, 1)
    t1 = ts.utc(dt.year, dt.month + 1, 1) if dt.month < 12 else ts.utc(dt.year + 1, 1, 1)

    f = almanac.moon_nodes(eph)
    times, events = almanac.find_discrete(t0, t1, f)

    node_events = []
    for ti, yi in zip(times, events):
        node_events.append({
            "time_utc": ti.utc_iso(),
            "type": almanac.MOON_NODES[yi]
        })

    return {
        "datetime_utc": iso_date,
        "location": {
            "latitude": lat,
            "longitude": lon
        },
        "bodies": bodies_data,
        "lunar_node_events_in_month": node_events
    }