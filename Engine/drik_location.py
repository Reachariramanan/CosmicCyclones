# drik_location.py

import json
import math


def haversine(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points on earth (in km)"""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat/2)**2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dlon/2)**2
    )

    return 2 * R * math.asin(math.sqrt(a))


def load_geonames(geo_file):
    """Load geonames data once"""
    with open(geo_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["geonames"]


def find_nearest_location(lat, lon, geonames_list):
    """Find nearest location from pre-loaded geonames list"""
    nearest = None
    min_distance = float("inf")

    for item in geonames_list:
        glat = float(item["latitude"])
        glon = float(item["longitude"])

        distance = haversine(lat, lon, glat, glon)

        if distance < min_distance:
            min_distance = distance
            nearest = item

    return nearest