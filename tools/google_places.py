"""
Google Places API tool — finds nearest hospitals ranked by distance.
Used by the Resource Agent.
"""
import httpx
from typing import List
from models.output_models import HospitalInfo
from config import settings


async def find_nearest_hospitals(
    latitude: float,
    longitude: float,
    radius_meters: int = 10000,
    max_results: int = 3,
) -> List[HospitalInfo]:
    """
    Calls Google Places Nearby Search (v1) to find hospitals.
    Returns a ranked list by distance with ETA estimates.
    """
    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": settings.google_places_api_key,
        "X-Goog-FieldMask": (
            "places.displayName,places.formattedAddress,"
            "places.location,places.internationalPhoneNumber"
        ),
    }
    body = {
        "includedTypes": ["hospital"],  # only one type, "emergency_room" is invalid
        "maxResultCount": max_results,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": float(radius_meters),
            }
        },
        "rankPreference": "DISTANCE",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code != 200:
            print(f"[Places API Error] {resp.status_code}: {resp.text}")  # remove after fixing
        resp.raise_for_status()
        data = resp.json()


    hospitals = []
    for place in data.get("places", []):
        loc = place.get("location", {})
        dist_km = _haversine(latitude, longitude, loc.get("latitude", 0), loc.get("longitude", 0))
        hospitals.append(
            HospitalInfo(
                name=place["displayName"]["text"],
                address=place.get("formattedAddress", ""),
                distance_km=round(dist_km, 2),
                eta_minutes=max(1, int(dist_km / 0.5)),  # ~30 km/h ambulance estimate
                phone=place.get("internationalPhoneNumber"),
            )
        )
    return sorted(hospitals, key=lambda h: h.distance_km)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))
