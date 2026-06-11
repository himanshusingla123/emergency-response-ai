"""
tools/reverse_geocode.py

Google Maps Geocoding / Reverse-Geocoding tool.

Supports two modes:
  • Reverse geocoding  – lat/lng  → structured address
  • Forward geocoding  – address string → lat/lng + structured address

Both modes normalise the response into the same dict shape so that
agents/location.py never has to parse raw Google API responses.

Environment variable required:
    GOOGLE_MAPS_API_KEY   (loaded via config.py / python-dotenv)
"""

import logging
import httpx
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
REQUEST_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_component(components: list[dict], *types: str) -> str:
    """
    Walk the address_components list returned by Google and return the
    long_name of the first component whose types list intersects *types*.
    Returns "" when nothing matches.
    """
    for comp in components:
        if set(comp.get("types", [])) & set(types):
            return comp.get("long_name", "")
    return ""


def _parse_google_result(result: dict) -> dict:
    """
    Convert a single element from Google's `results` array into our
    standardised location dict.
    """
    comps = result.get("address_components", [])
    geo   = result.get("geometry", {}).get("location", {})

    return {
        "location_resolved": True,
        "formatted_address": result.get("formatted_address", ""),
        "street": (
            _extract_component(comps, "street_number") + " " +
            _extract_component(comps, "route")
        ).strip(),
        "area":        _extract_component(comps, "sublocality", "sublocality_level_1",
                                          "neighborhood"),
        "city":        _extract_component(comps, "locality", "postal_town"),
        "state":       _extract_component(comps, "administrative_area_level_1"),
        "country":     _extract_component(comps, "country"),
        "postal_code": _extract_component(comps, "postal_code"),
        "latitude":    geo.get("lat"),
        "longitude":   geo.get("lng"),
        "location_accuracy": "gps",   # overridden by caller for forward geocoding
        "error": None,
    }


def _error_response(message: str) -> dict:
    return {
        "location_resolved": False,
        "formatted_address": None,
        "street":       None,
        "area":         None,
        "city":         None,
        "state":        None,
        "country":      None,
        "postal_code":  None,
        "latitude":     None,
        "longitude":    None,
        "location_accuracy": None,
        "error": message,
    }


# ---------------------------------------------------------------------------
# Public tool function  (called by FunctionTool inside location agent)
# ---------------------------------------------------------------------------

async def reverse_geocode_tool(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    address: Optional[str] = None,
) -> dict:
    """
    Resolve a location using the Google Maps Geocoding API.

    Parameters
    ----------
    latitude  : float, optional
        Decimal latitude  (required for reverse-geocoding mode).
    longitude : float, optional
        Decimal longitude (required for reverse-geocoding mode).
    address   : str, optional
        Plain-text address string (used when coordinates are unavailable).

    Returns
    -------
    dict
        Standardised location dict (see _parse_google_result).
        On failure returns _error_response().
    """
    api_key = settings.google_maps_api_key
    if not api_key:
        logger.error("GOOGLE_MAPS_API_KEY is not configured.")
        return _error_response("Google Maps API key is not configured.")

    # ------------------------------------------------------------------
    # Build query params
    # ------------------------------------------------------------------
    if latitude is not None and longitude is not None:
        params = {
            "latlng": f"{latitude},{longitude}",
            "key": api_key,
        }
        mode = "reverse"
    elif address:
        params = {
            "address": address,
            "key": api_key,
        }
        mode = "forward"
    else:
        return _error_response(
            "reverse_geocode_tool requires either (latitude, longitude) or address."
        )

    # ------------------------------------------------------------------
    # HTTP call
    # ------------------------------------------------------------------
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(GEOCODE_URL, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        logger.warning("Google Geocoding API request timed out.")
        return _error_response("Geocoding request timed out. Please try again.")
    except httpx.HTTPStatusError as exc:
        logger.error("Geocoding API HTTP error: %s", exc)
        return _error_response(f"Geocoding API returned HTTP {exc.response.status_code}.")
    except Exception as exc:
        logger.exception("Unexpected error calling Geocoding API.")
        return _error_response(f"Unexpected error: {str(exc)}")

    # ------------------------------------------------------------------
    # Parse response
    # ------------------------------------------------------------------
    status = data.get("status")

    if status == "OK":
        results = data.get("results", [])
        if not results:
            return _error_response("Geocoding API returned OK but no results.")

        parsed = _parse_google_result(results[0])

        # For forward geocoding, mark accuracy differently
        if mode == "forward":
            parsed["location_accuracy"] = "address"

        logger.info(
            "Location resolved [%s]: %s", mode, parsed["formatted_address"]
        )
        return parsed

    elif status == "ZERO_RESULTS":
        return _error_response(
            "No address found for the given location. "
            "Try providing a more specific address or valid coordinates."
        )
    elif status == "REQUEST_DENIED":
        return _error_response(
            "Google Maps API request was denied. Check your API key and billing."
        )
    elif status == "OVER_DAILY_LIMIT":
        return _error_response("Google Maps API daily quota exceeded.")
    elif status == "INVALID_REQUEST":
        return _error_response(
            f"Invalid request sent to Google Maps API. Params: {params}"
        )
    else:
        return _error_response(f"Google Maps Geocoding API error: {status}")