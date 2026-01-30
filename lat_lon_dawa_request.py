import requests

def dawa_geocode(address: str):
    """
    Geocode an address using DAWA and return (lat, lon).
    Raises an error if the address is not found.
    """
    url = "https://api.dataforsyningen.dk/adresser"
    params = {"q": address, "struktur": "mini"}  # "mini" returns a clean compact structure

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    results = r.json()

    if not results:
        raise ValueError(f"No DAWA match found for address: {address}")

    # DAWA returns coordinates as x=lon, y=lat in WGS84
    lon, lat = results[0]["x"], results[0]["y"]

    return lat, lon