import csv
import json
import math
import urllib.parse
import urllib.request
from typing import Dict, List, Sequence

from route_methods import POI


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return radius_km * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def make_euclidean_commute_matrix(pois: Sequence[POI], speed_kmh: float) -> List[List[float]]:
    matrix: List[List[float]] = []
    for origin in pois:
        row = []
        for destination in pois:
            if origin.id == destination.id:
                row.append(0.0)
                continue
            distance = haversine_km(origin.lat, origin.lon, destination.lat, destination.lon)
            row.append(distance / speed_kmh)
        matrix.append(row)
    return matrix


def amap_pois_from_response(
    payload: Dict,
    start_index: int = 0,
    default_dwell_time: float = 2.0,
) -> List[POI]:
    pois = []
    for offset, item in enumerate(payload.get("pois", [])):
        location = item.get("location", "")
        lon_text, lat_text = (location.split(",", 1) + ["0", "0"])[:2]
        biz_ext = item.get("biz_ext") or {}
        rating = _safe_float(biz_ext.get("rating"), default=0.0)
        pois.append(
            POI(
                id=start_index + offset,
                name=item.get("name", ""),
                rating=rating,
                review_count=int(_safe_float(item.get("review_count"), default=0.0)),
                positive_rate=_safe_float(item.get("positive_rate"), default=0.0),
                dwell_time=default_dwell_time,
                lon=_safe_float(lon_text, default=0.0),
                lat=_safe_float(lat_text, default=0.0),
            )
        )
    return pois


def fetch_amap_pois(
    api_key: str,
    city: str,
    keywords: str,
    types: str = "110000",
    offset: int = 25,
    pages: int = 1,
    default_dwell_time: float = 2.0,
) -> List[POI]:
    all_pois: List[POI] = []
    for page in range(1, pages + 1):
        params = {
            "key": api_key,
            "city": city,
            "keywords": keywords,
            "types": types,
            "offset": str(offset),
            "page": str(page),
            "extensions": "all",
            "output": "json",
        }
        url = "https://restapi.amap.com/v3/place/text?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        all_pois.extend(
            amap_pois_from_response(
                payload,
                start_index=len(all_pois),
                default_dwell_time=default_dwell_time,
            )
        )
    return all_pois


def osm_graph_from_place(place_name: str, network_type: str = "drive"):
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError("Install osmnx to download and convert OSM road networks.") from exc
    return ox.graph_from_place(place_name, network_type=network_type)


def osm_graph_from_bbox(north: float, south: float, east: float, west: float, network_type: str = "drive"):
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError("Install osmnx to download and convert OSM road networks.") from exc
    return ox.graph_from_bbox(north, south, east, west, network_type=network_type)


def osm_commute_matrix_from_graph(graph, pois: Sequence[POI], speed_kmh: float = 50.0) -> List[List[float]]:
    try:
        import networkx as nx
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError("Install osmnx and networkx to compute road-network commute matrices.") from exc

    nodes = ox.distance.nearest_nodes(
        graph,
        X=[poi.lon for poi in pois],
        Y=[poi.lat for poi in pois],
    )
    matrix: List[List[float]] = []
    for origin_idx, origin_node in enumerate(nodes):
        row = []
        for dest_idx, dest_node in enumerate(nodes):
            if origin_idx == dest_idx:
                row.append(0.0)
                continue
            try:
                distance_m = nx.shortest_path_length(graph, origin_node, dest_node, weight="length")
                row.append((distance_m / 1000.0) / speed_kmh)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                fallback = haversine_km(
                    pois[origin_idx].lat,
                    pois[origin_idx].lon,
                    pois[dest_idx].lat,
                    pois[dest_idx].lon,
                )
                row.append(fallback / speed_kmh)
        matrix.append(row)
    return matrix


def write_pois_csv(pois: Sequence[POI], path: str) -> None:
    fieldnames = [
        "id",
        "name",
        "rating",
        "review_count",
        "positive_rate",
        "dwell_time",
        "lon",
        "lat",
        "is_start",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for poi in pois:
            writer.writerow(
                {
                    "id": poi.id,
                    "name": poi.name,
                    "rating": poi.rating,
                    "review_count": poi.review_count,
                    "positive_rate": poi.positive_rate,
                    "dwell_time": poi.dwell_time,
                    "lon": poi.lon,
                    "lat": poi.lat,
                    "is_start": int(poi.is_start),
                }
            )


def write_matrix_csv(matrix: Sequence[Sequence[float]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow(row)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, "", []):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
