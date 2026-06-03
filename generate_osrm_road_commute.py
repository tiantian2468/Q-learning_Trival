import argparse
import csv
import json
import math
import urllib.request
from pathlib import Path
from typing import List

from route_methods import POI, load_pois_csv


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


def fetch_osrm_duration_hours(pois: List[POI], speed_kmh: float) -> List[List[float]]:
    coords = ";".join(f"{poi.lon},{poi.lat}" for poi in pois)
    url = (
        "http://router.project-osrm.org/table/v1/driving/"
        f"{coords}?annotations=duration"
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("code") != "Ok":
        raise RuntimeError(f"OSRM request failed: {payload}")

    durations = payload.get("durations")
    if not durations:
        raise RuntimeError(f"OSRM returned no duration matrix: {payload}")

    matrix: List[List[float]] = []
    for i, row in enumerate(durations):
        matrix_row = []
        for j, seconds in enumerate(row):
            if i == j:
                matrix_row.append(0.0)
            elif seconds is None:
                fallback_hours = (
                    haversine_km(pois[i].lat, pois[i].lon, pois[j].lat, pois[j].lon)
                    / speed_kmh
                )
                matrix_row.append(round(fallback_hours, 6))
            else:
                matrix_row.append(round(float(seconds) / 3600.0, 6))
        matrix.append(matrix_row)
    return matrix


def write_matrix_csv(matrix: List[List[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate route_methods-ready road_commute.csv from OSRM durations."
    )
    parser.add_argument("--pois", required=True, help="Input pois.csv path.")
    parser.add_argument("--output", required=True, help="Output road_commute.csv path.")
    parser.add_argument(
        "--fallback-speed-kmh",
        type=float,
        default=50.0,
        help="Fallback speed for unreachable OSRM pairs.",
    )
    args = parser.parse_args()

    pois = load_pois_csv(args.pois)
    matrix = fetch_osrm_duration_hours(pois, speed_kmh=args.fallback_speed_kmh)
    write_matrix_csv(matrix, Path(args.output))


if __name__ == "__main__":
    main()
