import argparse
import csv
import json
import os
import urllib.parse
import urllib.request
from typing import Dict, List, Sequence

from plot_chapter4_figures import read_csv_rows, route_from_table4


def generate_route_geometry(
    pois_path: str,
    route: Sequence[int],
    output_path: str,
    base_url: str = "http://router.project-osrm.org",
) -> str:
    poi_rows = read_csv_rows(pois_path)
    by_id = {int(row["id"]): row for row in poi_rows}
    geometry_rows = []

    for segment_index, (origin, destination) in enumerate(zip(route, route[1:]), start=1):
        if origin not in by_id or destination not in by_id:
            continue
        points = fetch_segment_geometry(by_id[origin], by_id[destination], base_url=base_url)
        for point_index, (lon, lat) in enumerate(points):
            geometry_rows.append(
                {
                    "segment": segment_index,
                    "point_index": point_index,
                    "origin": origin,
                    "destination": destination,
                    "lon": lon,
                    "lat": lat,
                }
            )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["segment", "point_index", "origin", "destination", "lon", "lat"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(geometry_rows)
    return output_path


def fetch_segment_geometry(origin: Dict[str, str], destination: Dict[str, str], base_url: str) -> List[List[float]]:
    coords = f'{origin["lon"]},{origin["lat"]};{destination["lon"]},{destination["lat"]}'
    params = urllib.parse.urlencode({"overview": "full", "geometries": "geojson"})
    url = f"{base_url}/route/v1/driving/{coords}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("code") == "Ok" and payload.get("routes"):
            coordinates = payload["routes"][0]["geometry"]["coordinates"]
            if coordinates:
                return coordinates
    except Exception:
        pass
    return [
        [float(origin["lon"]), float(origin["lat"])],
        [float(destination["lon"]), float(destination["lat"])],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OSRM route geometry CSV for a route in table4.")
    parser.add_argument("--pois", required=True)
    parser.add_argument("--table4", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="Ours")
    parser.add_argument("--base-url", default="http://router.project-osrm.org")
    args = parser.parse_args()

    route = route_from_table4(read_csv_rows(args.table4), args.method)
    if not route:
        raise ValueError(f"No route found for method {args.method!r} in {args.table4}")
    generate_route_geometry(args.pois, route, args.output, base_url=args.base_url)


if __name__ == "__main__":
    main()
