import argparse
import csv
import os
from typing import Sequence

from route_methods import POI, compute_poi_utilities, load_pois_csv


def export_poi_utility_table(
    pois_path: str,
    output_path: str,
    rating_weight: float = 1.0 / 3.0,
    review_weight: float = 1.0 / 3.0,
    positive_weight: float = 1.0 / 3.0,
) -> str:
    pois = compute_poi_utilities(
        load_pois_csv(pois_path),
        rating_weight=rating_weight,
        review_weight=review_weight,
        positive_weight=positive_weight,
    )
    write_poi_utility_table(pois, output_path)
    return output_path


def write_poi_utility_table(pois: Sequence[POI], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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
        "utility",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
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
                    "utility": poi.utility,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export POI attributes with computed utility.")
    parser.add_argument("--pois", required=True, help="Input pois.csv path.")
    parser.add_argument("--output", required=True, help="Output table2_poi_utility.csv path.")
    parser.add_argument("--rating-weight", type=float, default=1.0 / 3.0)
    parser.add_argument("--review-weight", type=float, default=1.0 / 3.0)
    parser.add_argument("--positive-weight", type=float, default=1.0 / 3.0)
    args = parser.parse_args()

    export_poi_utility_table(
        args.pois,
        args.output,
        rating_weight=args.rating_weight,
        review_weight=args.review_weight,
        positive_weight=args.positive_weight,
    )


if __name__ == "__main__":
    main()
