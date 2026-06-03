import argparse
import os

from data_pipeline import (
    fetch_amap_pois,
    make_euclidean_commute_matrix,
    osm_commute_matrix_from_graph,
    osm_graph_from_bbox,
    osm_graph_from_place,
    write_matrix_csv,
    write_pois_csv,
)
from route_methods import load_pois_csv


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    amap = subparsers.add_parser("fetch-amap-pois")
    amap.add_argument("--api-key", required=True)
    amap.add_argument("--city", required=True)
    amap.add_argument("--keywords", required=True)
    amap.add_argument("--types", default="110000")
    amap.add_argument("--pages", type=int, default=1)
    amap.add_argument("--offset", type=int, default=25)
    amap.add_argument("--default-dwell-time", type=float, default=2.0)
    amap.add_argument("--output", required=True)

    euclidean = subparsers.add_parser("euclidean-matrix")
    euclidean.add_argument("--pois", required=True)
    euclidean.add_argument("--speed-kmh", type=float, default=50.0)
    euclidean.add_argument("--output", required=True)

    osm_place = subparsers.add_parser("osm-place-matrix")
    osm_place.add_argument("--pois", required=True)
    osm_place.add_argument("--place", required=True)
    osm_place.add_argument("--network-type", default="drive")
    osm_place.add_argument("--speed-kmh", type=float, default=50.0)
    osm_place.add_argument("--output", required=True)

    osm_bbox = subparsers.add_parser("osm-bbox-matrix")
    osm_bbox.add_argument("--pois", required=True)
    osm_bbox.add_argument("--north", type=float, required=True)
    osm_bbox.add_argument("--south", type=float, required=True)
    osm_bbox.add_argument("--east", type=float, required=True)
    osm_bbox.add_argument("--west", type=float, required=True)
    osm_bbox.add_argument("--network-type", default="drive")
    osm_bbox.add_argument("--speed-kmh", type=float, default=50.0)
    osm_bbox.add_argument("--output", required=True)

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.command == "fetch-amap-pois":
        pois = fetch_amap_pois(
            api_key=args.api_key,
            city=args.city,
            keywords=args.keywords,
            types=args.types,
            offset=args.offset,
            pages=args.pages,
            default_dwell_time=args.default_dwell_time,
        )
        write_pois_csv(pois, args.output)
        return

    pois = load_pois_csv(args.pois)
    if args.command == "euclidean-matrix":
        matrix = make_euclidean_commute_matrix(pois, args.speed_kmh)
    elif args.command == "osm-place-matrix":
        graph = osm_graph_from_place(args.place, network_type=args.network_type)
        matrix = osm_commute_matrix_from_graph(graph, pois, speed_kmh=args.speed_kmh)
    elif args.command == "osm-bbox-matrix":
        graph = osm_graph_from_bbox(
            args.north,
            args.south,
            args.east,
            args.west,
            network_type=args.network_type,
        )
        matrix = osm_commute_matrix_from_graph(graph, pois, speed_kmh=args.speed_kmh)
    else:
        raise ValueError(args.command)
    write_matrix_csv(matrix, args.output)


if __name__ == "__main__":
    main()
