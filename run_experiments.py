import argparse
import csv
import os
import statistics
from typing import Dict, List

from route_methods import (
    aco_route,
    compute_poi_utilities,
    exact_dp_route,
    greedy_ratio_route,
    greedy_utility_route,
    load_matrix_csv,
    load_pois_csv,
    q_learning_route,
    route_metrics,
)


def run_main_comparison(pois, commute, start_id: int, budget: float, episodes: int, seed: int):
    q_route, _ = q_learning_route(
        pois,
        commute,
        start_id=start_id,
        budget=budget,
        episodes=episodes,
        seed=seed,
    )
    routes = {
        "Ours": q_route,
        "Greedy-Utility": greedy_utility_route(pois, commute, start_id, budget),
        "Greedy-Ratio": greedy_ratio_route(pois, commute, start_id, budget),
        "ACO": aco_route(pois, commute, start_id, budget, seed=seed),
    }
    return [_row_for_route(method, route, pois, commute, budget) for method, route in routes.items()]


def run_dp_comparison(pois, commute, start_id: int, budgets: List[float], episodes: int, seed: int):
    rows = []
    for budget in budgets:
        q_route, _ = q_learning_route(
            pois,
            commute,
            start_id=start_id,
            budget=budget,
            episodes=episodes,
            seed=seed,
        )
        dp_route = exact_dp_route(pois, commute, start_id, budget)
        q_metrics = route_metrics(q_route, pois, commute, budget)
        dp_metrics = route_metrics(dp_route, pois, commute, budget)
        gap = 0.0
        if dp_metrics["utility"] > 0:
            gap = (dp_metrics["utility"] - q_metrics["utility"]) / dp_metrics["utility"] * 100.0
        rows.append(
            {
                "budget_h": budget,
                "dp_route": "-".join(map(str, dp_route)),
                "q_route": "-".join(map(str, q_route)),
                "dp_utility": dp_metrics["utility"],
                "q_utility": q_metrics["utility"],
                "optimality_gap_percent": gap,
                "dp_commute_time": dp_metrics["commute_time"],
                "q_commute_time": q_metrics["commute_time"],
            }
        )
    return rows


def run_budget_sensitivity(
    pois,
    commute,
    start_id: int,
    budgets: List[float],
    episodes: int,
    seed: int,
    methods: List[str] = None,
):
    if methods is None:
        methods = ["Ours", "Greedy-Utility", "Greedy-Ratio", "ACO"]
    return run_budget_sensitivity_for_methods(
        pois,
        commute,
        start_id=start_id,
        budgets=budgets,
        episodes=episodes,
        seed=seed,
        methods=methods,
    )


def run_budget_sensitivity_for_methods(
    pois,
    commute,
    start_id: int,
    budgets: List[float],
    episodes: int,
    seed: int,
    methods: List[str],
):
    rows = []
    for budget in budgets:
        for method in methods:
            route = _route_for_method(method, pois, commute, start_id, budget, episodes, seed)
            row = _row_for_route(method, route, pois, commute, budget)
            row["budget_h"] = budget
            rows.append(row)
    return rows


def run_utility_weight_sensitivity(raw_pois, commute, start_id: int, budget: float, episodes: int, seed: int):
    rows = []
    scenarios = [
        ("equal", 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ("rating-dominant", 0.60, 0.20, 0.20),
        ("review-dominant", 0.20, 0.60, 0.20),
        ("positive-dominant", 0.20, 0.20, 0.60),
    ]
    for label, rating_weight, review_weight, positive_weight in scenarios:
        pois = compute_poi_utilities(
            raw_pois,
            rating_weight=rating_weight,
            review_weight=review_weight,
            positive_weight=positive_weight,
        )
        route, _ = q_learning_route(
            pois,
            commute,
            start_id=start_id,
            budget=budget,
            episodes=episodes,
            seed=seed,
        )
        row = _row_for_route("Ours", route, pois, commute, budget)
        row["scenario"] = label
        row["rating_weight"] = rating_weight
        row["review_weight"] = review_weight
        row["positive_weight"] = positive_weight
        rows.append(row)
    return rows


def run_weight_sensitivity(pois, commute, start_id: int, budget: float, episodes: int, seed: int):
    rows = []
    for alpha, beta in [(1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]:
        route, _ = q_learning_route(
            pois,
            commute,
            start_id=start_id,
            budget=budget,
            episodes=episodes,
            seed=seed,
            alpha=alpha,
            beta=beta,
        )
        row = _row_for_route("Ours", route, pois, commute, budget)
        row["alpha"] = alpha
        row["beta"] = beta
        rows.append(row)
    return rows


def run_distance_ablation(
    pois,
    road_commute,
    euclidean_commute,
    start_id: int,
    budget: float,
    episodes: int,
    seed: int,
):
    rows = []
    for label, planning_matrix in [
        ("Road-network", road_commute),
        ("Euclidean", euclidean_commute),
    ]:
        route, _ = q_learning_route(
            pois,
            planning_matrix,
            start_id=start_id,
            budget=budget,
            episodes=episodes,
            seed=seed,
        )
        planning_metrics = route_metrics(route, pois, planning_matrix, budget)
        road_metrics = route_metrics(route, pois, road_commute, budget)
        rows.append(
            {
                "distance_model": label,
                "route": "-".join(map(str, route)),
                "planned_commute_time": planning_metrics["commute_time"],
                "road_verified_commute_time": road_metrics["commute_time"],
                "utility": road_metrics["utility"],
                "feasible_under_road_budget": road_metrics["feasible"],
            }
        )
    return rows


def run_robustness(pois, commute, start_id: int, budgets: List[float], episodes: int, seeds: List[int]):
    rows = []
    for budget in budgets:
        metrics = []
        for seed in seeds:
            route, _ = q_learning_route(
                pois,
                commute,
                start_id=start_id,
                budget=budget,
                episodes=episodes,
                seed=seed,
            )
            metrics.append(route_metrics(route, pois, commute, budget))
        rows.append(
            {
                "budget_h": budget,
                "utility_mean": _mean(metrics, "utility"),
                "utility_std": _std(metrics, "utility"),
                "commute_time_mean": _mean(metrics, "commute_time"),
                "commute_time_std": _std(metrics, "commute_time"),
                "visited_count_mean": _mean(metrics, "visited_count"),
                "visited_count_std": _std(metrics, "visited_count"),
                "time_efficiency_mean": _mean(metrics, "time_efficiency"),
                "time_efficiency_std": _std(metrics, "time_efficiency"),
            }
        )
    return rows


def run_training_history(pois, commute, start_id: int, budget: float, episodes: int, seeds: List[int]):
    rows = []
    for seed in seeds:
        _route, _q_values, history = q_learning_route(
            pois,
            commute,
            start_id=start_id,
            budget=budget,
            episodes=episodes,
            seed=seed,
            return_history=True,
        )
        for row in history:
            output = dict(row)
            output["seed"] = seed
            output["budget_h"] = budget
            rows.append(output)
    return rows


def write_table(rows: List[Dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row_for_route(method: str, route, pois, commute, budget: float):
    metrics = route_metrics(route, pois, commute, budget)
    return {
        "method": method,
        "route": "-".join(map(str, route)),
        "utility": metrics["utility"],
        "total_time": metrics["total_time"],
        "dwell_time": metrics["dwell_time"],
        "commute_time": metrics["commute_time"],
        "time_efficiency": metrics["time_efficiency"],
        "visited_count": int(metrics["visited_count"]),
        "feasible": metrics["feasible"],
    }


def _route_for_method(method: str, pois, commute, start_id: int, budget: float, episodes: int, seed: int):
    if method == "Ours":
        route, _ = q_learning_route(
            pois,
            commute,
            start_id=start_id,
            budget=budget,
            episodes=episodes,
            seed=seed,
        )
        return route
    if method == "Greedy-Utility":
        return greedy_utility_route(pois, commute, start_id, budget)
    if method == "Greedy-Ratio":
        return greedy_ratio_route(pois, commute, start_id, budget)
    if method == "ACO":
        return aco_route(pois, commute, start_id, budget, seed=seed)
    if method == "Exact/DP":
        return exact_dp_route(pois, commute, start_id, budget)
    raise ValueError(f"Unsupported method: {method}")


def _mean(rows: List[Dict], key: str) -> float:
    return statistics.mean(row[key] for row in rows)


def _std(rows: List[Dict], key: str) -> float:
    if len(rows) < 2:
        return 0.0
    return statistics.stdev(row[key] for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pois", required=True)
    parser.add_argument("--road-commute", required=True)
    parser.add_argument("--euclidean-commute")
    parser.add_argument("--output-dir", default="experiment_outputs")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--budgets", default="6,8,10,12")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    args = parser.parse_args()

    raw_pois = load_pois_csv(args.pois)
    pois = compute_poi_utilities(raw_pois)
    road_commute = load_matrix_csv(args.road_commute)
    budgets = [float(value) for value in args.budgets.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    seed = seeds[0] if seeds else 1

    write_table(
        run_main_comparison(pois, road_commute, args.start_id, args.budget, args.episodes, seed),
        os.path.join(args.output_dir, "table4_main_comparison.csv"),
    )
    write_table(
        run_dp_comparison(pois, road_commute, args.start_id, budgets, args.episodes, seed),
        os.path.join(args.output_dir, "table5_dp_comparison.csv"),
    )
    write_table(
        run_budget_sensitivity(pois, road_commute, args.start_id, budgets, args.episodes, seed),
        os.path.join(args.output_dir, "table6_budget_sensitivity.csv"),
    )
    write_table(
        run_weight_sensitivity(pois, road_commute, args.start_id, args.budget, args.episodes, seed),
        os.path.join(args.output_dir, "table7_weight_sensitivity.csv"),
    )
    write_table(
        run_utility_weight_sensitivity(
            raw_pois,
            road_commute,
            args.start_id,
            args.budget,
            args.episodes,
            seed,
        ),
        os.path.join(args.output_dir, "table7b_utility_weight_sensitivity.csv"),
    )
    if args.euclidean_commute:
        euclidean_commute = load_matrix_csv(args.euclidean_commute)
        write_table(
            run_distance_ablation(
                pois,
                road_commute,
                euclidean_commute,
                args.start_id,
                args.budget,
                args.episodes,
                seed,
            ),
            os.path.join(args.output_dir, "table8_distance_ablation.csv"),
        )
    write_table(
        run_robustness(pois, road_commute, args.start_id, budgets, args.episodes, seeds),
        os.path.join(args.output_dir, "table9_robustness.csv"),
    )
    write_table(
        run_training_history(pois, road_commute, args.start_id, args.budget, args.episodes, seeds),
        os.path.join(args.output_dir, "figure9_training_history.csv"),
    )


if __name__ == "__main__":
    main()
