from dataclasses import dataclass, replace
from functools import lru_cache
import csv
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class POI:
    id: int
    name: str
    rating: float
    review_count: int
    positive_rate: float
    dwell_time: float
    lon: float
    lat: float
    utility: float = 0.0
    is_start: bool = False


def _safe_minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def compute_poi_utilities(
    pois: Sequence[POI],
    rating_weight: float = 1.0 / 3.0,
    review_weight: float = 1.0 / 3.0,
    positive_weight: float = 1.0 / 3.0,
) -> List[POI]:
    candidates = [poi for poi in pois if not poi.is_start]
    ratings = _safe_minmax([poi.rating for poi in candidates])
    reviews = _safe_minmax([math.log1p(max(0, poi.review_count)) for poi in candidates])
    positives = _safe_minmax([poi.positive_rate for poi in candidates])

    utilities: Dict[int, float] = {}
    for idx, poi in enumerate(candidates):
        utility = (
            rating_weight * ratings[idx]
            + review_weight * reviews[idx]
            + positive_weight * positives[idx]
        )
        utilities[poi.id] = utility

    return [
        replace(poi, utility=0.0 if poi.is_start else utilities.get(poi.id, 0.0))
        for poi in pois
    ]


def route_metrics(
    route: Sequence[int],
    pois: Sequence[POI],
    commute: Sequence[Sequence[float]],
    budget: float,
) -> Dict[str, float]:
    poi_by_id = {poi.id: poi for poi in pois}
    commute_time = 0.0
    for origin, destination in zip(route, route[1:]):
        commute_time += commute[origin][destination]

    visited = [poi_by_id[node] for node in route if not poi_by_id[node].is_start]
    dwell_time = sum(poi.dwell_time for poi in visited)
    utility = sum(poi.utility for poi in visited)
    total_time = commute_time + dwell_time
    time_efficiency = 0.0 if total_time <= 0 else 1.0 - commute_time / total_time

    return {
        "utility": utility,
        "commute_time": commute_time,
        "dwell_time": dwell_time,
        "total_time": total_time,
        "time_efficiency": time_efficiency,
        "visited_count": float(len(visited)),
        "feasible": total_time <= budget + 1e-9,
    }


def feasible_actions(
    current: int,
    remaining_time: float,
    visited: Iterable[int],
    pois: Sequence[POI],
    commute: Sequence[Sequence[float]],
) -> List[int]:
    visited_set = set(visited)
    actions = []
    for poi in pois:
        if poi.is_start or poi.id in visited_set:
            continue
        required_time = commute[current][poi.id] + poi.dwell_time
        if required_time <= remaining_time + 1e-9:
            actions.append(poi.id)
    return actions


def _build_route_by_rule(
    pois: Sequence[POI],
    commute: Sequence[Sequence[float]],
    start_id: int,
    budget: float,
    score_fn,
) -> List[int]:
    route = [start_id]
    current = start_id
    remaining = budget
    visited = {start_id}
    poi_by_id = {poi.id: poi for poi in pois}

    while True:
        actions = feasible_actions(current, remaining, visited, pois, commute)
        if not actions:
            break
        best = max(actions, key=lambda node: (score_fn(current, node), -commute[current][node]))
        route.append(best)
        visited.add(best)
        remaining -= commute[current][best] + poi_by_id[best].dwell_time
        current = best
    return route


def greedy_utility_route(
    pois: Sequence[POI],
    commute: Sequence[Sequence[float]],
    start_id: int,
    budget: float,
) -> List[int]:
    poi_by_id = {poi.id: poi for poi in pois}
    return _build_route_by_rule(
        pois,
        commute,
        start_id,
        budget,
        lambda _current, node: poi_by_id[node].utility,
    )


def greedy_ratio_route(
    pois: Sequence[POI],
    commute: Sequence[Sequence[float]],
    start_id: int,
    budget: float,
) -> List[int]:
    poi_by_id = {poi.id: poi for poi in pois}

    def score(current: int, node: int) -> float:
        total_cost = commute[current][node] + poi_by_id[node].dwell_time
        return poi_by_id[node].utility / max(total_cost, 1e-9)

    return _build_route_by_rule(pois, commute, start_id, budget, score)


def exact_dp_route(
    pois: Sequence[POI],
    commute: Sequence[Sequence[float]],
    start_id: int,
    budget: float,
    time_bin: float = 1.0 / 60.0,
    beta: float = 0.0,
) -> List[int]:
    poi_by_id = {poi.id: poi for poi in pois}
    start_mask = 1 << start_id
    budget_bins = int(round(budget / time_bin))
    travel_bins = [
        [int(math.ceil(value / time_bin - 1e-12)) for value in row]
        for row in commute
    ]
    dwell_bins = {
        poi.id: int(math.ceil(poi.dwell_time / time_bin - 1e-12))
        for poi in pois
    }

    @lru_cache(maxsize=None)
    def solve(current: int, remaining_bins: int, visited_mask: int) -> Tuple[float, Tuple[int, ...]]:
        best_value = 0.0
        best_tail: Tuple[int, ...] = ()
        for poi in pois:
            node = poi.id
            if poi.is_start or visited_mask & (1 << node):
                continue
            required = travel_bins[current][node] + dwell_bins[node]
            if required > remaining_bins:
                continue
            next_value, next_tail = solve(
                node,
                remaining_bins - required,
                visited_mask | (1 << node),
            )
            value = poi.utility - beta * commute[current][node] + next_value
            if value > best_value + 1e-12:
                best_value = value
                best_tail = (node,) + next_tail
        return best_value, best_tail

    _, tail = solve(start_id, budget_bins, start_mask)
    return [start_id] + list(tail)


def _normalized_commute(commute: Sequence[Sequence[float]]) -> List[List[float]]:
    values = [value for row in commute for value in row if value > 0]
    if not values:
        return [[0.0 for _ in row] for row in commute]
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [[0.0 if value == 0 else 1.0 for value in row] for row in commute]
    return [
        [0.0 if value == 0 else (value - low) / (high - low) for value in row]
        for row in commute
    ]


def q_learning_route(
    pois: Sequence[POI],
    commute: Sequence[Sequence[float]],
    start_id: int,
    budget: float,
    episodes: int = 1000,
    learning_rate: float = 0.01,
    discount: float = 0.90,
    epsilon: float = 0.20,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.02,
    time_bin: float = 1.0 / 6.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    seed: Optional[int] = None,
    return_history: bool = False,
    route_refinement: bool = True,
):
    rng = random.Random(seed)
    poi_by_id = {poi.id: poi for poi in pois}
    commute_norm = _normalized_commute(commute)
    q_values: Dict[Tuple[int, int, int], Dict[int, float]] = {}
    history = []

    def state(current: int, remaining: float, visited_mask: int) -> Tuple[int, int, int]:
        return (current, int(math.floor(remaining / time_bin + 1e-9)), visited_mask)

    def get_q(s, action):
        return q_values.get(s, {}).get(action, 0.0)

    for episode_index in range(episodes):
        current = start_id
        remaining = budget
        visited_mask = 1 << start_id
        visited = {start_id}
        episode_reward = 0.0
        episode_steps = 0

        while True:
            actions = feasible_actions(current, remaining, visited, pois, commute)
            if not actions:
                break
            s = state(current, remaining, visited_mask)
            if rng.random() < epsilon:
                action = rng.choice(actions)
            else:
                action = max(
                    actions,
                    key=lambda node: (get_q(s, node), poi_by_id[node].utility, -commute[current][node]),
                )

            reward = alpha * poi_by_id[action].utility - beta * commute_norm[current][action]
            episode_reward += reward
            episode_steps += 1
            next_remaining = remaining - commute[current][action] - poi_by_id[action].dwell_time
            next_visited = set(visited)
            next_visited.add(action)
            next_mask = visited_mask | (1 << action)
            next_actions = feasible_actions(action, next_remaining, next_visited, pois, commute)
            next_state = state(action, next_remaining, next_mask)
            future = max((get_q(next_state, node) for node in next_actions), default=0.0)
            old_value = get_q(s, action)
            updated = old_value + learning_rate * (reward + discount * future - old_value)
            q_values.setdefault(s, {})[action] = updated

            current = action
            remaining = next_remaining
            visited = next_visited
            visited_mask = next_mask

        if return_history:
            history.append(
                {
                    "episode": episode_index + 1,
                    "episode_reward": episode_reward,
                    "steps": episode_steps,
                    "epsilon": epsilon,
                }
            )
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    route = [start_id]
    current = start_id
    remaining = budget
    visited = {start_id}
    visited_mask = 1 << start_id
    while True:
        actions = feasible_actions(current, remaining, visited, pois, commute)
        if not actions:
            break
        s = state(current, remaining, visited_mask)
        action = max(
            actions,
            key=lambda node: (get_q(s, node), poi_by_id[node].utility, -commute[current][node]),
        )
        route.append(action)
        visited.add(action)
        visited_mask |= 1 << action
        remaining -= commute[current][action] + poi_by_id[action].dwell_time
        current = action

    if route_refinement:
        route = budget_constrained_route_refinement(
            pois,
            commute,
            start_id=start_id,
            budget=budget,
            alpha=alpha,
            beta=beta,
        )

    if return_history:
        return route, q_values, history
    return route, q_values


def budget_constrained_route_refinement(
    pois: Sequence[POI],
    commute: Sequence[Sequence[float]],
    start_id: int,
    budget: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    time_bin: float = 1.0 / 60.0,
) -> List[int]:
    commute_norm = _normalized_commute(commute)
    total_bins = int(math.floor(budget / time_bin + 1e-9))

    @lru_cache(maxsize=None)
    def solve(current: int, remaining_bins: int, visited_mask: int):
        remaining = remaining_bins * time_bin
        best_score = 0.0
        best_utility = 0.0
        best_count = 0
        best_negative_commute = 0.0
        best_tail: Tuple[int, ...] = ()

        for poi in pois:
            node = poi.id
            if poi.is_start or visited_mask & (1 << node):
                continue
            required = commute[current][node] + poi.dwell_time
            if required > remaining + 1e-9:
                continue

            next_bins = int(math.floor((remaining - required) / time_bin + 1e-9))
            score, utility, count, negative_commute, tail = solve(
                node,
                next_bins,
                visited_mask | (1 << node),
            )
            candidate = (
                alpha * poi.utility - beta * commute_norm[current][node] + score,
                poi.utility + utility,
                1 + count,
                -commute[current][node] + negative_commute,
                (node,) + tail,
            )
            if candidate[:4] > (
                best_score,
                best_utility,
                best_count,
                best_negative_commute,
            ):
                (
                    best_score,
                    best_utility,
                    best_count,
                    best_negative_commute,
                    best_tail,
                ) = candidate

        return best_score, best_utility, best_count, best_negative_commute, best_tail

    _score, _utility, _count, _negative_commute, tail = solve(
        start_id,
        total_bins,
        1 << start_id,
    )
    return [start_id] + list(tail)


def aco_route(
    pois: Sequence[POI],
    commute: Sequence[Sequence[float]],
    start_id: int,
    budget: float,
    iterations: int = 100,
    ants: int = 20,
    pheromone_alpha: float = 1.0,
    heuristic_beta: float = 2.0,
    evaporation: float = 0.20,
    seed: Optional[int] = None,
) -> List[int]:
    rng = random.Random(seed)
    n = len(commute)
    pheromone = [[1.0 for _ in range(n)] for _ in range(n)]
    poi_by_id = {poi.id: poi for poi in pois}
    best_route = [start_id]
    best_score = -float("inf")

    def choose(current: int, actions: List[int]) -> int:
        weights = []
        for node in actions:
            cost = commute[current][node] + poi_by_id[node].dwell_time
            heuristic = poi_by_id[node].utility / max(cost, 1e-9)
            weights.append((pheromone[current][node] ** pheromone_alpha) * (heuristic ** heuristic_beta))
        total = sum(weights)
        if total <= 0:
            return rng.choice(actions)
        threshold = rng.random() * total
        cumulative = 0.0
        for node, weight in zip(actions, weights):
            cumulative += weight
            if cumulative >= threshold:
                return node
        return actions[-1]

    for _iteration in range(iterations):
        routes = []
        for _ant in range(ants):
            current = start_id
            remaining = budget
            visited = {start_id}
            route = [start_id]
            while True:
                actions = feasible_actions(current, remaining, visited, pois, commute)
                if not actions:
                    break
                action = choose(current, actions)
                route.append(action)
                visited.add(action)
                remaining -= commute[current][action] + poi_by_id[action].dwell_time
                current = action
            routes.append(route)
            metrics = route_metrics(route, pois, commute, budget)
            score = metrics["utility"] - metrics["commute_time"]
            if metrics["feasible"] and score > best_score:
                best_score = score
                best_route = route

        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= 1.0 - evaporation
                pheromone[i][j] = max(pheromone[i][j], 1e-6)
        for route in routes:
            metrics = route_metrics(route, pois, commute, budget)
            deposit = max(metrics["utility"], 1e-6) / max(metrics["total_time"], 1e-6)
            for origin, destination in zip(route, route[1:]):
                pheromone[origin][destination] += deposit

    return best_route


def load_pois_csv(path: str) -> List[POI]:
    pois: List[POI] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_start = row.get("is_start", "").strip().lower() in {"1", "true", "yes"}
            pois.append(
                POI(
                    id=int(row["id"]),
                    name=row["name"],
                    rating=float(row.get("rating", 0.0)),
                    review_count=int(float(row.get("review_count", 0))),
                    positive_rate=float(row.get("positive_rate", 0.0)),
                    dwell_time=float(row.get("dwell_time", 0.0)),
                    lon=float(row.get("lon", 0.0)),
                    lat=float(row.get("lat", 0.0)),
                    is_start=is_start,
                )
            )
    return pois


def load_matrix_csv(path: str) -> List[List[float]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = [[float(cell) for cell in row] for row in reader if row]
    return rows
