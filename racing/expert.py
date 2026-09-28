from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

import numpy as np
import pygame

if TYPE_CHECKING:
    from racing.simulator import RacingGame


class ArtificialExpert:
    """Grid-planning safety teacher for DAgger labels."""

    cell_size = 20

    def __init__(self, label_all: bool = False) -> None:
        self.label_all = label_all
        self.route: list[pygame.Vector2] = []
        self.route_goal: tuple[int, int] | None = None

    def safe(self, game: "RacingGame", point: pygame.Vector2) -> bool:
        # Keep a full car-width margin from track borders and obstacles.
        for offset in (pygame.Vector2(), pygame.Vector2(24, 0), pygame.Vector2(-24, 0), pygame.Vector2(0, 20), pygame.Vector2(0, -20), pygame.Vector2(18, 16), pygame.Vector2(-18, 16), pygame.Vector2(18, -16), pygame.Vector2(-18, -16)):
            if not game.is_road(point + offset):
                return False
        return True

    def cell(self, point: pygame.Vector2) -> tuple[int, int]:
        return round(point.x / self.cell_size), round(point.y / self.cell_size)

    def point(self, cell: tuple[int, int]) -> pygame.Vector2:
        return pygame.Vector2(cell[0] * self.cell_size, cell[1] * self.cell_size)

    def plan(self, game: "RacingGame", start: pygame.Vector2, goal: pygame.Vector2) -> list[pygame.Vector2]:
        start_cell, goal_cell = self.cell(start), self.cell(goal)
        # If the exact gate is close to a wall, target a safe cell around it.
        if not self.safe(game, self.point(goal_cell)):
            candidates = [(self.point((goal_cell[0] + dx, goal_cell[1] + dy)), (dx, dy)) for dx in range(-3, 4) for dy in range(-3, 4)]
            safe_candidates = [item for item in candidates if self.safe(game, item[0])]
            if safe_candidates:
                goal_cell = self.cell(min(safe_candidates, key=lambda item: item[0].distance_to(goal))[0])
        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start_cell)]
        previous: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
        cost = {start_cell: 0.0}
        neighbours = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_cell:
                path = []
                while current is not None:
                    path.append(self.point(current))
                    current = previous[current]
                return list(reversed(path))
            for dx, dy in neighbours:
                candidate = current[0] + dx, current[1] + dy
                location = self.point(candidate)
                if not self.safe(game, location):
                    continue
                next_cost = cost[current] + math.hypot(dx, dy)
                if next_cost < cost.get(candidate, float("inf")):
                    cost[candidate], previous[candidate] = next_cost, current
                    heuristic = math.hypot(candidate[0] - goal_cell[0], candidate[1] - goal_cell[1])
                    heapq.heappush(frontier, (next_cost + heuristic, candidate))
        return []

    def act(self, game: "RacingGame", lidar: np.ndarray) -> tuple[float, float, bool]:
        gate = game.track.finish if game.awaiting_finish else game.track.checkpoints[game.next_checkpoint]
        goal = self.cell(gate)
        if self.route_goal != goal or not self.route or game.car.position.distance_to(self.route[min(1, len(self.route) - 1)]) > 80:
            self.route, self.route_goal = self.plan(game, game.car.position, gate), goal
        target = gate
        while len(self.route) > 1 and game.car.position.distance_to(self.route[0]) < 25:
            self.route.pop(0)
        if self.route:
            target = self.route[min(1, len(self.route) - 1)]
        vector = target - game.car.position
        desired = math.degrees(math.atan2(vector.y, vector.x))
        delta = (desired - game.car.angle + 180) % 360 - 180
        steering = max(-1.0, min(1.0, delta / 32.0))
        forward_clearance = float(np.min(lidar[len(lidar)//2 - 1:len(lidar)//2 + 2]))
        sharp_turn = abs(delta) > 32 or forward_clearance < 0.42
        throttle = 0.04 if sharp_turn else 0.22
        handbrake = bool(abs(delta) > 55 and game.car.speed > 55)
        return steering, throttle, handbrake

    def should_label(self, lidar: np.ndarray, learner_action: tuple[float, float, bool]) -> bool:
        steering, throttle, handbrake = learner_action
        return self.label_all or float(np.min(lidar)) < 0.38 or abs(steering) > 0.55 or throttle > 0.65 or handbrake
