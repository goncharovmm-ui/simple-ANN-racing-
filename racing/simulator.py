from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pygame

from racing.neural import NeuralPolicy
from racing.expert import ArtificialExpert

WIDTH, HEIGHT = 1280, 800
LIDAR_ANGLES = np.linspace(-110.0, 110.0, 15, dtype=np.float32)
MAX_LIDAR_DISTANCE = 260.0
LAP_BONUS = 10_000
CHECKPOINT_BONUS = 2_500
CHECKPOINT_RADIUS = 64.0
FORWARD_PROGRESS_BONUS = 1.5
REVERSE_PROGRESS_PENALTY = 2.0


@dataclass
class Track:
    road: pygame.Rect = field(default_factory=lambda: pygame.Rect(80, 80, 1120, 640))
    inner_grass: pygame.Rect = field(default_factory=lambda: pygame.Rect(330, 255, 620, 290))
    start: pygame.Vector2 = field(default_factory=lambda: pygame.Vector2(245, 170))
    start_angle: float = 0.0
    finish: pygame.Vector2 = field(default_factory=lambda: pygame.Vector2(170, 215))
    checkpoints: list[pygame.Vector2] = field(default_factory=lambda: [
        pygame.Vector2(570, 170), pygame.Vector2(1030, 170), pygame.Vector2(1090, 370),
        pygame.Vector2(1090, 625), pygame.Vector2(650, 630), pygame.Vector2(190, 630), pygame.Vector2(170, 390)])
    obstacles: list[pygame.Rect] = field(default_factory=lambda: [
        pygame.Rect(610, 120, 48, 105), pygame.Rect(1000, 500, 70, 48), pygame.Rect(215, 530, 70, 48)])
    name: str = "Untitled track"

    @classmethod
    def load(cls, path: Path | None) -> "Track":
        if path is None or not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(pygame.Rect(raw["road"]), pygame.Rect(raw["inner_grass"]), pygame.Vector2(raw["start"][:2]),
                   float(raw["start"][2]), pygame.Vector2(raw.get("finish", [170, 215])), [pygame.Vector2(p) for p in raw["checkpoints"]],
                   [pygame.Rect(r) for r in raw["obstacles"]], raw.get("name", path.stem))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = {"name": self.name, "road": list(self.road), "inner_grass": list(self.inner_grass),
               "start": [round(self.start.x), round(self.start.y), self.start_angle], "finish": [round(self.finish.x), round(self.finish.y)],
               "checkpoints": [[round(p.x), round(p.y)] for p in self.checkpoints],
               "obstacles": [list(rect) for rect in self.obstacles]}
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def fingerprint(self) -> str:
        """Stable content ID, so an edited map never shares a dataset bucket with its old version."""
        raw = {"road": list(self.road), "inner_grass": list(self.inner_grass),
               "start": [self.start.x, self.start.y, self.start_angle], "finish": [self.finish.x, self.finish.y],
               "checkpoints": [[p.x, p.y] for p in self.checkpoints], "obstacles": [list(rect) for rect in self.obstacles]}
        return hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    def course_points(self) -> list[pygame.Vector2]:
        """Ordered clockwise route used for directional progress scoring."""
        return [self.start.copy(), *(point.copy() for point in self.checkpoints), self.finish.copy()]

    def course_length(self) -> float:
        points = self.course_points()
        return sum(points[index].distance_to(points[(index + 1) % len(points)]) for index in range(len(points)))

    def course_position(self, point: pygame.Vector2) -> float:
        """Project a world position onto the ordered closed route in distance units."""
        points = self.course_points()
        best_distance, best_position, travelled = float("inf"), 0.0, 0.0
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            segment = end - start
            length_squared = segment.length_squared()
            if length_squared == 0:
                travelled += start.distance_to(end)
                continue
            factor = max(0.0, min(1.0, (point - start).dot(segment) / length_squared))
            projection = start + segment * factor
            distance = point.distance_to(projection)
            length = segment.length()
            if distance < best_distance:
                best_distance = distance
                best_position = travelled + factor * length
            travelled += length
        return best_position

    def random_start(self) -> tuple[pygame.Vector2, float]:
        for _ in range(80):
            candidate = self.start + pygame.Vector2(random.uniform(-48, 48), random.uniform(-35, 35))
            valid = self.road.collidepoint(candidate) and not self.inner_grass.collidepoint(candidate) and not any(rect.collidepoint(candidate) for rect in self.obstacles)
            if valid:
                target = self.checkpoints[0] if self.checkpoints else self.finish
                angle = math.degrees(math.atan2(target.y - candidate.y, target.x - candidate.x)) + random.uniform(-12, 12)
                return candidate, angle
        return self.start.copy(), self.start_angle

    def random_route_start(self) -> tuple[pygame.Vector2, float, int]:
        """Spawn on a random course segment, facing the next checkpoint.

        This gives DAgger recovery examples across the entire circuit instead of
        repeatedly teaching only the first corner after the start line.
        """
        if not self.checkpoints:
            position, angle = self.random_start()
            return position, angle, 0
        target_index = random.randrange(len(self.checkpoints))
        previous = self.start if target_index == 0 else self.checkpoints[target_index - 1]
        target = self.checkpoints[target_index]
        direction = target - previous
        if direction.length_squared() == 0:
            position, angle = self.random_start()
            return position, angle, 0
        direction = direction.normalize()
        for _ in range(80):
            distance = random.uniform(0.18, 0.72) * previous.distance_to(target)
            candidate = previous + direction * distance + pygame.Vector2(random.uniform(-20, 20), random.uniform(-20, 20))
            valid = self.road.collidepoint(candidate) and not self.inner_grass.collidepoint(candidate) and not any(rect.collidepoint(candidate) for rect in self.obstacles)
            if valid:
                angle = math.degrees(math.atan2(target.y - candidate.y, target.x - candidate.x)) + random.uniform(-16, 16)
                return candidate, angle, target_index
        position, angle = self.random_start()
        return position, angle, 0


@dataclass
class Car:
    position: pygame.Vector2
    angle: float
    velocity: pygame.Vector2 = field(default_factory=pygame.Vector2)
    last_position: pygame.Vector2 = field(default_factory=pygame.Vector2, init=False)

    @property
    def speed(self) -> float:
        return self.velocity.length()

    def heading(self) -> pygame.Vector2:
        radians = math.radians(self.angle)
        return pygame.Vector2(math.cos(radians), math.sin(radians))

    def update(self, steering: float, throttle: float, handbrake: bool, dt: float) -> None:
        """Arcade model: handbrake preserves lateral momentum to make a drift."""
        steering, throttle = max(-1.0, min(1.0, steering)), max(-1.0, min(1.0, throttle))
        forward = self.heading()
        side = pygame.Vector2(-forward.y, forward.x)
        forward_speed, lateral_speed = self.velocity.dot(forward), self.velocity.dot(side)
        forward_speed = max(-90.0, min(360.0, (forward_speed + throttle * 300.0 * dt) * 0.985 ** (dt * 60.0)))
        self.angle += steering * 145.0 * min(1.0, abs(forward_speed) / 75.0) * dt * (1 if forward_speed >= 0 else -1)
        lateral_speed *= math.exp(-(0.42 if handbrake else 7.5) * dt)
        forward = self.heading()
        self.velocity = forward * forward_speed + pygame.Vector2(-forward.y, forward.x) * lateral_speed
        self.last_position = self.position.copy()
        self.position += self.velocity * dt

    def hit_wall(self) -> None:
        self.position = self.last_position.copy()
        self.velocity *= -0.18


class DemoWriter:
    fields = ["run_id", "mode", "driver", "track_id", "map_path", "episode", "step", "x", "y", "angle", "speed", "next_checkpoint", "progress", "course_progress", "directional_score", "checkpoint_points", "checkpoint_accuracy", "collisions", "elapsed_s", "lap_s", "fitness", "steering", "throttle", "handbrake", *[f"lidar_{i}" for i in range(len(LIDAR_ANGLES))]]

    def __init__(self, directory: Path, metadata: dict[str, str], require_finish: bool = True) -> None:
        self.require_finish = require_finish
        self.metadata = metadata
        filename = f"{metadata['driver']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{metadata['run_id'][:8]}.csv"
        self.final_path = directory / filename
        if require_finish:
            pending = Path("data/demos/_pending")
            pending.mkdir(parents=True, exist_ok=True)
            self.path, self.committed = pending / filename, False
        else:
            directory.mkdir(parents=True, exist_ok=True)
            self.path, self.committed = self.final_path, True
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.writer.writeheader()

    def write(self, game: "RacingGame", steering: float, throttle: float, handbrake: bool, lidar: np.ndarray) -> None:
        car = game.car
        row = {**self.metadata, "episode": game.episode, "step": game.step, "x": round(car.position.x, 3), "y": round(car.position.y, 3),
               "angle": round(car.angle, 3), "speed": round(car.speed, 3), "next_checkpoint": game.next_checkpoint,
               "progress": game.progress, "course_progress": round(game.course_progress, 5), "directional_score": round(game.directional_score, 3), "checkpoint_points": round(game.checkpoint_points, 2), "checkpoint_accuracy": round(game.last_checkpoint_accuracy, 4), "collisions": game.collisions, "elapsed_s": round(game.elapsed, 3), "lap_s": round(game.lap_elapsed, 3),
               "fitness": round(game.fitness, 3), "steering": round(steering, 3), "throttle": round(throttle, 3), "handbrake": int(handbrake)}
        row.update({f"lidar_{i}": round(float(v), 4) for i, v in enumerate(lidar)})
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def commit(self) -> None:
        if self.committed:
            return
        self.file.close()
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.path.replace(self.final_path)
        self.path = self.final_path
        self.file = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.committed = True

    def discard(self) -> None:
        self.file.close()
        if self.require_finish:
            self.path.unlink(missing_ok=True)


def draw_track(screen: pygame.Surface, track: Track, next_checkpoint: int | None = None, finish_active: bool = False) -> None:
    screen.fill((36, 112, 57))
    pygame.draw.rect(screen, (61, 65, 72), track.road, border_radius=28)
    pygame.draw.rect(screen, (36, 112, 57), track.inner_grass, border_radius=20)
    pygame.draw.rect(screen, (239, 110, 64), track.road, width=4, border_radius=28)
    pygame.draw.rect(screen, (239, 110, 64), track.inner_grass, width=4, border_radius=20)
    for obstacle in track.obstacles:
        pygame.draw.rect(screen, (193, 58, 55), obstacle, border_radius=5)
    zones = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    for i, checkpoint in enumerate(track.checkpoints):
        selected = i == next_checkpoint
        pygame.draw.circle(zones, (246, 208, 74, 105) if selected else (175, 185, 195, 48), checkpoint, round(CHECKPOINT_RADIUS), 2)
        pygame.draw.circle(screen, (246, 208, 74) if selected else (170, 174, 178), checkpoint, 9)
    screen.blit(zones, (0, 0))
    finish_color = (81, 231, 126) if finish_active else (118, 168, 132)
    pygame.draw.circle(screen, finish_color, track.finish, 15, width=4)
    pygame.draw.line(screen, finish_color, track.finish + pygame.Vector2(-11, 0), track.finish + pygame.Vector2(11, 0), 3)


def speed_color(speed: float, maximum: float = 360.0) -> tuple[int, int, int]:
    """Blue = slow, yellow = medium, red = fast."""
    ratio = max(0.0, min(1.0, speed / maximum))
    return round(35 + 220 * ratio), round(205 - 125 * ratio), round(255 - 220 * ratio)


def draw_speed_trajectory(screen: pygame.Surface, samples: list[tuple[pygame.Vector2, float]], width: int = 3) -> None:
    for (start, start_speed), (end, end_speed) in zip(samples, samples[1:]):
        pygame.draw.line(screen, speed_color((start_speed + end_speed) * 0.5), start, end, width)


class RacingGame:
    def __init__(self, track: Track, record: bool = False, screen: pygame.Surface | None = None, policy: NeuralPolicy | None = None, game_mode: str = "race", map_path: Path = Path("maps/default.json"), expert: ArtificialExpert | None = None, max_episodes: int | None = None, expert_controls: bool = False, route_random_start: bool = False, require_full_lap: bool = False) -> None:
        pygame.init()
        self.screen = screen if screen is not None else pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock, self.font = pygame.time.Clock(), pygame.font.Font(None, 28)
        pygame.display.set_caption("Simple ANN Racing — manual data collection")
        self.track, self.recording, self.policy, self.expert, self.expert_controls = track, record, policy, expert, expert_controls
        self.route_random_start = route_random_start
        self.require_full_lap = require_full_lap
        self.game_mode, self.map_path = game_mode, map_path
        self.track_id = f"{map_path.stem}-{track.fingerprint()}"
        self.run_id = uuid4().hex
        self.writer: DemoWriter | None = None
        self.collisions = 0
        self.game_over = False
        self.idle_seconds = 0.0
        self.dagger_label: tuple[float, float, bool] | None = None
        self.max_episodes, self.finished_episodes, self.successful_episodes = max_episodes, 0, 0
        self.trajectory: list[pygame.Vector2] = []
        self.trajectory_speeds: list[float] = []
        self.trajectory_history: list[tuple[list[pygame.Vector2], tuple[int, int, int]]] = []
        self.reference_trajectory = self.load_best_reference_trajectory()
        if record:
            self.start_recording()
        self.best_lap: float | None = None
        self.reset("Ready")
        self.episode = 1

    def start_recording(self) -> None:
        if self.policy and not self.expert:
            self.recording = False
            self.message = "AI actions are not recorded as demonstrations"
            return
        driver = "artificial_expert" if self.expert else "player"
        metadata = {"run_id": self.run_id, "mode": self.game_mode, "driver": driver, "track_id": self.track_id, "map_path": str(self.map_path)}
        self.writer = DemoWriter(Path("data/demos") / self.game_mode / self.track_id, metadata, require_finish=not bool(self.expert))
        self.recording = True

    def load_best_reference_trajectory(self) -> list[tuple[pygame.Vector2, float]]:
        """Load the fastest completed player path for this exact map revision."""
        candidates: list[tuple[float, list[tuple[pygame.Vector2, float]]]] = []
        directory = Path("data/demos/race")
        if not directory.exists():
            return []
        for path in directory.rglob("*.csv"):
            try:
                with path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if not rows or rows[0].get("track_id") != self.track_id or rows[0].get("driver") != "player":
                    continue
                samples = [(pygame.Vector2(float(row["x"]), float(row["y"])), float(row.get("speed") or 0.0)) for row in rows]
                lap = max(float(row.get("lap_s") or 0.0) for row in rows)
                if samples and lap > 0:
                    candidates.append((lap, samples))
            except (OSError, ValueError, KeyError):
                continue
        return min(candidates, key=lambda item: item[0])[1] if candidates else []

    @property
    def fitness(self) -> float:
        """Reward clockwise progress and penalize reverse movement explicitly."""
        return self.completed_laps * LAP_BONUS + self.checkpoint_points + self.directional_score - self.elapsed * 10 - self.collisions * 250

    def is_road(self, point: pygame.Vector2) -> bool:
        return self.track.road.collidepoint(point) and not self.track.inner_grass.collidepoint(point) and not any(r.collidepoint(point) for r in self.track.obstacles)

    def car_points(self) -> list[pygame.Vector2]:
        forward = self.car.heading()
        side = pygame.Vector2(-forward.y, forward.x)
        return [self.car.position + forward * 16 + side * 10, self.car.position + forward * 16 - side * 10, self.car.position - forward * 16 - side * 10, self.car.position - forward * 16 + side * 10]

    def raycast(self) -> np.ndarray:
        values = []
        for angle in LIDAR_ANGLES:
            radians = math.radians(self.car.angle + float(angle))
            direction, distance = pygame.Vector2(math.cos(radians), math.sin(radians)), 0.0
            while distance < MAX_LIDAR_DISTANCE and self.is_road(self.car.position + direction * distance):
                distance += 4.0
            values.append(min(distance, MAX_LIDAR_DISTANCE) / MAX_LIDAR_DISTANCE)
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def crossed_zone(point: pygame.Vector2, start: pygame.Vector2, end: pygame.Vector2, radius: float = 64.0) -> bool:
        """Detect a gate crossed during this frame, not only the final position after a wall bounce."""
        segment = end - start
        length_squared = segment.length_squared()
        if length_squared == 0:
            return point.distance_to(end) <= radius
        factor = max(0.0, min(1.0, (point - start).dot(segment) / length_squared))
        return point.distance_to(start + segment * factor) <= radius

    @staticmethod
    def segment_distance(point: pygame.Vector2, start: pygame.Vector2, end: pygame.Vector2) -> float:
        segment = end - start
        length_squared = segment.length_squared()
        if length_squared == 0:
            return point.distance_to(end)
        factor = max(0.0, min(1.0, (point - start).dot(segment) / length_squared))
        return point.distance_to(start + segment * factor)

    def reset(self, reason: str) -> None:
        if self.writer:
            if self.writer.committed:
                self.writer.close()
            else:
                self.writer.discard()
            self.writer = None
            self.run_id = uuid4().hex
            self.start_recording()
        if self.route_random_start:
            position, angle, initial_checkpoint = self.track.random_route_start()
        else:
            position, angle, initial_checkpoint = *self.track.random_start(), 0
        self.car = Car(position, angle)
        self.next_checkpoint, self.progress, self.completed_laps, self.step = initial_checkpoint, 0, 0, 0
        self.visited_checkpoints: set[int] = set()
        self.checkpoint_points = 0.0
        self.last_checkpoint_accuracy = 0.0
        self.course_distance = self.track.course_position(self.car.position)
        self.course_progress = self.course_distance / max(1.0, self.track.course_length())
        self.directional_score = 0.0
        self.last_lap_progress = 0
        self.awaiting_finish = False
        self.elapsed = self.lap_elapsed = 0.0
        self.collisions = 0
        self.game_over = False
        self.idle_seconds = 0.0
        self.trajectory = []
        self.trajectory_speeds = []
        self.last_lap: float | None = None
        self.message = reason

    def update_directional_score(self, movement_start: pygame.Vector2, movement_end: pygame.Vector2) -> None:
        """Reward movement along the ordered route and penalize reverse movement."""
        total_length = max(1.0, self.track.course_length())
        next_distance = self.track.course_position(movement_end)
        delta = next_distance - self.course_distance
        # Treat crossing the finish/start seam as a small forward step.
        if delta > total_length * 0.5:
            delta -= total_length
        elif delta < -total_length * 0.5:
            delta += total_length
        movement_length = movement_start.distance_to(movement_end)
        if movement_length == 0.0:
            delta = 0.0
        else:
            # A nearest-segment projection can jump at intersections; never let
            # that create a score spike larger than the actual movement.
            delta = max(-movement_length * 1.5, min(movement_length * 1.5, delta))
        self.course_distance = next_distance
        self.course_progress = next_distance / total_length
        if delta >= 0:
            self.directional_score += delta * FORWARD_PROGRESS_BONUS
        else:
            self.directional_score += delta * REVERSE_PROGRESS_PENALTY

    def update(self, dt: float) -> tuple[float, float, bool, np.ndarray]:
        self.dagger_label = None
        if self.game_over:
            return 0.0, 0.0, False, self.raycast()
        keys = pygame.key.get_pressed()
        if self.policy:
            state_lidar = self.raycast()
            steering, throttle, handbrake = self.policy.act(state_lidar, self.car.speed)
            if self.expert and self.expert.should_label(state_lidar, (steering, throttle, handbrake)):
                self.dagger_label = self.expert.act(self, state_lidar)
                if self.expert_controls:
                    steering, throttle, handbrake = self.dagger_label
        else:
            steering = float(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - float(keys[pygame.K_LEFT] or keys[pygame.K_a])
            throttle = float(keys[pygame.K_UP] or keys[pygame.K_w]) - float(keys[pygame.K_DOWN] or keys[pygame.K_s])
            handbrake = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
        self.car.update(steering, throttle, handbrake, dt)
        movement_start, movement_end = self.car.last_position.copy(), self.car.position.copy()
        self.trajectory.append(self.car.position.copy())
        self.trajectory_speeds.append(self.car.speed)
        self.elapsed += dt
        self.lap_elapsed += dt
        if self.car.speed < 12.0:
            self.idle_seconds += dt
        else:
            self.idle_seconds = 0.0
        if self.idle_seconds >= 5.0:
            self.game_over = True
            self.message = "Game over: inactive for 5 seconds — press R/К for a new attempt"
        if not all(self.is_road(point) for point in self.car_points()):
            self.car.hit_wall()
            self.collisions += 1
            if self.collisions >= 5:
                self.game_over = True
                self.message = "Game over: 5 wall hits — press R/К for a new attempt"
            else:
                self.message = f"Wall hit: speed reduced ({self.collisions}/5)"
        self.update_directional_score(movement_start, self.car.position.copy())
        if self.track.checkpoints and not self.awaiting_finish:
            for index, checkpoint in enumerate(self.track.checkpoints):
                if index in self.visited_checkpoints or not self.crossed_zone(checkpoint, movement_start, movement_end, CHECKPOINT_RADIUS):
                    continue
                distance = self.segment_distance(checkpoint, movement_start, movement_end)
                self.last_checkpoint_accuracy = max(0.0, min(1.0, 1.0 - distance / CHECKPOINT_RADIUS))
                checkpoint_bonus = CHECKPOINT_BONUS * (0.25 + 0.75 * self.last_checkpoint_accuracy)
                self.checkpoint_points += checkpoint_bonus
                self.visited_checkpoints.add(index)
                self.progress = len(self.visited_checkpoints)
                self.message = f"Checkpoint {self.progress}/{len(self.track.checkpoints)} — accuracy {self.last_checkpoint_accuracy:.0%}, +{checkpoint_bonus:.0f}"
            remaining = [index for index in range(len(self.track.checkpoints)) if index not in self.visited_checkpoints]
            if remaining:
                self.next_checkpoint = min(remaining, key=lambda index: self.car.position.distance_to(self.track.checkpoints[index]))
            else:
                self.awaiting_finish = True
                self.next_checkpoint = len(self.track.checkpoints)
                self.message = "All checkpoints passed — head to the finish"
        # A finish may be crossed at any time.  Checkpoints are a large fitness
        # bonus rather than a hard gate, which lets the agent discover shortcuts.
        if self.crossed_zone(self.track.finish, movement_start, movement_end) and (not self.require_full_lap or len(self.visited_checkpoints) == len(self.track.checkpoints)):
            self.completed_laps += 1
            self.last_lap = self.lap_elapsed
            self.last_lap_progress = self.progress
            self.best_lap = self.last_lap if self.best_lap is None else min(self.best_lap, self.last_lap)
            self.lap_elapsed, self.next_checkpoint, self.awaiting_finish = 0.0, 0, False
            self.visited_checkpoints.clear()
            self.progress = 0
            if self.writer and not self.writer.committed:
                self.writer.commit()
                self.message = f"Lap {self.completed_laps}: {self.last_lap:.2f}s, checkpoints {self.progress} — run saved"
            else:
                self.message = f"Lap {self.completed_laps}: {self.last_lap:.2f}s, checkpoints {self.progress}"
        return steering, throttle, handbrake, self.raycast()

    def draw(self, lidar: np.ndarray) -> None:
        draw_track(self.screen, self.track, None if self.awaiting_finish else self.next_checkpoint, True)
        for points, color in self.trajectory_history:
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 3)
        if not self.policy and self.reference_trajectory:
            draw_speed_trajectory(self.screen, self.reference_trajectory, 4)
        if len(self.trajectory_speeds) > 1:
            draw_speed_trajectory(self.screen, list(zip(self.trajectory, self.trajectory_speeds)), 3)
        for angle, distance in zip(LIDAR_ANGLES, lidar):
            radians = math.radians(self.car.angle + float(angle))
            endpoint = self.car.position + pygame.Vector2(math.cos(radians), math.sin(radians)) * distance * MAX_LIDAR_DISTANCE
            pygame.draw.line(self.screen, (99, 197, 238), self.car.position, endpoint, 1)
        pygame.draw.polygon(self.screen, (76, 172, 247), self.car_points())
        pygame.draw.circle(self.screen, (250, 250, 250), self.car.position + self.car.heading() * 10, 3)
        best = "--" if self.best_lap is None else f"{self.best_lap:.2f}s"
        lines = [f"Lap {self.completed_laps + 1}: {self.lap_elapsed:.2f}s | best: {best} | score: {self.fitness:.0f}",
                 f"checkpoints {self.progress} | course {self.course_progress:.0%} | direction {self.directional_score:+.0f} | speed {self.car.speed:5.1f} | hits {self.collisions} | mode: {'AI' if self.policy else 'manual'}",
                 "WASD/arrows — drive   Shift — drift   R/К — restart   Esc — quit", self.message]
        if not self.policy and self.reference_trajectory:
            lines.insert(2, "reference/current trajectory: blue = slow, red = fast")
        if self.expert and self.max_episodes:
            lines.insert(2, f"DAgger batch: {self.finished_episodes}/{self.max_episodes} | finished: {self.successful_episodes} | green = better trajectory")
        for i, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, (250, 250, 250) if i < 3 else (255, 226, 102)), (24, 20 + i * 30))
        pygame.display.flip()

    def finish_dagger_episode(self) -> None:
        success = self.completed_laps > 0
        if success:
            self.successful_episodes += 1
        if success:
            time_quality = max(0.0, min(1.0, 1.0 - (self.last_lap or 30.0) / 30.0))
            damage_quality = 1.0 - min(1.0, self.collisions / 5.0)
            quality = 0.55 + 0.45 * (0.7 * time_quality + 0.3 * damage_quality)
        else:
            quality = max(0.0, 0.35 - self.collisions * 0.05)
        color = (round(225 * (1 - quality) + 35 * quality), round(60 * (1 - quality) + 200 * quality), 65)
        self.trajectory_history.append((self.trajectory.copy(), color))
        self.trajectory_history = self.trajectory_history[-20:]
        self.finished_episodes += 1

    def run(self) -> None:
        running = True
        try:
            while running:
                dt = min(self.clock.tick(60) / 1000.0, 0.05)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE: running = False
                        elif event.key == pygame.K_r or event.unicode.lower() == "к":
                            self.reset("Restarted"); self.episode += 1
                steering, throttle, handbrake, lidar = self.update(dt)
                if self.recording and self.writer:
                    if self.dagger_label:
                        self.writer.write(self, *self.dagger_label, lidar)
                    elif not self.expert:
                        self.writer.write(self, steering, throttle, handbrake, lidar)
                self.step += 1
                if self.expert and self.max_episodes and (self.game_over or self.completed_laps > 0):
                    self.finish_dagger_episode()
                    if self.finished_episodes >= self.max_episodes:
                        self.message = f"DAgger batch complete: {self.successful_episodes}/{self.max_episodes} finished"
                        self.draw(lidar)
                        running = False
                        continue
                    self.reset(f"DAgger attempt {self.finished_episodes + 1}/{self.max_episodes}")
                    self.episode += 1
                self.draw(lidar)
        finally:
            if self.writer:
                if self.writer.committed:
                    self.writer.close(); print(f"Saved demonstrations to {self.writer.path}")
                else:
                    self.writer.discard(); print("Discarded incomplete demonstration run")
            pygame.quit()


class LegacyMapEditor:
    def __init__(self, track: Track, save_path: Path) -> None:
        pygame.init()
        self.screen, self.clock, self.font = pygame.display.set_mode((WIDTH, HEIGHT)), pygame.time.Clock(), pygame.font.Font(None, 25)
        pygame.display.set_caption("Simple ANN Racing — track sandbox")
        self.track, self.save_path, self.drag_start = track, save_path, None
        self.testing = False
        self.preview_game: RacingGame | None = None
        self.message = "Left-drag: obstacle. Tab: test track. F5: save."

    def draw(self) -> None:
        draw_track(self.screen, self.track)
        pygame.draw.circle(self.screen, (90, 188, 255), self.track.start, 12)
        direction = pygame.Vector2(math.cos(math.radians(self.track.start_angle)), math.sin(math.radians(self.track.start_angle)))
        pygame.draw.line(self.screen, (250, 250, 250), self.track.start, self.track.start + direction * 25, 3)
        if self.drag_start:
            mouse = pygame.Vector2(pygame.mouse.get_pos())
            pygame.draw.rect(self.screen, (255, 234, 114), pygame.Rect(min(self.drag_start.x, mouse.x), min(self.drag_start.y, mouse.y), abs(mouse.x-self.drag_start.x), abs(mouse.y-self.drag_start.y)), 2)
        lines = ["TRACK SANDBOX | left-drag: obstacle | right-click: delete obstacle", "C: checkpoint | X: remove last checkpoint | S: start | Tab: test | F5: save | Esc: quit", f"file: {self.save_path} | checkpoints: {len(self.track.checkpoints)} | obstacles: {len(self.track.obstacles)}", self.message]
        for i, line in enumerate(lines): self.screen.blit(self.font.render(line, True, (250, 250, 250) if i < 3 else (255, 226, 102)), (20, 18+i*29))
        pygame.display.flip()

    def run(self) -> None:
        running = True
        try:
            while running:
                dt = min(self.clock.tick(60) / 1000.0, 0.05)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                        self.testing = not self.testing
                        if self.testing:
                            self.preview_game = RacingGame(self.track, record=False, screen=self.screen, game_mode="sandbox_test", map_path=self.save_path)
                            self.message = "Test mode: drive with WASD/arrows, Shift to drift, Tab to edit"
                        else:
                            self.preview_game = None
                            self.message = "Edit mode"
                    elif self.testing:
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            self.testing = False
                            self.preview_game = None
                            self.message = "Edit mode"
                        elif event.type == pygame.KEYDOWN and (event.key == pygame.K_r or event.unicode.lower() == "к"):
                            assert self.preview_game is not None
                            self.preview_game.reset("Restarted")
                            self.preview_game.episode += 1
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: self.drag_start = pygame.Vector2(event.pos)
                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.drag_start:
                        end = pygame.Vector2(event.pos)
                        rect = pygame.Rect(min(self.drag_start.x,end.x), min(self.drag_start.y,end.y), abs(end.x-self.drag_start.x), abs(end.y-self.drag_start.y))
                        if rect.width >= 12 and rect.height >= 12: self.track.obstacles.append(rect); self.message = "Obstacle added"
                        self.drag_start = None
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                        before = len(self.track.obstacles); self.track.obstacles = [r for r in self.track.obstacles if not r.collidepoint(event.pos)]
                        self.message = "Obstacle deleted" if len(self.track.obstacles) != before else "No obstacle here"
                    elif event.type == pygame.KEYDOWN:
                        mouse = pygame.Vector2(pygame.mouse.get_pos())
                        if event.key == pygame.K_ESCAPE: running = False
                        elif event.key == pygame.K_F5: self.track.save(self.save_path); self.message = f"Saved: {self.save_path}"
                        elif event.key == pygame.K_c or event.unicode.lower() == "с": self.track.checkpoints.append(mouse); self.message = "Checkpoint added"
                        elif event.key == pygame.K_x or event.unicode.lower() == "ч":
                            if self.track.checkpoints: self.track.checkpoints.pop(); self.message = "Last checkpoint removed"
                        elif event.key == pygame.K_s or event.unicode.lower() == "ы": self.track.start = mouse; self.message = "Start moved"
                if self.testing:
                    assert self.preview_game is not None
                    steering, throttle, handbrake, lidar = self.preview_game.update(dt)
                    self.preview_game.step += 1
                    self.preview_game.draw(lidar)
                else:
                    self.draw()
        finally:
            pygame.quit()


class MapEditor:
    """Button-driven sandbox: mouse moves objects, while tools create/delete them."""

    def __init__(self, track: Track, save_path: Path) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock, self.font = pygame.time.Clock(), pygame.font.Font(None, 23)
        pygame.display.set_caption("Simple ANN Racing — track sandbox")
        self.track, self.save_path = track, save_path
        self.testing = False
        self.preview_game: RacingGame | None = None
        self.tool = "move"
        self.dragging: tuple[str, int | None] | None = None
        self.message = "Select a tool, then click the map. Drag objects in Move mode."
        names = [("move", "Move"), ("obstacle", "Add obstacle"), ("checkpoint", "Add checkpoint"),
                 ("start", "Set start"), ("finish", "Set finish"), ("delete", "Delete"), ("save", "Save map"), ("test", "Test track")]
        self.buttons = [(action, label, pygame.Rect(10 + index * 158, 14, 150, 42)) for index, (action, label) in enumerate(names)]

    def draw_toolbar(self) -> None:
        for action, label, rect in self.buttons:
            selected = action == self.tool and action not in ("save", "test")
            color = (58, 117, 168) if selected else (54, 61, 72)
            pygame.draw.rect(self.screen, color, rect, border_radius=7)
            pygame.draw.rect(self.screen, (245, 205, 80) if selected else (184, 190, 198), rect, 2, border_radius=7)
            text = self.font.render(label, True, (255, 255, 255))
            self.screen.blit(text, text.get_rect(center=rect.center))

    def draw(self) -> None:
        draw_track(self.screen, self.track)
        pygame.draw.circle(self.screen, (90, 188, 255), self.track.start, 12)
        direction = pygame.Vector2(math.cos(math.radians(self.track.start_angle)), math.sin(math.radians(self.track.start_angle)))
        pygame.draw.line(self.screen, (250, 250, 250), self.track.start, self.track.start + direction * 25, 3)
        self.draw_toolbar()
        details = f"{self.tool.upper()} | checkpoints: {len(self.track.checkpoints)} | obstacles: {len(self.track.obstacles)} | {self.message}"
        self.screen.blit(self.font.render(details, True, (255, 230, 115)), (14, 64))
        pygame.display.flip()

    def toggle_test(self) -> None:
        self.testing = True
        self.preview_game = RacingGame(self.track, record=False, screen=self.screen, game_mode="sandbox_test", map_path=self.save_path)
        self.message = "Test mode — Esc returns to sandbox"

    def object_at(self, position: tuple[int, int]) -> tuple[str, int | None] | None:
        for index in range(len(self.track.obstacles) - 1, -1, -1):
            if self.track.obstacles[index].collidepoint(position):
                return "obstacle", index
        for index in range(len(self.track.checkpoints) - 1, -1, -1):
            if self.track.checkpoints[index].distance_to(position) <= 15:
                return "checkpoint", index
        if self.track.start.distance_to(position) <= 16:
            return "start", None
        if self.track.finish.distance_to(position) <= 18:
            return "finish", None
        return None

    def handle_tool(self, position: tuple[int, int]) -> None:
        for action, _, rect in self.buttons:
            if rect.collidepoint(position):
                if action == "save":
                    self.track.save(self.save_path)
                    self.message = f"Saved: {self.save_path}"
                elif action == "test":
                    self.toggle_test()
                else:
                    self.tool = action
                    self.message = f"Selected: {action}"
                return
        if position[1] < 80:
            return
        point = pygame.Vector2(position)
        if self.tool == "obstacle":
            self.track.obstacles.append(pygame.Rect(round(point.x - 30), round(point.y - 20), 60, 40))
            self.message = "Obstacle added — use Move to position it"
        elif self.tool == "checkpoint":
            self.track.checkpoints.append(point)
            self.message = "Checkpoint added"
        elif self.tool == "start":
            self.track.start = point
            self.message = "Start moved"
        elif self.tool == "finish":
            self.track.finish = point
            self.message = "Finish moved"
        elif self.tool == "delete":
            found = self.object_at(position)
            if found is None:
                self.message = "Nothing to delete here"
            elif found[0] == "obstacle":
                self.track.obstacles.pop(found[1])
                self.message = "Obstacle deleted"
            elif found[0] == "checkpoint":
                self.track.checkpoints.pop(found[1])
                self.message = "Checkpoint deleted"
            else:
                self.message = "Start and finish cannot be deleted; use their buttons to move them"
        elif self.tool == "move":
            self.dragging = self.object_at(position)
            self.message = "Dragging object" if self.dragging else "Choose an object to move"

    def move_dragged(self, position: tuple[int, int]) -> None:
        if not self.dragging:
            return
        kind, index = self.dragging
        point = pygame.Vector2(position)
        if kind == "obstacle":
            self.track.obstacles[index].center = (round(point.x), round(point.y))
        elif kind == "checkpoint":
            self.track.checkpoints[index] = point
        elif kind == "start":
            self.track.start = point
        else:
            self.track.finish = point

    def run(self) -> None:
        running = True
        try:
            while running:
                dt = min(self.clock.tick(60) / 1000.0, 0.05)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif self.testing:
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            self.testing, self.preview_game = False, None
                            self.message = "Back in sandbox"
                        elif event.type == pygame.KEYDOWN and (event.key == pygame.K_r or event.unicode.lower() == "к"):
                            assert self.preview_game is not None
                            self.preview_game.reset("Restarted")
                            self.preview_game.episode += 1
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.handle_tool(event.pos)
                    elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                        self.move_dragged(event.pos)
                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        self.dragging = None
                if self.testing:
                    assert self.preview_game is not None
                    steering, throttle, handbrake, lidar = self.preview_game.update(dt)
                    self.preview_game.step += 1
                    self.preview_game.draw(lidar)
                else:
                    self.draw()
        finally:
            pygame.quit()
