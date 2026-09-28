from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pygame

WIDTH, HEIGHT = 1280, 800
LIDAR_ANGLES = np.linspace(-110.0, 110.0, 15, dtype=np.float32)
MAX_LIDAR_DISTANCE = 260.0


@dataclass
class Track:
    road: pygame.Rect = field(default_factory=lambda: pygame.Rect(80, 80, 1120, 640))
    inner_grass: pygame.Rect = field(default_factory=lambda: pygame.Rect(330, 255, 620, 290))
    start: pygame.Vector2 = field(default_factory=lambda: pygame.Vector2(245, 170))
    start_angle: float = 0.0
    checkpoints: list[pygame.Vector2] = field(default_factory=lambda: [
        pygame.Vector2(570, 170), pygame.Vector2(1030, 170), pygame.Vector2(1090, 370),
        pygame.Vector2(1090, 625), pygame.Vector2(650, 630), pygame.Vector2(190, 630), pygame.Vector2(170, 390)])
    obstacles: list[pygame.Rect] = field(default_factory=lambda: [
        pygame.Rect(610, 120, 48, 105), pygame.Rect(1000, 500, 70, 48), pygame.Rect(215, 530, 70, 48)])

    @classmethod
    def load(cls, path: Path | None) -> "Track":
        if path is None or not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(pygame.Rect(raw["road"]), pygame.Rect(raw["inner_grass"]), pygame.Vector2(raw["start"][:2]),
                   float(raw["start"][2]), [pygame.Vector2(p) for p in raw["checkpoints"]],
                   [pygame.Rect(r) for r in raw["obstacles"]])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = {"road": list(self.road), "inner_grass": list(self.inner_grass),
               "start": [round(self.start.x), round(self.start.y), self.start_angle],
               "checkpoints": [[round(p.x), round(p.y)] for p in self.checkpoints],
               "obstacles": [list(rect) for rect in self.obstacles]}
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class Car:
    position: pygame.Vector2
    angle: float
    velocity: pygame.Vector2 = field(default_factory=pygame.Vector2)
    crashed: bool = False

    @property
    def speed(self) -> float:
        return self.velocity.length()

    def heading(self) -> pygame.Vector2:
        radians = math.radians(self.angle)
        return pygame.Vector2(math.cos(radians), math.sin(radians))

    def update(self, steering: float, throttle: float, handbrake: bool, dt: float) -> None:
        """Arcade model: handbrake preserves lateral momentum to make a drift."""
        if self.crashed:
            return
        steering, throttle = max(-1.0, min(1.0, steering)), max(-1.0, min(1.0, throttle))
        forward = self.heading()
        side = pygame.Vector2(-forward.y, forward.x)
        forward_speed, lateral_speed = self.velocity.dot(forward), self.velocity.dot(side)
        forward_speed = max(-90.0, min(360.0, (forward_speed + throttle * 300.0 * dt) * 0.985 ** (dt * 60.0)))
        self.angle += steering * 145.0 * min(1.0, abs(forward_speed) / 75.0) * dt * (1 if forward_speed >= 0 else -1)
        lateral_speed *= math.exp(-(0.42 if handbrake else 7.5) * dt)
        forward = self.heading()
        self.velocity = forward * forward_speed + pygame.Vector2(-forward.y, forward.x) * lateral_speed
        self.position += self.velocity * dt


class DemoWriter:
    fields = ["episode", "step", "x", "y", "angle", "speed", "next_checkpoint", "progress", "elapsed_s", "lap_s", "fitness", "steering", "throttle", "handbrake", *[f"lidar_{i}" for i in range(len(LIDAR_ANGLES))]]

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"demo_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.writer.writeheader()

    def write(self, game: "RacingGame", steering: float, throttle: float, handbrake: bool, lidar: np.ndarray) -> None:
        car = game.car
        row = {"episode": game.episode, "step": game.step, "x": round(car.position.x, 3), "y": round(car.position.y, 3),
               "angle": round(car.angle, 3), "speed": round(car.speed, 3), "next_checkpoint": game.next_checkpoint,
               "progress": game.progress, "elapsed_s": round(game.elapsed, 3), "lap_s": round(game.lap_elapsed, 3),
               "fitness": round(game.fitness, 3), "steering": round(steering, 3), "throttle": round(throttle, 3), "handbrake": int(handbrake)}
        row.update({f"lidar_{i}": round(float(v), 4) for i, v in enumerate(lidar)})
        self.writer.writerow(row)

    def close(self) -> None:
        self.file.close()


def draw_track(screen: pygame.Surface, track: Track, next_checkpoint: int | None = None) -> None:
    screen.fill((36, 112, 57))
    pygame.draw.rect(screen, (61, 65, 72), track.road, border_radius=28)
    pygame.draw.rect(screen, (36, 112, 57), track.inner_grass, border_radius=20)
    pygame.draw.rect(screen, (239, 110, 64), track.road, width=4, border_radius=28)
    pygame.draw.rect(screen, (239, 110, 64), track.inner_grass, width=4, border_radius=20)
    for obstacle in track.obstacles:
        pygame.draw.rect(screen, (193, 58, 55), obstacle, border_radius=5)
    for i, checkpoint in enumerate(track.checkpoints):
        pygame.draw.circle(screen, (246, 208, 74) if i == next_checkpoint else (170, 174, 178), checkpoint, 9)


class RacingGame:
    def __init__(self, track: Track, record: bool = False, screen: pygame.Surface | None = None) -> None:
        pygame.init()
        self.screen = screen if screen is not None else pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock, self.font = pygame.time.Clock(), pygame.font.Font(None, 28)
        pygame.display.set_caption("Simple ANN Racing — manual data collection")
        self.track, self.recording = track, record
        self.writer = DemoWriter(Path("data/demos")) if record else None
        self.best_lap: float | None = None
        self.reset("Ready")
        self.episode = 1

    @property
    def fitness(self) -> float:
        return self.completed_laps * 10_000 + self.progress * 1_000 - self.elapsed * 10

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

    def reset(self, reason: str) -> None:
        self.car = Car(self.track.start.copy(), self.track.start_angle)
        self.next_checkpoint = self.progress = self.completed_laps = self.step = 0
        self.elapsed = self.lap_elapsed = 0.0
        self.last_lap: float | None = None
        self.message = reason

    def update(self, dt: float) -> tuple[float, float, bool, np.ndarray]:
        keys = pygame.key.get_pressed()
        steering = float(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - float(keys[pygame.K_LEFT] or keys[pygame.K_a])
        throttle = float(keys[pygame.K_UP] or keys[pygame.K_w]) - float(keys[pygame.K_DOWN] or keys[pygame.K_s])
        handbrake = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
        self.car.update(steering, throttle, handbrake, dt)
        self.elapsed += dt
        self.lap_elapsed += dt
        if not all(self.is_road(point) for point in self.car_points()):
            self.car.crashed, self.message = True, "Collision — press R/К to restart"
        if self.track.checkpoints and self.car.position.distance_to(self.track.checkpoints[self.next_checkpoint]) < 58:
            self.progress += 1
            self.next_checkpoint = (self.next_checkpoint + 1) % len(self.track.checkpoints)
            if self.next_checkpoint == 0:
                self.completed_laps += 1
                self.last_lap = self.lap_elapsed
                self.best_lap = self.last_lap if self.best_lap is None else min(self.best_lap, self.last_lap)
                self.lap_elapsed, self.message = 0.0, f"Lap {self.completed_laps}: {self.last_lap:.2f}s"
        return steering, throttle, handbrake, self.raycast()

    def draw(self, lidar: np.ndarray) -> None:
        draw_track(self.screen, self.track, self.next_checkpoint)
        for angle, distance in zip(LIDAR_ANGLES, lidar):
            radians = math.radians(self.car.angle + float(angle))
            endpoint = self.car.position + pygame.Vector2(math.cos(radians), math.sin(radians)) * distance * MAX_LIDAR_DISTANCE
            pygame.draw.line(self.screen, (99, 197, 238), self.car.position, endpoint, 1)
        pygame.draw.polygon(self.screen, (76, 172, 247) if not self.car.crashed else (130, 40, 40), self.car_points())
        pygame.draw.circle(self.screen, (250, 250, 250), self.car.position + self.car.heading() * 10, 3)
        best = "--" if self.best_lap is None else f"{self.best_lap:.2f}s"
        lines = [f"Lap {self.completed_laps + 1}: {self.lap_elapsed:.2f}s | best: {best} | score: {self.fitness:.0f}",
                 f"checkpoints {self.progress} | speed {self.car.speed:5.1f} | recording: {'ON' if self.recording else 'OFF'}",
                 "WASD/arrows — drive   Shift — drift   R/К — restart   Space — recording   Esc — quit", self.message]
        for i, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, (250, 250, 250) if i < 3 else (255, 226, 102)), (24, 20 + i * 30))
        pygame.display.flip()

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
                        elif event.key == pygame.K_SPACE:
                            self.recording = not self.recording; self.message = f"Recording {'enabled' if self.recording else 'paused'}"
                steering, throttle, handbrake, lidar = self.update(dt)
                if self.recording and not self.car.crashed and self.writer:
                    self.writer.write(self, steering, throttle, handbrake, lidar)
                self.step += 1
                self.draw(lidar)
        finally:
            if self.writer:
                self.writer.close(); print(f"Saved demonstrations to {self.writer.path}")
            pygame.quit()


class MapEditor:
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
                            self.preview_game = RacingGame(self.track, record=False, screen=self.screen)
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
