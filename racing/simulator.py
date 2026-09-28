from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pygame

WIDTH, HEIGHT = 1280, 800
ROAD = pygame.Rect(80, 80, 1120, 640)
INNER_GRASS = pygame.Rect(330, 255, 620, 290)
START = pygame.Vector2(245, 170)
START_ANGLE = 0.0
CHECKPOINTS = [
    pygame.Vector2(570, 170), pygame.Vector2(1030, 170), pygame.Vector2(1090, 370),
    pygame.Vector2(1090, 625), pygame.Vector2(650, 630), pygame.Vector2(190, 630),
    pygame.Vector2(170, 390),
]
OBSTACLES = [
    pygame.Rect(610, 120, 48, 105), pygame.Rect(1000, 500, 70, 48),
    pygame.Rect(215, 530, 70, 48),
]
LIDAR_ANGLES = np.linspace(-110.0, 110.0, 15, dtype=np.float32)
MAX_LIDAR_DISTANCE = 260.0


@dataclass
class Car:
    position: pygame.Vector2
    angle: float = START_ANGLE
    speed: float = 0.0
    crashed: bool = False

    def heading(self) -> pygame.Vector2:
        radians = math.radians(self.angle)
        return pygame.Vector2(math.cos(radians), math.sin(radians))

    def update(self, steering: float, throttle: float, dt: float) -> None:
        if self.crashed:
            return
        steering = max(-1.0, min(1.0, steering))
        throttle = max(-1.0, min(1.0, throttle))
        self.speed += throttle * 230.0 * dt
        self.speed *= 0.992 ** (dt * 60.0)
        self.speed = max(-75.0, min(320.0, self.speed))
        turn_factor = min(1.0, abs(self.speed) / 65.0)
        self.angle += steering * 155.0 * turn_factor * dt * (1 if self.speed >= 0 else -1)
        self.position += self.heading() * self.speed * dt


class DemoWriter:
    fields = [
        "episode", "step", "x", "y", "angle", "speed", "next_checkpoint",
        "progress", "steering", "throttle", *[f"lidar_{i}" for i in range(len(LIDAR_ANGLES))],
    ]

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = directory / f"demo_{stamp}.csv"
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.writer.writeheader()

    def write(self, episode: int, step: int, car: Car, checkpoint: int, progress: int,
              steering: float, throttle: float, lidar: np.ndarray) -> None:
        row = {
            "episode": episode, "step": step, "x": round(car.position.x, 3),
            "y": round(car.position.y, 3), "angle": round(car.angle, 3),
            "speed": round(car.speed, 3), "next_checkpoint": checkpoint,
            "progress": progress, "steering": round(steering, 3), "throttle": round(throttle, 3),
        }
        row.update({f"lidar_{i}": round(float(value), 4) for i, value in enumerate(lidar)})
        self.writer.writerow(row)

    def close(self) -> None:
        self.file.close()


class RacingGame:
    def __init__(self, record: bool = False) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simple ANN Racing — manual data collection")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.car = Car(START.copy())
        self.next_checkpoint = 0
        self.progress = 0
        self.episode = 1
        self.step = 0
        self.recording = record
        self.writer = DemoWriter(Path("data/demos")) if record else None
        self.message = ""

    @staticmethod
    def is_road(point: pygame.Vector2) -> bool:
        return ROAD.collidepoint(point) and not INNER_GRASS.collidepoint(point) and not any(
            obstacle.collidepoint(point) for obstacle in OBSTACLES
        )

    def car_points(self) -> list[pygame.Vector2]:
        forward = self.car.heading()
        side = pygame.Vector2(-forward.y, forward.x)
        return [self.car.position + forward * 16 + side * 10, self.car.position + forward * 16 - side * 10,
                self.car.position - forward * 16 - side * 10, self.car.position - forward * 16 + side * 10]

    def raycast(self) -> np.ndarray:
        values = []
        for relative_angle in LIDAR_ANGLES:
            radians = math.radians(self.car.angle + float(relative_angle))
            direction = pygame.Vector2(math.cos(radians), math.sin(radians))
            distance = 0.0
            while distance < MAX_LIDAR_DISTANCE:
                point = self.car.position + direction * distance
                if not self.is_road(point):
                    break
                distance += 4.0
            values.append(min(distance, MAX_LIDAR_DISTANCE) / MAX_LIDAR_DISTANCE)
        return np.asarray(values, dtype=np.float32)

    def reset(self, reason: str) -> None:
        self.car = Car(START.copy())
        self.next_checkpoint = 0
        self.progress = 0
        self.step = 0
        self.episode += 1
        self.message = reason

    def update(self, dt: float) -> tuple[float, float, np.ndarray]:
        keys = pygame.key.get_pressed()
        steering = float(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - float(keys[pygame.K_LEFT] or keys[pygame.K_a])
        throttle = float(keys[pygame.K_UP] or keys[pygame.K_w]) - float(keys[pygame.K_DOWN] or keys[pygame.K_s])
        self.car.update(steering, throttle, dt)
        if not all(self.is_road(point) for point in self.car_points()):
            self.car.crashed = True
            self.message = "Collision — press R to restart"
        target = CHECKPOINTS[self.next_checkpoint]
        if self.car.position.distance_to(target) < 58:
            self.progress += 1
            self.next_checkpoint = (self.next_checkpoint + 1) % len(CHECKPOINTS)
            if self.next_checkpoint == 0:
                self.message = "Lap completed!"
        lidar = self.raycast()
        return steering, throttle, lidar

    def draw(self, lidar: np.ndarray) -> None:
        self.screen.fill((36, 112, 57))
        pygame.draw.rect(self.screen, (61, 65, 72), ROAD, border_radius=28)
        pygame.draw.rect(self.screen, (36, 112, 57), INNER_GRASS, border_radius=20)
        pygame.draw.rect(self.screen, (239, 110, 64), ROAD, width=4, border_radius=28)
        pygame.draw.rect(self.screen, (239, 110, 64), INNER_GRASS, width=4, border_radius=20)
        for obstacle in OBSTACLES:
            pygame.draw.rect(self.screen, (193, 58, 55), obstacle, border_radius=5)
        for index, checkpoint in enumerate(CHECKPOINTS):
            color = (246, 208, 74) if index == self.next_checkpoint else (170, 174, 178)
            pygame.draw.circle(self.screen, color, checkpoint, 9)
        for relative_angle, normalized_distance in zip(LIDAR_ANGLES, lidar):
            radians = math.radians(self.car.angle + float(relative_angle))
            endpoint = self.car.position + pygame.Vector2(math.cos(radians), math.sin(radians)) * (normalized_distance * MAX_LIDAR_DISTANCE)
            pygame.draw.line(self.screen, (99, 197, 238), self.car.position, endpoint, 1)
        polygon = [(point.x, point.y) for point in self.car_points()]
        pygame.draw.polygon(self.screen, (76, 172, 247) if not self.car.crashed else (130, 40, 40), polygon)
        pygame.draw.circle(self.screen, (250, 250, 250), self.car.position + self.car.heading() * 10, 3)
        status = f"Episode {self.episode} | checkpoints {self.progress} | speed {self.car.speed:5.1f} | recording: {'ON' if self.recording else 'OFF'}"
        hint = "WASD / arrows — drive   R/К — restart   Space — toggle recording   Esc — quit"
        self.screen.blit(self.font.render(status, True, (250, 250, 250)), (24, 20))
        self.screen.blit(self.font.render(hint, True, (250, 250, 250)), (24, 50))
        if self.message:
            self.screen.blit(self.font.render(self.message, True, (255, 226, 102)), (24, 80))
        pygame.display.flip()

    def run(self) -> None:
        running = True
        try:
            while running:
                dt = min(self.clock.tick(60) / 1000.0, 0.05)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r or event.unicode.lower() == "к":
                            self.reset("Restarted")
                        elif event.key == pygame.K_SPACE:
                            self.recording = not self.recording
                            self.message = f"Recording {'enabled' if self.recording else 'paused'}"
                steering, throttle, lidar = self.update(dt)
                if self.recording and not self.car.crashed and self.writer:
                    self.writer.write(self.episode, self.step, self.car, self.next_checkpoint,
                                      self.progress, steering, throttle, lidar)
                self.step += 1
                self.draw(lidar)
        finally:
            if self.writer:
                self.writer.close()
                print(f"Saved demonstrations to {self.writer.path}")
            pygame.quit()
