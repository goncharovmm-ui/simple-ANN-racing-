"""Frame-by-frame neural signal replay for a recorded human run."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pygame

from racing.neural import NeuralPolicy
from racing.simulator import HEIGHT, LIDAR_ANGLES, MAX_LIDAR_DISTANCE, Track, draw_speed_trajectory, draw_track, speed_color


class ReplayViewer:
    """Animate one recorded trajectory while showing the selected network outputs."""

    def __init__(self, track: Track, map_path: Path, policy_path: Path, replay_path: Path | None = None) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((1280, HEIGHT))
        pygame.display.set_caption("Simple ANN Racing — neural replay")
        self.clock = pygame.time.Clock()
        self.font, self.small_font = pygame.font.Font(None, 28), pygame.font.Font(None, 22)
        self.track, self.map_path, self.policy_path = track, map_path, policy_path
        self.policy = NeuralPolicy.load(policy_path)
        self.replays = [replay_path] if replay_path and replay_path.exists() else self.find_replays()
        self.replay_index = 0
        self.rows: list[dict[str, str]] = []
        self.samples: list[tuple[pygame.Vector2, float]] = []
        self.outputs: list[tuple[float, float, bool]] = []
        self.frame_index, self.accumulator = 0, 0.0
        self.fps, self.playback_speed = 30, 1.0
        self.paused = False
        self.load_replay()

    def find_replays(self) -> list[Path]:
        track_id = f"{self.map_path.stem}-{self.track.fingerprint()}"
        directory = Path("data/demos/race")
        candidates: list[tuple[float, Path]] = []
        if not directory.exists():
            return []
        for path in directory.rglob("*.csv"):
            try:
                with path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if not rows or rows[0].get("track_id") != track_id or rows[0].get("driver") != "player":
                    continue
                lap = max(float(row.get("lap_s") or 0.0) for row in rows)
                candidates.append((lap, path))
            except (OSError, ValueError):
                continue
        return [path for _, path in sorted(candidates, key=lambda item: item[0])]

    def load_replay(self) -> None:
        self.rows, self.samples, self.outputs = [], [], []
        self.frame_index, self.accumulator, self.paused = 0, 0.0, False
        if not self.replays:
            return
        try:
            with self.replays[self.replay_index].open(encoding="utf-8", newline="") as handle:
                self.rows = list(csv.DictReader(handle))
            for row in self.rows:
                lidar = np.asarray([float(row[f"lidar_{i}"]) for i in range(15)], dtype=np.float32)
                speed = float(row.get("speed") or 0.0)
                self.samples.append((pygame.Vector2(float(row["x"]), float(row["y"])), speed))
                self.outputs.append(self.policy.act(lidar, speed))
        except (OSError, KeyError, ValueError):
            self.rows, self.samples, self.outputs = [], [], []

    def next_replay(self, direction: int) -> None:
        if not self.replays:
            return
        self.replay_index = (self.replay_index + direction) % len(self.replays)
        self.load_replay()

    def draw_bar(self, label: str, value: float, y: int, color: tuple[int, int, int], minimum: float = -1.0, maximum: float = 1.0) -> None:
        left, width = 970, 245
        self.screen.blit(self.small_font.render(label, True, (235, 235, 235)), (900, y))
        pygame.draw.rect(self.screen, (55, 60, 70), pygame.Rect(left, y, width, 20), border_radius=4)
        ratio = max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
        pygame.draw.rect(self.screen, color, pygame.Rect(left, y, round(width * ratio), 20), border_radius=4)
        self.screen.blit(self.small_font.render(f"{value:+.3f}", True, (245, 245, 245)), (1225, y))

    def draw(self) -> None:
        draw_track(self.screen, self.track, None, True)
        if self.samples:
            draw_speed_trajectory(self.screen, self.samples, 3)
            position, speed = self.samples[self.frame_index]
            row = self.rows[self.frame_index]
            steering, throttle, handbrake = self.outputs[self.frame_index]
            angle = float(row.get("angle") or 0.0)
            heading = pygame.Vector2(math.cos(math.radians(angle)), math.sin(math.radians(angle)))
            side = pygame.Vector2(-heading.y, heading.x)
            points = [position + heading * 16 + side * 10, position + heading * 16 - side * 10, position - heading * 16 - side * 10, position - heading * 16 + side * 10]
            pygame.draw.polygon(self.screen, speed_color(speed), points)
            for lidar_angle, distance in zip(LIDAR_ANGLES, [float(row[f"lidar_{i}"]) for i in range(15)]):
                direction = pygame.Vector2(math.cos(math.radians(angle + float(lidar_angle))), math.sin(math.radians(angle + float(lidar_angle))))
                pygame.draw.line(self.screen, (80, 180, 230), position, position + direction * distance * MAX_LIDAR_DISTANCE, 1)
            panel = pygame.Rect(875, 0, 405, HEIGHT)
            pygame.draw.rect(self.screen, (23, 29, 38), panel)
            self.screen.blit(self.font.render("Neural signal replay", True, (250, 250, 250)), (900, 28))
            self.screen.blit(self.small_font.render(self.policy_path.as_posix()[-55:], True, (190, 200, 210)), (900, 65))
            self.screen.blit(self.small_font.render(self.replays[self.replay_index].name, True, (190, 200, 210)), (900, 92))
            self.draw_bar("steering", steering, 145, (243, 178, 67))
            self.draw_bar("throttle", throttle, 195, (70, 210, 120))
            self.draw_bar("speed", speed / 180.0 - 1.0, 245, speed_color(speed), -1.0, 1.0)
            self.screen.blit(self.small_font.render(f"handbrake: {'ON' if handbrake else 'off'}", True, (255, 210, 100)), (900, 295))
            self.screen.blit(self.small_font.render(f"frame: {self.frame_index + 1}/{len(self.rows)}   recorded speed: {speed:.1f}", True, (235, 235, 235)), (900, 335))
            self.screen.blit(self.small_font.render(f"FPS cap: {self.fps}   playback: {self.playback_speed:.2f}x", True, (235, 235, 235)), (900, 365))
            self.screen.blit(self.small_font.render("Space pause | ↑/↓ speed | +/- FPS", True, (190, 200, 210)), (900, 420))
            self.screen.blit(self.small_font.render("←/→ frame | N/P run | R restart | Esc menu", True, (190, 200, 210)), (900, 447))
        else:
            self.screen.blit(self.font.render("Нет завершённых ручных записей для этой карты", True, (255, 220, 110)), (60, 120))
            self.screen.blit(self.small_font.render("Сначала пройдите полный круг в режиме Player.", True, (230, 230, 230)), (60, 160))
        pygame.display.flip()

    def run(self) -> None:
        running = True
        try:
            while running:
                dt = self.clock.tick(self.fps) / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                        elif event.key == pygame.K_UP:
                            self.playback_speed = min(16.0, self.playback_speed * 2)
                        elif event.key == pygame.K_DOWN:
                            self.playback_speed = max(0.125, self.playback_speed / 2)
                        elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                            self.fps = min(120, self.fps + 5)
                        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            self.fps = max(1, self.fps - 5)
                        elif event.key == pygame.K_RIGHT and self.rows:
                            self.frame_index = min(len(self.rows) - 1, self.frame_index + 1)
                            self.paused = True
                        elif event.key == pygame.K_LEFT and self.rows:
                            self.frame_index = max(0, self.frame_index - 1)
                            self.paused = True
                        elif event.key == pygame.K_n:
                            self.next_replay(1)
                        elif event.key == pygame.K_p:
                            self.next_replay(-1)
                        elif event.key == pygame.K_r:
                            self.frame_index, self.accumulator, self.paused = 0, 0.0, False
                if self.rows and not self.paused:
                    self.accumulator += dt * 60.0 * self.playback_speed
                    while self.accumulator >= 1.0:
                        self.frame_index += 1
                        self.accumulator -= 1.0
                        if self.frame_index >= len(self.rows):
                            self.frame_index = len(self.rows) - 1
                            self.paused = True
                            break
                self.draw()
        finally:
            pygame.quit()
