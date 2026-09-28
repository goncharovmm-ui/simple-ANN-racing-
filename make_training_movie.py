"""Render the recorded headless-training history as a five-minute MP4 time-lapse."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio_ffmpeg
import pygame

from racing.simulator import Track, WIDTH, HEIGHT, draw_track


def trajectory_color(trajectory: dict, fastest: float | None, slowest: float | None) -> tuple[int, int, int]:
    """Colour a completed lap by its time; keep failed attempts red."""
    if not trajectory["success"]:
        return (220, 45, 45)
    if fastest is None or slowest == fastest:
        return (0, 255, 0)
    green = round(255 * (slowest - trajectory["lap_seconds"]) / (slowest - fastest))
    return (0, green, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a five-minute training movie from training-history.json")
    parser.add_argument("--history", type=Path, default=Path("artifacts/training-history.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/training-progress.mp4"))
    parser.add_argument("--duration", type=int, default=300)
    args = parser.parse_args()
    history = json.loads(args.history.read_text(encoding="utf-8"))
    raw = history["track"]
    track = Track(pygame.Rect(raw["road"]), pygame.Rect(raw["inner_grass"]), pygame.Vector2(raw.get("start", [245, 170])), 0.0,
                  pygame.Vector2(raw["finish"]), [pygame.Vector2(point) for point in raw["checkpoints"]], [pygame.Rect(rect) for rect in raw["obstacles"]])
    pygame.init()
    screen, font = pygame.Surface((WIDTH, HEIGHT)), pygame.font.Font(None, 30)
    frames = args.output.parent / "training-frames"
    frames.mkdir(parents=True, exist_ok=True)
    frame_number, per_epoch = 0, max(1, args.duration // len(history["epochs"]))
    epochs = history["epochs"]
    for epoch_index, epoch in enumerate(epochs):
        for phase in range(1, per_epoch + 1):
            draw_track(screen, track)
            fraction = phase / per_epoch
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            # Draw every completed epoch together.  The newest one is opaque;
            # older generations remain visible as a transparent history layer.
            for source_index, source_epoch in enumerate(epochs[:epoch_index + 1]):
                successful_times = [item["lap_seconds"] for item in source_epoch["trajectories"] if item["success"] and item["lap_seconds"] is not None]
                fastest, slowest = (min(successful_times), max(successful_times)) if successful_times else (None, None)
                alpha = round(35 + 220 * (source_index + 1) / (epoch_index + 1))
                for trajectory in source_epoch["trajectories"]:
                    points = [pygame.Vector2(point) for point in trajectory["points"]]
                    point_count = len(points) if source_index < epoch_index else max(2, round(len(points) * fraction))
                    visible = points[:point_count]
                    if len(visible) > 1:
                        pygame.draw.lines(overlay, (*trajectory_color(trajectory, fastest, slowest), alpha), False, visible, 3)
            screen.blit(overlay, (0, 0))
            labels = [f"DAgger training: epoch {epoch['epoch']}/{len(epochs)} | 20 parallel cars", f"finished laps: {epoch['finished']}/{epoch['agents']} | phase {phase}/{per_epoch}", "all previous epochs fade out; black → green = slowest → fastest finish; red = failed"]
            for index, label in enumerate(labels):
                screen.blit(font.render(label, True, (250, 250, 250)), (20, 18 + index * 30))
            pygame.image.save(screen, frames / f"frame_{frame_number:04d}.png")
            frame_number += 1
    pygame.quit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", "1", "-i", str(frames / "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", str(args.output)]
    subprocess.run(command, check=True)
    print(f"Created {args.output} ({args.duration} seconds)")


if __name__ == "__main__":
    main()
