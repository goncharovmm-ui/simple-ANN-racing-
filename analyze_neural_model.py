"""Create an activation montage, prune weak hidden units, and make a toggle animation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pygame

from racing.neural import NeuralPolicy, make_features
from racing.simulator import HEIGHT, WIDTH, Track, draw_speed_trajectory, draw_track, speed_color


def find_best_csv(track: Track, map_path: Path) -> Path:
    track_id = f"{map_path.stem}-{track.fingerprint()}"
    candidates: list[tuple[float, Path]] = []
    for path in Path("data/demos/race").rglob("*.csv") if Path("data/demos/race").exists() else []:
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if rows and rows[0].get("track_id") == track_id and rows[0].get("driver") == "player":
                lap = max(float(row.get("lap_s") or 0.0) for row in rows)
                candidates.append((lap, path))
        except (OSError, ValueError):
            continue
    if not candidates:
        raise FileNotFoundError(f"No player CSV found for {track_id}")
    return min(candidates, key=lambda item: item[0])[1]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def forward(policy: NeuralPolicy, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lidar = np.asarray([float(row[f"lidar_{i}"]) for i in range(15)], dtype=np.float32)
    features = make_features(lidar, float(row.get("speed") or 0.0))
    hidden = features
    for weight, bias in zip(policy.weights[:-1], policy.biases[:-1]):
        hidden = np.maximum(hidden @ weight + bias, 0.0)
    raw = hidden @ policy.weights[-1] + policy.biases[-1]
    output = raw.copy()
    output[:2] = np.tanh(raw[:2])
    output[2] = 1.0 / (1.0 + np.exp(-raw[2]))
    return features, hidden, output


def node_color(value: float, positive: tuple[int, int, int], negative: tuple[int, int, int] = (90, 95, 105)) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, abs(float(value))))
    return tuple(round(base * (0.25 + 0.75 * amount) + 22 * (1 - amount)) for base in positive)


def signal_alpha(value: float, floor: int = 5) -> int:
    """Map signal strength to alpha; weak activations should visually recede."""
    amount = max(0.0, min(1.0, abs(float(value))))
    return round(floor + (255 - floor) * amount ** 1.8)


def render_tile(screen: pygame.Surface, frame: int, row: dict[str, str], features: np.ndarray, hidden: np.ndarray, output: np.ndarray,
                weights: dict[str, np.ndarray], track: Track, all_samples: list[tuple[pygame.Vector2, float]], font: pygame.font.Font,
                small: pygame.font.Font, model_label: str = "", architecture: list[int] | None = None) -> None:
    tile_x, tile_y = screen.get_width(), screen.get_height()
    screen.fill((25, 30, 38))
    pygame.draw.rect(screen, (44, 50, 60), pygame.Rect(0, 0, tile_x, tile_y), 1)
    title = f"frame {frame + 1}"
    if model_label:
        title += f"  •  {model_label}"
    screen.blit(font.render(title, True, (245, 245, 245)), (10, 8))
    position = pygame.Vector2(float(row["x"]), float(row["y"]))
    speed = float(row.get("speed") or 0.0)
    map_source = pygame.Surface((WIDTH, HEIGHT))
    draw_track(map_source, track, None, True)
    map_surface = pygame.transform.smoothscale(map_source, (220, 145))
    if all_samples:
        scale_x, scale_y = 220 / WIDTH, 145 / HEIGHT
        draw_speed_trajectory(map_surface, [(pygame.Vector2(p.x * scale_x, p.y * scale_y), s) for p, s in all_samples], 2)
    pygame.draw.circle(map_surface, speed_color(speed), (round(position.x * 220 / WIDTH), round(position.y * 145 / HEIGHT)), 5)
    screen.blit(map_surface, (8, 38))
    screen.blit(small.render(f"speed {speed:.1f}", True, (220, 225, 230)), (10, 190))
    hidden_count = len(hidden)
    network_label = " -> ".join(str(value) for value in (architecture or [16, hidden_count, 3]))
    screen.blit(small.render(f"network {network_label}", True, (210, 215, 225)), (10, 215))
    # Keep the network inside the 400 px tile, including output labels.
    input_x, hidden_x, output_x = 235, 270, 336
    input_nodes = [(input_x, 62 + i * 14) for i in range(16)]
    hidden_columns = 8
    hidden_nodes = [(hidden_x + (i % hidden_columns) * 8, 64 + (i // hidden_columns) * 28) for i in range(hidden_count)]
    output_nodes = [(output_x, 105 + i * 70) for i in range(3)]
    contributions = features[:, None] * weights["w1"]
    edge_surface = pygame.Surface((tile_x, tile_y), pygame.SRCALPHA)
    contribution_scale = max(0.001, float(np.percentile(np.abs(contributions), 95)))
    for input_index, source in enumerate(input_nodes):
        for hidden_index, target in enumerate(hidden_nodes):
            amount = min(1.0, abs(float(contributions[input_index, hidden_index])) / contribution_scale)
            if amount < 0.04:
                continue
            color = (70, 220, 130, signal_alpha(amount)) if contributions[input_index, hidden_index] >= 0 else (235, 90, 90, signal_alpha(amount))
            pygame.draw.line(edge_surface, color, source, target, 1)
    hidden_output = hidden[:, None] * weights["w2"]
    output_scale = max(0.001, float(np.percentile(np.abs(hidden_output), 95)))
    for hidden_index, source in enumerate(hidden_nodes):
        for output_index, target in enumerate(output_nodes):
            amount = min(1.0, abs(float(hidden_output[hidden_index, output_index])) / output_scale)
            if amount < 0.04:
                continue
            color = (70, 220, 130, signal_alpha(amount)) if hidden_output[hidden_index, output_index] >= 0 else (235, 90, 90, signal_alpha(amount))
            pygame.draw.line(edge_surface, color, source, target, 1)
    screen.blit(edge_surface, (0, 0))
    node_surface = pygame.Surface((tile_x, tile_y), pygame.SRCALPHA)
    for index, (x, y) in enumerate(input_nodes):
        color = (*node_color(features[index], (80, 180, 240)), signal_alpha(features[index], 24))
        pygame.draw.circle(node_surface, color, (x, y), 4)
    hidden_scale = max(0.001, float(np.percentile(hidden, 95)))
    for index, (x, y) in enumerate(hidden_nodes):
        normalized = hidden[index] / hidden_scale
        color = (*node_color(normalized, (235, 190, 70)), signal_alpha(normalized, 18))
        pygame.draw.circle(node_surface, color, (x, y), 4)
    for index, (x, y) in enumerate(output_nodes):
        color = (*node_color(output[index], (80, 220, 130)), signal_alpha(output[index], 45))
        pygame.draw.circle(node_surface, color, (x, y), 8)
    screen.blit(node_surface, (0, 0))
    for index, (x, y) in enumerate(output_nodes):
        label = ["steer", "gas", "brake"][index]
        screen.blit(small.render(f"{label} {output[index]:+.2f}", True, (235, 240, 245)), (338, y - 8))
    screen.blit(small.render("L0…L14, S", True, (185, 195, 205)), (245, 300))


def ablation_report(policy: NeuralPolicy, rows: list[dict[str, str]]) -> dict[str, object]:
    features, hidden_values, baseline = [], [], []
    for row in rows:
        x, hidden, output = forward(policy, row)
        features.append(x); hidden_values.append(hidden); baseline.append(output)
    x_array, hidden_array, baseline_array = np.asarray(features), np.asarray(hidden_values), np.asarray(baseline)
    influence = []
    for index in range(hidden_array.shape[1]):
        ablated = hidden_array.copy()
        ablated[:, index] = 0.0
        raw = ablated @ policy.w2 + policy.b2
        output = raw.copy()
        output[:, :2] = np.tanh(raw[:, :2])
        output[:, 2] = 1.0 / (1.0 + np.exp(-raw[:, 2]))
        influence.append(float(np.mean(np.abs(output - baseline_array))))
    active_fraction = np.mean(hidden_array > 1e-6, axis=0)
    ranking = np.argsort(influence)[::-1]
    threshold = max(1e-5, max(influence) * 0.05)
    return {"samples": len(rows), "hidden_neurons": len(influence),
            "effective_at_5_percent": int(sum(value >= threshold for value in influence)),
            "prune_threshold": round(float(threshold), 8),
            "influence": [round(float(value), 8) for value in influence],
            "top_neurons": [{"index": int(index), "influence": round(influence[index], 6), "active_fraction": round(float(active_fraction[index]), 4)} for index in ranking[:15]],
            "influence_mean": round(float(np.mean(influence)), 6), "influence_max": round(float(np.max(influence)), 6)}


def prune_policy(policy: NeuralPolicy, report: dict[str, object]) -> tuple[NeuralPolicy, list[int], list[int]]:
    """Remove hidden units whose single-unit ablation changes output by <5% of max."""
    influence = np.asarray(report["influence"], dtype=np.float32)
    threshold = float(report["prune_threshold"])
    keep = [int(index) for index, value in enumerate(influence) if value >= threshold]
    if not keep:
        keep = [int(np.argmax(influence))]
    removed = [index for index in range(policy.w1.shape[1]) if index not in keep]
    weights = [weight.copy() for weight in policy.weights]
    biases = [bias.copy() for bias in policy.biases]
    weights[-2] = weights[-2][:, keep]
    biases[-2] = biases[-2][keep]
    weights[-1] = weights[-1][keep, :]
    return NeuralPolicy(weights, biases), keep, removed


def compare_models(full: NeuralPolicy, pruned: NeuralPolicy, rows: list[dict[str, str]]) -> dict[str, object]:
    full_outputs = np.asarray([forward(full, row)[2] for row in rows], dtype=np.float32)
    pruned_outputs = np.asarray([forward(pruned, row)[2] for row in rows], dtype=np.float32)
    difference = np.abs(full_outputs - pruned_outputs)
    return {
        "mean_abs_output_difference": round(float(np.mean(difference)), 8),
        "max_abs_output_difference": round(float(np.max(difference)), 8),
        "steering_mae": round(float(np.mean(difference[:, 0])), 8),
        "throttle_mae": round(float(np.mean(difference[:, 1])), 8),
        "handbrake_probability_mae": round(float(np.mean(difference[:, 2])), 8),
        "handbrake_decision_agreement": round(float(np.mean((full_outputs[:, 2] >= 0.5) == (pruned_outputs[:, 2] >= 0.5))), 6),
    }


def write_animation(path: Path, frame_indices: np.ndarray, rows: list[dict[str, str]], samples: list[tuple[pygame.Vector2, float]],
                    track: Track, full: NeuralPolicy, pruned: NeuralPolicy) -> None:
    """Write a short MP4 that alternates full and pruned network at each trajectory point."""
    import imageio_ffmpeg

    frame_dir = path.parent / f"{path.stem}-frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    font, small = pygame.font.Font(None, 25), pygame.font.Font(None, 16)
    frame_number = 0
    for frame_index in frame_indices:
        row = rows[int(frame_index)]
        for label, model in ((f"FULL • {full.w1.shape[1]} neurons", full), (f"PRUNED • {pruned.w1.shape[1]} neurons", pruned)):
            features, hidden, output = forward(model, row)
            for _ in range(2):
                tile = pygame.Surface((400, 400))
                render_tile(tile, int(frame_index), row, features, hidden, output,
                            {"w1": model.w1, "w2": model.w2}, track, samples, font, small, label, model.architecture)
                image = pygame.transform.smoothscale(tile, (800, 800))
                pygame.image.save(image, frame_dir / f"frame_{frame_number:04d}.png")
                frame_number += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-y", "-framerate", "2", "-i", str(frame_dir / "frame_%04d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize neural activations over a recorded race")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--map", type=Path, default=Path("maps/default.json"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/neural-analysis.png"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/neural-analysis.json"))
    parser.add_argument("--pruned-model", type=Path, default=Path("artifacts/neural-analysis-pruned.npz"))
    parser.add_argument("--animation", type=Path, default=Path("artifacts/neural-analysis-toggle.mp4"))
    args = parser.parse_args()
    pygame.init()
    track = Track.load(args.map)
    data_path = args.data or find_best_csv(track, args.map)
    rows = load_rows(data_path)
    policy = NeuralPolicy.load(args.model)
    frame_indices = np.linspace(0, len(rows) - 1, 10, dtype=int)
    samples = [(pygame.Vector2(float(row["x"]), float(row["y"])), float(row.get("speed") or 0.0)) for row in rows]
    screen = pygame.Surface((2000, 800))
    font, small = pygame.font.Font(None, 25), pygame.font.Font(None, 16)
    for tile_index, frame_index in enumerate(frame_indices):
        row = rows[int(frame_index)]
        features, hidden, output = forward(policy, row)
        tile = pygame.Surface((400, 400))
        render_tile(tile, int(frame_index), row, features, hidden, output, {"w1": policy.w1, "w2": policy.w2}, track, samples, font, small, "FULL", policy.architecture)
        screen.blit(tile, ((tile_index % 5) * 400, (tile_index // 5) * 400))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(screen, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = ablation_report(policy, rows)
    pruned, keep_indices, removed_indices = prune_policy(policy, report)
    pruned.save(args.pruned_model)
    comparison = compare_models(policy, pruned, rows)
    report.update({"model": str(args.model), "map": str(args.map), "data": str(data_path),
                   "architecture": policy.architecture,
                   "pruned_model": str(args.pruned_model), "pruned_hidden_neurons": int(pruned.w1.shape[1]),
                   "removed_hidden_neurons": removed_indices, "kept_hidden_neurons": keep_indices,
                   "comparison": comparison, "animation": str(args.animation)})
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_animation(args.animation, frame_indices, rows, samples, track, policy, pruned)
    pygame.quit()
    print(f"Saved {args.output}")
    print(json.dumps({key: report[key] for key in ("samples", "architecture", "effective_at_5_percent", "influence_max",
                                                   "pruned_hidden_neurons", "comparison", "pruned_model", "animation")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
