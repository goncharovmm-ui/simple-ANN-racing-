"""Measure how far a simple lidar correlation rule gets without a neural network."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_rows(paths: list[Path], driver: str | None = "player", track_id: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    differences, fronts, speeds, targets, tracks = [], [], [], [], []
    files: list[Path] = []
    for path in paths:
        files.extend([path] if path.is_file() else sorted(path.rglob("*.csv")))
    for path in files:
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if driver and row.get("driver") != driver:
                        continue
                    if track_id and row.get("track_id") != track_id:
                        continue
                    try:
                        lidar = np.asarray([float(row[f"lidar_{index}"]) for index in range(15)], dtype=np.float32)
                        steering = float(row["steering"])
                        throttle = float(row.get("throttle") or 0.0)
                        speed = float(row.get("speed") or 0.0) / 360.0
                    except (KeyError, TypeError, ValueError):
                        continue
                    differences.append(float(np.mean(lidar[8:]) - np.mean(lidar[:7])))
                    fronts.append(float(np.mean(lidar[6:9])))
                    speeds.append(speed)
                    targets.append((steering, throttle))
                    tracks.append(row.get("track_id", "unknown"))
        except OSError:
            continue
    if not targets:
        raise ValueError("No matching rows found")
    return np.asarray(differences), np.asarray(fronts), np.asarray(speeds), np.asarray(targets), sorted(set(tracks))


def correlation(left_right: np.ndarray, target: np.ndarray) -> float:
    if np.std(left_right) < 1e-8 or np.std(target) < 1e-8:
        return 0.0
    return float(np.corrcoef(left_right, target)[0, 1])


def rule_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.abs(prediction - target)
    active = np.abs(target) >= 0.05
    return {
        "mae": round(float(np.mean(error)), 6),
        "rmse": round(float(np.sqrt(np.mean((prediction - target) ** 2))), 6),
        "sign_agreement": round(float(np.mean((np.sign(prediction[active]) == np.sign(target[active])))) if np.any(active) else 0.0, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a non-neural lidar correlation baseline")
    parser.add_argument("--data", type=Path, action="append", default=None)
    parser.add_argument("--driver", default="player", help="player, artificial_expert, or empty for every driver")
    parser.add_argument("--track-id", default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rule-baseline.json"))
    args = parser.parse_args()
    driver = args.driver or None
    difference, front, speed, targets, tracks = load_rows(args.data or [Path("data/demos")], driver, args.track_id)
    steering, throttle = targets[:, 0], targets[:, 1]
    order = np.random.default_rng(42).permutation(len(targets))
    split = max(1, round(len(order) * 0.8))
    train, valid = order[:split], order[split:]
    gains = np.linspace(0.1, 12.0, 240)
    best_gain = min(gains, key=lambda gain: float(np.mean(np.abs(np.clip(gain * difference[train], -1.0, 1.0) - steering[train]))))
    rule_steering = np.clip(best_gain * difference, -1.0, 1.0)
    constant = np.full_like(steering, float(np.mean(steering[train])))
    # A linear, still non-neural baseline using only three interpretable values.
    design = np.column_stack((np.ones(len(targets)), difference, front, speed))
    coefficients, *_ = np.linalg.lstsq(design[train], steering[train], rcond=None)
    linear_steering = np.clip(design @ coefficients, -1.0, 1.0)
    throttle_rule = np.clip(1.35 * front - 0.45 * np.abs(difference), -1.0, 1.0)
    report = {
        "samples": int(len(targets)),
        "tracks": tracks,
        "driver": driver or "all",
        "correlation_right_minus_left_to_steering": round(correlation(difference, steering), 6),
        "correlation_front_to_throttle": round(correlation(front, throttle), 6),
        "rule": {"gain": round(float(best_gain), 6), "train": rule_metrics(rule_steering[train], steering[train]), "validation": rule_metrics(rule_steering[valid], steering[valid])},
        "constant_steering": {"validation": rule_metrics(constant[valid], steering[valid])},
        "linear_three_feature_steering": {"coefficients": [round(float(value), 6) for value in coefficients], "validation": rule_metrics(linear_steering[valid], steering[valid])},
        "throttle_rule": rule_metrics(throttle_rule[valid], throttle[valid]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
