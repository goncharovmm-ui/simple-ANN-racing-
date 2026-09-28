"""Train a small behavioural-cloning policy from manually recorded CSV demonstrations."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from racing.neural import make_features


def load_demos(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    rows_x, rows_y = [], []
    files = sorted(directory.rglob("*.csv"))
    if not files:
        raise ValueError(f"No CSV demonstrations in {directory}. Run: python main.py --record")
    for path in files:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    lidar = np.array([float(row[f"lidar_{i}"]) for i in range(15)], dtype=np.float32)
                    rows_x.append(make_features(lidar, float(row["speed"])))
                    rows_y.append([float(row["steering"]), float(row["throttle"]), float(row.get("handbrake", 0))])
                except (KeyError, ValueError):
                    continue
    if not rows_x:
        raise ValueError("Demonstrations have no valid rows.")
    return np.asarray(rows_x, dtype=np.float32), np.asarray(rows_y, dtype=np.float32)


def forward(x: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = np.maximum(x @ params["w1"] + params["b1"], 0.0)
    raw = hidden @ params["w2"] + params["b2"]
    output = raw.copy()
    output[:, :2] = np.tanh(raw[:, :2])
    output[:, 2] = 1.0 / (1.0 + np.exp(-raw[:, 2]))
    return hidden, raw, output


def loss(x: np.ndarray, y: np.ndarray, params: dict[str, np.ndarray]) -> float:
    _, _, prediction = forward(x, params)
    controls = np.mean((prediction[:, :2] - y[:, :2]) ** 2)
    handbrake = -np.mean(y[:, 2] * np.log(prediction[:, 2] + 1e-7) + (1 - y[:, 2]) * np.log(1 - prediction[:, 2] + 1e-7))
    return float(controls + handbrake)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train imitation-learning policy from racing demonstrations")
    parser.add_argument("--data", type=Path, default=Path("data/demos"))
    parser.add_argument("--output", type=Path, default=Path("models/policy.npz"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    x, y = load_demos(args.data)
    if len(x) < 100:
        raise ValueError(f"Only {len(x)} samples. Record at least a few complete laps before training.")
    rng = np.random.default_rng(42)
    order = rng.permutation(len(x))
    split = max(1, int(len(x) * 0.9))
    train_i, valid_i = order[:split], order[split:]
    x_train, y_train = x[train_i], y[train_i]
    x_valid, y_valid = x[valid_i], y[valid_i]
    hidden_size = 64
    params = {
        "w1": rng.normal(0, np.sqrt(2 / x.shape[1]), (x.shape[1], hidden_size)).astype(np.float32),
        "b1": np.zeros(hidden_size, dtype=np.float32),
        "w2": rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, 3)).astype(np.float32),
        "b2": np.zeros(3, dtype=np.float32),
    }
    moments = {name: np.zeros_like(value) for name, value in params.items()}
    variances = {name: np.zeros_like(value) for name, value in params.items()}
    update = 0
    for epoch in range(1, args.epochs + 1):
        for indices in rng.permutation(len(x_train)).reshape(-1, 1) if len(x_train) == 1 else np.array_split(rng.permutation(len(x_train)), max(1, len(x_train) // args.batch_size)):
            xb, yb = x_train[indices], y_train[indices]
            hidden, _, output = forward(xb, params)
            dz2 = np.empty_like(output)
            dz2[:, :2] = (output[:, :2] - yb[:, :2]) * (1 - output[:, :2] ** 2) / len(xb)
            dz2[:, 2] = (output[:, 2] - yb[:, 2]) / len(xb)
            grads = {"w2": hidden.T @ dz2, "b2": dz2.sum(axis=0)}
            dhidden = dz2 @ params["w2"].T
            dz1 = dhidden * (hidden > 0)
            grads["w1"], grads["b1"] = xb.T @ dz1, dz1.sum(axis=0)
            update += 1
            for name in params:
                moments[name] = 0.9 * moments[name] + 0.1 * grads[name]
                variances[name] = 0.999 * variances[name] + 0.001 * grads[name] ** 2
                corrected_m = moments[name] / (1 - 0.9 ** update)
                corrected_v = variances[name] / (1 - 0.999 ** update)
                params[name] -= args.learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            validation = loss(x_valid, y_valid, params) if len(x_valid) else loss(x_train, y_train, params)
            print(f"epoch {epoch:>3}/{args.epochs}: validation loss {validation:.5f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **params)
    print(f"Saved policy to {args.output} ({len(x)} samples; {len(x_train)} train, {len(x_valid)} validation)")


if __name__ == "__main__":
    main()
