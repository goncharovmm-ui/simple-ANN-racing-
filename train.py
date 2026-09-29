"""Train a small behavioural-cloning policy from manually recorded CSV demonstrations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from racing.neural import NeuralPolicy, make_features, parse_hidden_sizes


def load_demos(directories: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    rows_x, rows_y = [], []
    files = []
    for directory in directories:
        files.extend([directory] if directory.is_file() else sorted(directory.rglob("*.csv")))
    if not files:
        raise ValueError("No CSV demonstrations found. Run a player race or DAgger batch first.")
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


def forward(x: np.ndarray, params: dict[str, list[np.ndarray]]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    activations = [x]
    for weight, bias in zip(params["weights"][:-1], params["biases"][:-1]):
        activations.append(np.maximum(activations[-1] @ weight + bias, 0.0))
    raw = activations[-1] @ params["weights"][-1] + params["biases"][-1]
    output = raw.copy()
    output[:, :2] = np.tanh(raw[:, :2])
    output[:, 2] = 1.0 / (1.0 + np.exp(-raw[:, 2]))
    return activations, raw, output


def loss(x: np.ndarray, y: np.ndarray, params: dict[str, list[np.ndarray]]) -> float:
    _, _, prediction = forward(x, params)
    controls = np.mean((prediction[:, :2] - y[:, :2]) ** 2)
    handbrake = -np.mean(y[:, 2] * np.log(prediction[:, 2] + 1e-7) + (1 - y[:, 2]) * np.log(1 - prediction[:, 2] + 1e-7))
    return float(controls + handbrake)


def make_params(input_size: int, hidden_sizes: list[int], rng: np.random.Generator) -> dict[str, list[np.ndarray]]:
    dimensions = [input_size, *hidden_sizes, 3]
    return {
        "weights": [rng.normal(0, np.sqrt(2 / dimensions[index]), (dimensions[index], dimensions[index + 1])).astype(np.float32)
                    for index in range(len(dimensions) - 1)],
        "biases": [np.zeros(dimensions[index + 1], dtype=np.float32) for index in range(len(dimensions) - 1)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train imitation-learning policy from racing demonstrations")
    parser.add_argument("--data", type=Path, action="append", default=None, help="CSV file or directory; repeat for several datasets")
    parser.add_argument("--output", type=Path, default=Path("models/policy.npz"))
    parser.add_argument("--init-model", type=Path, help="continue training from an existing .npz policy")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-layers", type=int, default=None, help="number of hidden layers")
    parser.add_argument("--hidden-units", type=str, default=None, help="neurons per layer: 64 or 128,64,32")
    parser.add_argument("--metrics-output", type=Path, help="optional JSON file for the final train/validation metrics")
    args = parser.parse_args()

    x, y = load_demos(args.data or [Path("data/demos")])
    if len(x) < 100:
        raise ValueError(f"Only {len(x)} samples. Record at least a few complete laps before training.")
    rng = np.random.default_rng(42)
    order = rng.permutation(len(x))
    split = max(1, int(len(x) * 0.9))
    train_i, valid_i = order[:split], order[split:]
    x_train, y_train = x[train_i], y[train_i]
    x_valid, y_valid = x[valid_i], y[valid_i]
    requested_sizes = None if args.hidden_layers is None and args.hidden_units is None else parse_hidden_sizes(args.hidden_layers, args.hidden_units)
    if args.init_model:
        loaded = NeuralPolicy.load(args.init_model)
        if requested_sizes and loaded.hidden_sizes != requested_sizes:
            print(f"Init model architecture {loaded.hidden_sizes} differs from requested {requested_sizes}; reinitializing weights")
            params = make_params(x.shape[1], requested_sizes, rng)
        else:
            params = {"weights": [weight.copy() for weight in loaded.weights], "biases": [bias.copy() for bias in loaded.biases]}
    else:
        params = make_params(x.shape[1], requested_sizes or [64], rng)
    moments = {"weights": [np.zeros_like(value) for value in params["weights"]],
               "biases": [np.zeros_like(value) for value in params["biases"]]}
    variances = {"weights": [np.zeros_like(value) for value in params["weights"]],
                 "biases": [np.zeros_like(value) for value in params["biases"]]}
    update = 0
    final_validation = None
    for epoch in range(1, args.epochs + 1):
        for indices in rng.permutation(len(x_train)).reshape(-1, 1) if len(x_train) == 1 else np.array_split(rng.permutation(len(x_train)), max(1, len(x_train) // args.batch_size)):
            xb, yb = x_train[indices], y_train[indices]
            activations, _, output = forward(xb, params)
            dz2 = np.empty_like(output)
            dz2[:, :2] = (output[:, :2] - yb[:, :2]) * (1 - output[:, :2] ** 2) / len(xb)
            dz2[:, 2] = (output[:, 2] - yb[:, 2]) / len(xb)
            gradients = {"weights": [np.zeros_like(value) for value in params["weights"]],
                         "biases": [np.zeros_like(value) for value in params["biases"]]}
            gradient = dz2
            for layer in reversed(range(len(params["weights"]))):
                gradients["weights"][layer] = activations[layer].T @ gradient
                gradients["biases"][layer] = gradient.sum(axis=0)
                if layer > 0:
                    gradient = (gradient @ params["weights"][layer].T) * (activations[layer] > 0)
            update += 1
            for name in ("weights", "biases"):
                for layer in range(len(params[name])):
                    moments[name][layer] = 0.9 * moments[name][layer] + 0.1 * gradients[name][layer]
                    variances[name][layer] = 0.999 * variances[name][layer] + 0.001 * gradients[name][layer] ** 2
                    corrected_m = moments[name][layer] / (1 - 0.9 ** update)
                    corrected_v = variances[name][layer] / (1 - 0.999 ** update)
                    params[name][layer] -= args.learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            validation = loss(x_valid, y_valid, params) if len(x_valid) else loss(x_train, y_train, params)
            final_validation = validation
            print(f"epoch {epoch:>3}/{args.epochs}: validation loss {validation:.5f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    NeuralPolicy(params["weights"], params["biases"]).save(args.output)
    policy = NeuralPolicy(params["weights"], params["biases"])
    metrics = {"architecture": policy.architecture, "samples": len(x), "train_samples": len(x_train),
               "validation_samples": len(x_valid), "epochs": args.epochs, "validation_loss": final_validation}
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved policy to {args.output} ({len(x)} samples; {len(x_train)} train, {len(x_valid)} validation; architecture {policy.architecture})")


if __name__ == "__main__":
    main()
