"""Small dependency-free neural policy used by the simulator and training script."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def make_features(lidar: np.ndarray, speed: float) -> np.ndarray:
    """The observation contract shared by collection, training and autonomous driving."""
    return np.concatenate((np.asarray(lidar, dtype=np.float32), [np.clip(speed / 360.0, -1.0, 1.0)])).astype(np.float32)


def parse_hidden_sizes(hidden_layers: int | None = None, hidden_units: str | None = None,
                      fallback: list[int] | None = None) -> list[int]:
    """Resolve CLI/UI architecture settings into one width per hidden layer."""
    if hidden_layers is None and hidden_units is None:
        return list(fallback or [64])
    layers = max(1, int(hidden_layers or 1))
    tokens = [token.strip() for token in (hidden_units or "64").split(",") if token.strip()]
    if not tokens:
        tokens = ["64"]
    try:
        widths = [int(token) for token in tokens]
    except ValueError as error:
        raise ValueError("hidden units must be comma-separated positive integers, e.g. 64,32") from error
    if len(widths) == 1:
        widths *= layers
    elif len(widths) != layers:
        raise ValueError(f"expected one width or {layers} widths, got {len(widths)}")
    if any(width < 1 or width > 2048 for width in widths):
        raise ValueError("each hidden layer must contain 1..2048 neurons")
    return widths


@dataclass
class NeuralPolicy:
    """A ReLU MLP with 16 inputs, configurable hidden layers and 3 controls."""
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.biases) or len(self.weights) < 2:
            raise ValueError("a policy needs at least one hidden layer")
        for weight, bias in zip(self.weights, self.biases):
            if weight.ndim != 2 or bias.ndim != 1 or weight.shape[1] != bias.shape[0]:
                raise ValueError("incompatible neural layer shapes")
        if self.weights[0].shape[0] != 16 or self.weights[-1].shape[1] != 3:
            raise ValueError("policy must have 16 inputs and 3 outputs")
        for previous, current in zip(self.weights, self.weights[1:]):
            if previous.shape[1] != current.shape[0]:
                raise ValueError("neural layers have incompatible widths")

    @property
    def architecture(self) -> list[int]:
        return [int(self.weights[0].shape[0]), *[int(weight.shape[1]) for weight in self.weights]]

    @property
    def hidden_sizes(self) -> list[int]:
        return self.architecture[1:-1]

    # Compatibility aliases for older tools and one-hidden-layer models.
    @property
    def w1(self) -> np.ndarray:
        return self.weights[0]

    @property
    def b1(self) -> np.ndarray:
        return self.biases[0]

    @property
    def w2(self) -> np.ndarray:
        return self.weights[-1]

    @property
    def b2(self) -> np.ndarray:
        return self.biases[-1]

    @classmethod
    def load(cls, path: Path) -> "NeuralPolicy":
        with np.load(path) as data:
            if "layer_w0" in data.files:
                indices = sorted(int(name.removeprefix("layer_w")) for name in data.files if name.startswith("layer_w"))
                weights = [data[f"layer_w{index}"].copy() for index in indices]
                biases = [data[f"layer_b{index}"].copy() for index in indices]
                return cls(weights, biases)
            # Legacy files contain w1/b1/w2/b2 and remain fully supported.
            return cls([data["w1"].copy(), data["w2"].copy()], [data["b1"].copy(), data["b2"].copy()])

    @classmethod
    def random(cls, seed: int = 42, hidden_size: int = 64, hidden_sizes: list[int] | None = None) -> "NeuralPolicy":
        widths = list(hidden_sizes or [hidden_size])
        rng = np.random.default_rng(seed)
        dimensions = [16, *widths, 3]
        weights = [rng.normal(0, np.sqrt(2 / dimensions[index]), (dimensions[index], dimensions[index + 1])).astype(np.float32)
                   for index in range(len(dimensions) - 1)]
        biases = [np.zeros(dimensions[index + 1], dtype=np.float32) for index in range(len(dimensions) - 1)]
        return cls(weights, biases)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {"architecture": np.asarray(self.architecture, dtype=np.int32)}
        payload.update({f"layer_w{index}": weight for index, weight in enumerate(self.weights)})
        payload.update({f"layer_b{index}": bias for index, bias in enumerate(self.biases)})
        # Keep old key names for external scripts that only understand 1-hidden-layer files.
        if len(self.weights) == 2:
            payload.update({"w1": self.weights[0], "b1": self.biases[0], "w2": self.weights[1], "b2": self.biases[1]})
        np.savez(path, **payload)

    def predict(self, features: np.ndarray) -> np.ndarray:
        activation = np.asarray(features, dtype=np.float32)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            activation = np.maximum(activation @ weight + bias, 0.0)
        raw = activation @ self.weights[-1] + self.biases[-1]
        return np.array([np.tanh(raw[0]), np.tanh(raw[1]), 1.0 / (1.0 + np.exp(-raw[2]))], dtype=np.float32)

    def act(self, lidar: np.ndarray, speed: float) -> tuple[float, float, bool]:
        steering, throttle, handbrake_probability = self.predict(make_features(lidar, speed))
        return float(steering), float(throttle), bool(handbrake_probability >= 0.5)
