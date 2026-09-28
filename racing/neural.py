"""Small dependency-free neural policy used by the simulator and training script."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def make_features(lidar: np.ndarray, speed: float) -> np.ndarray:
    """The observation contract shared by collection, training and autonomous driving."""
    return np.concatenate((np.asarray(lidar, dtype=np.float32), [np.clip(speed / 360.0, -1.0, 1.0)])).astype(np.float32)


@dataclass
class NeuralPolicy:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "NeuralPolicy":
        with np.load(path) as data:
            return cls(data["w1"], data["b1"], data["w2"], data["b2"])

    def predict(self, features: np.ndarray) -> np.ndarray:
        hidden = np.maximum(features @ self.w1 + self.b1, 0.0)
        raw = hidden @ self.w2 + self.b2
        return np.array([np.tanh(raw[0]), np.tanh(raw[1]), 1.0 / (1.0 + np.exp(-raw[2]))], dtype=np.float32)

    def act(self, lidar: np.ndarray, speed: float) -> tuple[float, float, bool]:
        steering, throttle, handbrake_probability = self.predict(make_features(lidar, speed))
        return float(steering), float(throttle), bool(handbrake_probability >= 0.5)
