"""Small interpretable controllers used as non-neural baselines."""
from __future__ import annotations

import numpy as np


class LidarCorrelationPolicy:
    """Turn away from the nearer side using only lidar and current speed."""

    def __init__(self, steering_gain: float = 2.2, throttle_gain: float = 1.35, turn_drag: float = 0.45) -> None:
        self.steering_gain = steering_gain
        self.throttle_gain = throttle_gain
        self.turn_drag = turn_drag

    def act(self, lidar: np.ndarray, speed: float) -> tuple[float, float, bool]:
        values = np.asarray(lidar, dtype=np.float32)
        left_clearance = float(np.mean(values[:7]))
        right_clearance = float(np.mean(values[8:]))
        side_difference = right_clearance - left_clearance
        steering = float(np.clip(self.steering_gain * side_difference, -1.0, 1.0))
        front_clearance = float(np.mean(values[6:9]))
        throttle = float(np.clip(self.throttle_gain * front_clearance - self.turn_drag * abs(side_difference), -1.0, 1.0))
        handbrake = abs(steering) > 0.72 and speed > 150.0
        return steering, throttle, handbrake
