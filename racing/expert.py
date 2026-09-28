from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from racing.simulator import RacingGame


class ArtificialExpert:
    """Rule-based safety teacher used to label the learner's risky rollouts for DAgger."""

    def act(self, game: "RacingGame", lidar: np.ndarray) -> tuple[float, float, bool]:
        target = game.track.checkpoints[game.next_checkpoint]
        vector = target - game.car.position
        desired = math.degrees(math.atan2(vector.y, vector.x))
        delta = (desired - game.car.angle + 180) % 360 - 180
        steering = delta / 42.0
        for angle, distance in zip(np.linspace(-110, 110, len(lidar)), lidar):
            if distance < 0.55:
                steering -= math.copysign((0.55 - float(distance)) * 1.7, float(angle) or 1.0)
        steering = max(-1.0, min(1.0, steering))
        forward_clearance = float(np.min(lidar[len(lidar)//2 - 1:len(lidar)//2 + 2]))
        sharp_turn = abs(delta) > 48 or forward_clearance < 0.38
        throttle = 0.22 if sharp_turn else 1.0
        handbrake = bool(abs(delta) > 68 and game.car.speed > 105)
        return steering, throttle, handbrake

    def should_label(self, lidar: np.ndarray, learner_action: tuple[float, float, bool]) -> bool:
        steering, throttle, handbrake = learner_action
        dangerous = float(np.min(lidar)) < 0.38
        differs = abs(steering) > 0.55 or throttle > 0.65 or handbrake
        return dangerous or differs
