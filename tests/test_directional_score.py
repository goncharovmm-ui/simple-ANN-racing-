import os
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from racing.simulator import RacingGame, Track


class DirectionalScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.track = Track.load(Path("maps/default.json"))
        cls.screen = pygame.Surface((1280, 720))

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_clockwise_motion_is_positive_and_reverse_is_penalized(self):
        game = RacingGame(self.track, record=False, screen=self.screen)
        start = self.track.start.copy()
        route_next = self.track.checkpoints[0]
        direction = route_next - start
        direction.scale_to_length(20.0)

        game.course_distance = self.track.course_position(start)
        game.course_progress = game.course_distance / self.track.course_length()
        game.directional_score = 0.0
        game.update_directional_score(start, start + direction)
        forward_score = game.directional_score

        game.update_directional_score(start + direction, start)

        self.assertGreater(forward_score, 0.0)
        self.assertLess(game.directional_score, forward_score)
        self.assertAlmostEqual(game.directional_score, -10.0, places=5)

    def test_route_progress_is_monotonic_for_clockwise_step(self):
        start = self.track.start.copy()
        route_next = self.track.checkpoints[0]
        direction = route_next - start
        direction.scale_to_length(20.0)
        before = self.track.course_position(start)
        after = self.track.course_position(start + direction)
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
