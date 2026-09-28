from __future__ import annotations

from pathlib import Path

import pygame

from racing.expert import ArtificialExpert
from racing.neural import NeuralPolicy
from racing.simulator import HEIGHT, WIDTH, RacingGame, Track, draw_track


class ParallelDaggerBatch:
    """Runs many independent DAgger rollouts at accelerated simulation speed."""

    def __init__(self, track: Track, policy: NeuralPolicy, map_path: Path, agents: int = 20, speed: int = 8, display: bool = True, game_mode: str = "dagger", label_all: bool = False, expert_controls: bool = False, max_steps: int = 3600) -> None:
        pygame.init()
        self.display = display
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT)) if display else pygame.Surface((WIDTH, HEIGHT))
        if display:
            pygame.display.set_caption("Simple ANN Racing — parallel DAgger")
        self.clock, self.font = pygame.time.Clock(), pygame.font.Font(None, 27)
        self.track, self.speed = track, max(1, speed)
        self.max_steps = max_steps
        self.games = [RacingGame(track, record=True, screen=pygame.Surface((WIDTH, HEIGHT)), policy=policy,
                                 game_mode=game_mode, map_path=map_path, expert=ArtificialExpert(label_all=label_all), expert_controls=expert_controls) for _ in range(agents)]
        self.done = [False] * agents

    def step_game(self, game: RacingGame) -> bool:
        steering, throttle, handbrake, lidar = game.update(1 / 60)
        if game.writer and game.dagger_label:
            game.writer.write(game, *game.dagger_label, lidar)
        game.step += 1
        if game.step >= self.max_steps:
            game.game_over = True
            game.message = "Episode time limit reached"
        if game.game_over or game.completed_laps > 0:
            game.finish_dagger_episode()
            if game.writer:
                game.writer.close()
            return True
        return False

    def draw(self) -> None:
        if not self.display:
            return
        draw_track(self.screen, self.track)
        finished = sum(self.done)
        successful = sum(game.successful_episodes for game in self.games)
        for game, done in zip(self.games, self.done):
            if done and game.trajectory_history:
                points, color = game.trajectory_history[-1]
                if len(points) > 1:
                    pygame.draw.lines(self.screen, color, False, points, 3)
            elif len(game.trajectory) > 1:
                pygame.draw.lines(self.screen, (236, 180, 62), False, game.trajectory, 1)
                pygame.draw.circle(self.screen, (82, 171, 247), game.car.position, 5)
        lines = [f"Parallel DAgger: {finished}/{len(self.games)} attempts complete | finished laps: {successful}",
                 f"Simulation speed: {self.speed}x | 20 cars run in parallel | labels are saved live",
                 "Green = better completed run; red = failed/damaged run; Esc = stop batch"]
        for index, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, (250, 250, 250)), (20, 18 + index * 30))
        pygame.display.flip()

    def run(self) -> None:
        running = True
        try:
            while running and not all(self.done):
                if self.display:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            running = False
                for _ in range(self.speed):
                    for index, game in enumerate(self.games):
                        if not self.done[index] and self.step_game(game):
                            self.done[index] = True
                self.draw()
                if self.display:
                    self.clock.tick(60)
        finally:
            for game, done in zip(self.games, self.done):
                if game.writer:
                    game.writer.close()
            pygame.quit()
