from __future__ import annotations

from pathlib import Path

import pygame

from racing.simulator import HEIGHT, WIDTH


class MainMenu:
    """Small launcher that exposes the two user-facing game modes."""

    def __init__(self, policy_paths: list[Path] | None = None) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simple ANN Racing")
        self.clock = pygame.time.Clock()
        self.title_font, self.font, self.small_font = pygame.font.Font(None, 64), pygame.font.Font(None, 34), pygame.font.Font(None, 26)
        self.policy_paths, self.policy_index, self.page = policy_paths or [], 0, "mode"

    def text(self, value: str, y: int, font: pygame.font.Font, color: tuple[int, int, int] = (245, 245, 245)) -> None:
        surface = font.render(value, True, color)
        self.screen.blit(surface, surface.get_rect(center=(WIDTH // 2, y)))

    def button(self, rect: pygame.Rect, label: str, selected: bool = False, disabled: bool = False) -> None:
        pygame.draw.rect(self.screen, (70, 70, 70) if disabled else ((69, 109, 145) if selected else (58, 65, 76)), rect, border_radius=10)
        pygame.draw.rect(self.screen, (245, 200, 74) if selected else (155, 165, 175), rect, 2, border_radius=10)
        self.text(label, rect.centery, self.font, (145, 145, 145) if disabled else (255, 255, 255))

    def draw(self) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
        self.screen.fill((27, 36, 48))
        self.text("Simple ANN Racing", 130, self.title_font)
        first = pygame.Rect(WIDTH // 2 - 235, 255, 470, 65)
        second = pygame.Rect(WIDTH // 2 - 235, 345, 470, 65)
        third = pygame.Rect(WIDTH // 2 - 235, 435, 470, 65)
        if self.page == "mode":
            self.text("Choose a mode", 215, self.font)
            self.button(first, "1  Sandbox: build and test a track")
            self.button(second, "2  Race: player or neural network")
            self.button(third, "3  Training center")
            self.text("Esc — quit", 680, self.small_font, (190, 200, 210))
        else:
            self.text("Race — choose the driver", 215, self.font)
            self.button(first, "Player (default)", selected=True)
            self.button(second, "Neural network", disabled=not self.policy_paths)
            selected = self.policy_paths[self.policy_index].as_posix() if self.policy_paths else "нет сохранённых моделей"
            self.text(f"Модель: {selected}", 515, self.small_font, (190, 200, 210) if self.policy_paths else (255, 220, 112))
            self.text("Enter/P — игрок | N — нейросеть | ←/→ — выбрать модель | Esc — назад", 550, self.small_font, (190, 200, 210))
        return first, second, third

    def run(self) -> tuple[str, str | None, Path | None] | None:
        while True:
            first, second, third = self.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.page == "race": self.page = "mode"
                        else: pygame.quit(); return None
                    elif self.page == "mode" and event.key == pygame.K_1:
                        pygame.quit(); return "sandbox", None, None
                    elif self.page == "mode" and event.key == pygame.K_2:
                        self.page = "race"
                    elif self.page == "mode" and event.key == pygame.K_3:
                        pygame.quit(); return "training", None, None
                    elif self.page == "race" and (event.key in (pygame.K_RETURN, pygame.K_p) or event.unicode.lower() == "з"):
                        pygame.quit(); return "race", "player", None
                    elif self.page == "race" and event.key == pygame.K_RIGHT and self.policy_paths:
                        self.policy_index = (self.policy_index + 1) % len(self.policy_paths)
                    elif self.page == "race" and event.key == pygame.K_LEFT and self.policy_paths:
                        self.policy_index = (self.policy_index - 1) % len(self.policy_paths)
                    elif self.page == "race" and (event.key == pygame.K_n or event.unicode.lower() == "т") and self.policy_paths:
                        pygame.quit(); return "race", "ai", self.policy_paths[self.policy_index]
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.page == "mode" and first.collidepoint(event.pos):
                        pygame.quit(); return "sandbox", None, None
                    if self.page == "mode" and second.collidepoint(event.pos): self.page = "race"
                    elif self.page == "mode" and third.collidepoint(event.pos):
                        pygame.quit(); return "training", None, None
                    elif self.page == "race" and first.collidepoint(event.pos):
                        pygame.quit(); return "race", "player", None
                    elif self.page == "race" and second.collidepoint(event.pos) and self.policy_paths:
                        pygame.quit(); return "race", "ai", self.policy_paths[self.policy_index]
            pygame.display.flip()
            self.clock.tick(60)
