from __future__ import annotations

import pygame

from racing.simulator import HEIGHT, WIDTH


class MainMenu:
    """Small launcher that exposes the two user-facing game modes."""

    def __init__(self, has_policy: bool) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simple ANN Racing")
        self.clock = pygame.time.Clock()
        self.title_font, self.font, self.small_font = pygame.font.Font(None, 64), pygame.font.Font(None, 34), pygame.font.Font(None, 26)
        self.has_policy, self.page = has_policy, "mode"

    def text(self, value: str, y: int, font: pygame.font.Font, color: tuple[int, int, int] = (245, 245, 245)) -> None:
        surface = font.render(value, True, color)
        self.screen.blit(surface, surface.get_rect(center=(WIDTH // 2, y)))

    def button(self, rect: pygame.Rect, label: str, selected: bool = False, disabled: bool = False) -> None:
        pygame.draw.rect(self.screen, (70, 70, 70) if disabled else ((69, 109, 145) if selected else (58, 65, 76)), rect, border_radius=10)
        pygame.draw.rect(self.screen, (245, 200, 74) if selected else (155, 165, 175), rect, 2, border_radius=10)
        self.text(label, rect.centery, self.font, (145, 145, 145) if disabled else (255, 255, 255))

    def draw(self) -> tuple[pygame.Rect, pygame.Rect]:
        self.screen.fill((27, 36, 48))
        self.text("Simple ANN Racing", 130, self.title_font)
        first, second = pygame.Rect(WIDTH // 2 - 235, 280, 470, 75), pygame.Rect(WIDTH // 2 - 235, 385, 470, 75)
        if self.page == "mode":
            self.text("Choose a mode", 215, self.font)
            self.button(first, "1  Sandbox: build and test a track")
            self.button(second, "2  Race: player or neural network")
            self.text("Esc — quit", 680, self.small_font, (190, 200, 210))
        else:
            self.text("Race — choose the driver", 215, self.font)
            self.button(first, "Player (default)", selected=True)
            self.button(second, "Neural network", disabled=not self.has_policy)
            note = "Enter/P — player | N — neural | Esc — back" if self.has_policy else "Train first: python train.py (no models/policy.npz yet)"
            self.text(note, 530, self.small_font, (190, 200, 210) if self.has_policy else (255, 220, 112))
        return first, second

    def run(self) -> tuple[str, str | None] | None:
        while True:
            first, second = self.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.page == "race": self.page = "mode"
                        else: pygame.quit(); return None
                    elif self.page == "mode" and event.key == pygame.K_1:
                        pygame.quit(); return "sandbox", None
                    elif self.page == "mode" and event.key == pygame.K_2:
                        self.page = "race"
                    elif self.page == "race" and (event.key in (pygame.K_RETURN, pygame.K_p) or event.unicode.lower() == "з"):
                        pygame.quit(); return "race", "player"
                    elif self.page == "race" and (event.key == pygame.K_n or event.unicode.lower() == "т") and self.has_policy:
                        pygame.quit(); return "race", "ai"
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.page == "mode" and first.collidepoint(event.pos):
                        pygame.quit(); return "sandbox", None
                    if self.page == "mode" and second.collidepoint(event.pos): self.page = "race"
                    elif self.page == "race" and first.collidepoint(event.pos):
                        pygame.quit(); return "race", "player"
                    elif self.page == "race" and second.collidepoint(event.pos) and self.has_policy:
                        pygame.quit(); return "race", "ai"
            pygame.display.flip()
            self.clock.tick(60)
