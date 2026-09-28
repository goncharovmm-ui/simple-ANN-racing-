"""Interactive training-center UI for the racing simulator."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pygame

from racing.simulator import HEIGHT, WIDTH, Track


ROOT = Path(__file__).resolve().parent.parent


class TrainingCenter:
    """Launch and monitor a headless DAgger run without blocking the Pygame UI."""

    def __init__(self, map_path: Path, model_path: Path | None = None) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simple ANN Racing — training center")
        self.clock = pygame.time.Clock()
        self.title_font, self.font, self.small_font = pygame.font.Font(None, 48), pygame.font.Font(None, 29), pygame.font.Font(None, 23)
        self.maps = sorted(Path("maps").glob("*.json")) or [map_path]
        self.map_index = next((i for i, path in enumerate(self.maps) if path.resolve() == map_path.resolve()), 0)
        self.models = self.available_models()
        self.model_index = next((i for i, path in enumerate(self.models) if model_path and path.resolve() == model_path.resolve()), 0)
        self.process: subprocess.Popen[str] | None = None
        self.video_process: subprocess.Popen[str] | None = None
        self.history_path: Path | None = None
        self.video_path: Path | None = None
        self.run_name = ""
        self.lines: list[str] = []
        self.epoch, self.total_epochs = 0, 10
        self.status = "Готово к обучению"
        self.started_at: float | None = None
        self.estimate_seconds = self.estimate_from_previous_runs()

    @staticmethod
    def available_models() -> list[Path]:
        paths = sorted(Path("models").rglob("*.npz"))
        desktop = Path.home() / "Desktop"
        if desktop.exists():
            paths.extend(sorted(desktop.glob("*.npz")))
        return paths or []

    @property
    def map_path(self) -> Path:
        return self.maps[self.map_index]

    @property
    def selected_model(self) -> Path | None:
        return self.models[self.model_index] if self.models else None

    def text(self, value: str, position: tuple[int, int], font: pygame.font.Font, color: tuple[int, int, int] = (240, 240, 240)) -> None:
        self.screen.blit(font.render(value, True, color), position)

    def button(self, rect: pygame.Rect, label: str, disabled: bool = False) -> None:
        pygame.draw.rect(self.screen, (65, 70, 80) if not disabled else (45, 48, 54), rect, border_radius=8)
        pygame.draw.rect(self.screen, (226, 192, 71) if not disabled else (105, 108, 112), rect, 2, border_radius=8)
        surface = self.font.render(label, True, (250, 250, 250) if not disabled else (135, 135, 135))
        self.screen.blit(surface, surface.get_rect(center=rect.center))

    def best_human_demo(self) -> Path | None:
        """Return the fastest completed player CSV recorded on the selected map."""
        try:
            track_id = f"{self.map_path.stem}-{Track.load(self.map_path).fingerprint()}"
        except (OSError, ValueError, KeyError):
            return None
        candidates: list[tuple[float, Path]] = []
        for path in Path("data/demos/race").rglob("*.csv") if Path("data/demos/race").exists() else []:
            try:
                with path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if not rows or rows[-1].get("track_id") != track_id or rows[-1].get("driver") != "player":
                    continue
                lap_values = [float(row.get("lap_s") or 0) for row in rows]
                peaks: list[float] = []
                current_peak = 0.0
                previous = 0.0
                for value in lap_values:
                    if value > 0 and previous > value + 0.1 and current_peak > 0:
                        peaks.append(current_peak)
                        current_peak = 0.0
                    current_peak = max(current_peak, value)
                    previous = value
                if current_peak > 0:
                    peaks.append(current_peak)
                if peaks:
                    candidates.append((min(peaks), path))
            except (OSError, ValueError):
                continue
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    @staticmethod
    def estimate_from_previous_runs() -> float:
        path = Path("artifacts/training-timings.json")
        if not path.exists():
            return 60.0 * 10
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            per_epoch = [float(item["seconds"]) / max(1, int(item["epochs"])) for item in values[-5:]]
            return max(30.0, sum(per_epoch) / len(per_epoch) * 10) if per_epoch else 600.0
        except (OSError, ValueError, KeyError, TypeError):
            return 600.0

    def record_timing(self) -> None:
        if self.started_at is None:
            return
        path = Path("artifacts/training-timings.json")
        try:
            values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except (OSError, ValueError):
            values = []
        values.append({"run": self.run_name, "epochs": self.total_epochs, "seconds": round(time.monotonic() - self.started_at, 1)})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values[-10:], ensure_ascii=False, indent=2), encoding="utf-8")

    def start_training(self) -> None:
        if self.process and self.process.poll() is None:
            return
        seed = self.best_human_demo()
        self.run_name = f"ui-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.history_path = Path("artifacts") / f"training-history-{self.run_name}.json"
        command = [sys.executable, "run_training_cycle.py", "--epochs", str(self.total_epochs), "--agents", "20", "--eval-agents", "20", "--train-epochs", "20", "--run-name", self.run_name, "--output", str(self.history_path)]
        if seed:
            command.extend(["--seed-data", str(seed)])
            self.status = f"Выбрана лучшая траектория: {seed.name}"
        else:
            self.status = "Полный человеческий круг не найден — запуск с экспертного старта"
        if self.selected_model and self.selected_model.exists():
            command.extend(["--base-model", str(self.selected_model)])
        self.process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.started_at, self.epoch, self.lines = time.monotonic(), 0, []
        self.status = "Обучение запущено"

    def start_video(self) -> None:
        if not self.history_path or not self.history_path.exists() or (self.video_process and self.video_process.poll() is None):
            return
        desktop = Path.home() / "Desktop"
        desktop.mkdir(parents=True, exist_ok=True)
        self.video_path = desktop / f"simple-ann-{self.run_name}.mp4"
        self.video_process = subprocess.Popen([sys.executable, "make_training_movie.py", "--history", str(self.history_path), "--output", str(self.video_path), "--duration", "300"], cwd=ROOT)
        self.status = f"Собираю видео: {self.video_path}"

    def export_model(self) -> None:
        model = self.selected_model
        if self.history_path and self.history_path.exists():
            try:
                history = json.loads(self.history_path.read_text(encoding="utf-8"))
                selected = history["epochs"][-1].get("selected_policy")
                if selected and Path(selected).exists():
                    model = Path(selected)
            except (OSError, ValueError, KeyError, IndexError):
                pass
        if not model or not model.exists():
            self.status = "Нет модели для сохранения"
            return
        desktop = Path.home() / "Desktop"
        desktop.mkdir(parents=True, exist_ok=True)
        destination = desktop / f"simple-ann-{model.stem}.npz"
        shutil.copy2(model, destination)
        self.status = f"Модель сохранена: {destination}"

    def open_video(self) -> None:
        if not self.video_path or not self.video_path.exists():
            self.status = "Видео ещё не готово"
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(self.video_path)])
        elif sys.platform.startswith("win"):
            os.startfile(self.video_path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(self.video_path)])

    def poll_processes(self) -> None:
        if self.process and self.process.stdout:
            try:
                text = self.process.stdout.read()
            except (BlockingIOError, OSError):
                text = ""
            if text:
                self.lines.extend(text.splitlines())
                self.lines = self.lines[-5:]
                match = re.search(r"Epoch (\d+)/(\d+)", text)
                if match:
                    self.epoch, self.total_epochs = int(match.group(1)), int(match.group(2))
                    self.status = f"Эпоха {self.epoch}/{self.total_epochs}: обучение и автономная проверка"
            if self.process.poll() is not None:
                self.record_timing()
                code = self.process.returncode
                self.process = None
                if code == 0:
                    self.status = "Обучение завершено — видео можно собрать"
                    self.models = self.available_models()
                    self.model_index = max(0, len(self.models) - 1)
                    self.start_video()
                else:
                    self.status = f"Обучение завершилось с ошибкой ({code})"
        if self.video_process and self.video_process.poll() is not None:
            if self.video_process.returncode == 0:
                self.status = f"Видео готово: {self.video_path}"
            else:
                self.status = "Не удалось собрать видео"
            self.video_process = None

    def draw(self) -> None:
        self.screen.fill((27, 36, 48))
        self.text("Центр обучения", (48, 38), self.title_font)
        self.text("Сначала сохраните карту и завершите хотя бы один круг игроком.", (52, 105), self.small_font, (190, 200, 210))
        self.text(f"Карта: {self.map_path}", (52, 155), self.font)
        seed = self.best_human_demo()
        self.text(f"Лучшая траектория: {seed.name if seed else 'не найдена'}", (52, 190), self.small_font, (120, 230, 150) if seed else (255, 205, 100))
        self.text(f"Базовая модель: {self.selected_model if self.selected_model else 'нет — будет создана новая'}", (52, 225), self.small_font)
        bar = pygame.Rect(52, 270, 1170, 30)
        pygame.draw.rect(self.screen, (50, 56, 66), bar, border_radius=7)
        progress = 1.0 if self.video_path and self.video_path.exists() else (self.epoch / max(1, self.total_epochs))
        pygame.draw.rect(self.screen, (70, 178, 104), pygame.Rect(bar.x, bar.y, round(bar.width * progress), bar.height), border_radius=7)
        pygame.draw.rect(self.screen, (180, 190, 200), bar, 2, border_radius=7)
        self.text(f"Прогресс: {round(progress * 100)}%   {self.status}", (52, 320), self.font, (235, 240, 245))
        elapsed = 0 if self.started_at is None else time.monotonic() - self.started_at
        remaining = max(0, self.estimate_seconds - elapsed) if self.process else self.estimate_seconds
        self.text(f"Прошло: {int(elapsed)//60:02d}:{int(elapsed)%60:02d}   Примерно осталось: {int(remaining)//60:02d}:{int(remaining)%60:02d} (оценка по прошлым запускам)", (52, 355), self.small_font, (190, 200, 210))
        self.button(pygame.Rect(52, 415, 250, 52), "Другая карта (M)", bool(self.process))
        self.button(pygame.Rect(320, 415, 250, 52), "Другая модель (L)", bool(self.process))
        self.button(pygame.Rect(588, 415, 300, 52), "Начать обучение (T)", bool(self.process))
        self.button(pygame.Rect(906, 415, 300, 52), "Сохранить модель (S)", False)
        self.button(pygame.Rect(52, 485, 350, 52), "Собрать/обновить видео (V)", False)
        self.button(pygame.Rect(420, 485, 350, 52), "Открыть видео (Enter)", False)
        self.text("Последние сообщения:", (52, 585), self.small_font, (190, 200, 210))
        for index, line in enumerate(self.lines):
            self.text(line[-150:], (52, 612 + index * 25), self.small_font, (220, 225, 230))
        self.text("Esc — вернуться в меню", (52, 755), self.small_font, (190, 200, 210))

    def run(self) -> None:
        running = True
        try:
            while running:
                self.poll_processes()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_t:
                            self.start_training()
                        elif event.key == pygame.K_m and not self.process:
                            self.map_index = (self.map_index + 1) % len(self.maps)
                        elif event.key == pygame.K_l and not self.process and self.models:
                            self.model_index = (self.model_index + 1) % len(self.models)
                        elif event.key == pygame.K_s:
                            self.export_model()
                        elif event.key == pygame.K_v:
                            self.start_video()
                        elif event.key == pygame.K_RETURN:
                            self.open_video()
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        x, y = event.pos
                        if 415 <= y <= 467 and x < 302 and not self.process:
                            self.map_index = (self.map_index + 1) % len(self.maps)
                        elif 415 <= y <= 467 and 320 <= x < 570 and not self.process and self.models:
                            self.model_index = (self.model_index + 1) % len(self.models)
                        elif 415 <= y <= 467 and 588 <= x < 888:
                            self.start_training()
                        elif 415 <= y <= 467 and x >= 906:
                            self.export_model()
                        elif 485 <= y <= 537 and x < 410:
                            self.start_video()
                        elif 485 <= y <= 537 and 420 <= x < 790:
                            self.open_video()
                self.draw()
                pygame.display.flip()
                self.clock.tick(30)
        finally:
            pygame.quit()
