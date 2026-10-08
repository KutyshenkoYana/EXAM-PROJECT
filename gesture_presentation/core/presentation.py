"""Завантаження слайдів, навігація та малювання поверх них."""

from pathlib import Path

import cv2
import numpy as np

from config.config import Settings


def load_slide(path: Path, scale: float) -> np.ndarray:
    """Завантажити файл PNG, підтримуючи шляхи з Unicode-символами."""

    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не вдалося відкрити зображення: {path}")
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


class Presentation:
    """Зберігати слайди та окремий прозорий шар малювання для кожного."""

    def __init__(self, settings: Settings) -> None:
        """Завантажити слайди та підготувати полотна."""

        self.settings = settings
        paths = sorted(settings.slides.directory.glob(settings.slides.pattern))
        if not paths:
            raise FileNotFoundError(
                f"У папці '{settings.slides.directory}' немає файлів "
                f"'{settings.slides.pattern}'. Додайте слайди перед запуском."
            )

        self.slides = [load_slide(path, settings.slides.scale) for path in paths]
        height, width = self.slides[0].shape[:2]
        # Слайди з різними вихідними розмірами приводимо до одного розміру.
        self.slides = [
            img if img.shape[:2] == (height, width) else cv2.resize(img, (width, height))
            for img in self.slides
        ]
        self.canvases = [np.zeros_like(img) for img in self.slides]
        self.width = width
        self.height = height
        self.index = -1
        self.color_index = 0
        self.last_point: tuple[int, int] | None = None

    @property
    def current_color(self) -> tuple[int, int, int]:
        """Активний колір олівця в BGR."""

        return self.settings.drawing.colors[self.color_index]

    def reset_stroke(self) -> None:
        """Завершити поточну безперервну лінію."""

        self.last_point = None

    def handle_gesture(self, gesture: str) -> None:
        """Виконати одноразову команду керування презентацією."""

        if gesture == "start":
            self.index = max(self.index, 0)
            self.canvases[self.index][:] = 0
        elif gesture == "next" and self.index >= 0:
            self.index = min(self.index + 1, len(self.slides) - 1)
        elif gesture == "prev" and self.index >= 0:
            self.index = max(self.index - 1, 0)
        elif gesture == "color":
            self.color_index = (self.color_index + 1) % len(self.settings.drawing.colors)
        self.reset_stroke()

    def render(self, point: np.ndarray | None, gesture: str | None, ready: bool) -> np.ndarray:
        """Зібрати кадр слайду, штрихи й контур курсора в одне зображення."""

        background = self.settings.display.background_value
        stage = np.full((self.height, self.width, 3), background, dtype=np.uint8)
        if self.index < 0:
            self.reset_stroke()
            return stage

        canvas = self.canvases[self.index]
        current = tuple(point.astype(int)) if point is not None else None
        if ready and current is not None and gesture in ("draw", "erase"):
            draw_settings = self.settings.drawing
            color, size = (
                (self.current_color, draw_settings.brush_size)
                if gesture == "draw"
                else (draw_settings.erased_color, draw_settings.eraser_radius * draw_settings.eraser_stroke_multiplier)
            )
            cv2.line(canvas, self.last_point or current, current, color, size)
            self.last_point = current
        else:
            self.reset_stroke()

        mask = canvas.any(axis=2)
        stage = self.slides[self.index].copy()
        stage[mask] = canvas[mask]
        if current is not None:
            radius = (
                self.settings.drawing.eraser_radius
                if gesture == "erase"
                else self.settings.drawing.brush_size + self.settings.drawing.cursor_radius_extra
            )
            cv2.circle(stage, current, radius, self.current_color,
                       self.settings.drawing.cursor_line_width, cv2.LINE_AA)
        return stage
