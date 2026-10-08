"""Розпізнавання конфігурації пальців та стабілізація жестів."""

import numpy as np

from config.config import GestureSettings, VisionSettings


def finger_pattern(points: np.ndarray, settings: VisionSettings) -> tuple[int, ...]:
    """Повернути п'ять бітів: великий, вказівний, середній, безіменний, мізинець."""

    def distance(a: int, b: int) -> float:
        """Обчислити евклідову відстань між двома ключовими точками руки."""

        return float(np.linalg.norm(points[a] - points[b]))

    thumb = distance(settings.thumb_tip, settings.thumb_base) > (
        settings.thumb_open_ratio
        * distance(settings.thumb_reference_start, settings.thumb_reference_end)
    )
    fingers = tuple(
        distance(tip, settings.wrist_index)
        > distance(tip - settings.finger_joint_offset, settings.wrist_index)
        for tip in settings.finger_tips
    )
    return tuple(int(value) for value in (thumb, *fingers))


class GestureTracker:
    """Фільтрувати випадкові зміни жестів і виконувати дії один раз за утримання."""

    def __init__(self, settings: GestureSettings) -> None:
        """Ініціалізувати стан визначення жестів."""

        self.settings = settings
        self.held: str | None = None
        self.pending: str | None = None
        self.since = 0.0
        self.changed = 0.0
        self.fired = False
        self.progress = 0.0

    def update(self, gesture: str | None, now: float) -> bool:
        """Оновити стан і повернути True, якщо підтверджений жест змінився."""

        previous = self.held
        if gesture == self.held:
            self.pending = gesture
        elif gesture != self.pending:
            self.pending = gesture
            self.changed = now
        elif now - self.changed >= self.settings.debounce_seconds:
            self.held = gesture
            self.since = self.changed
            self.fired = False

        self.progress = (
            min(max((now - self.since) / self.settings.hold_seconds, 0.0), 1.0)
            if self.held is not None
            else 0.0
        )
        return self.held != previous

    @property
    def ready(self) -> bool:
        """Чи утримують розпізнаний жест достатньо довго?"""

        return self.held is not None and self.progress >= 1.0

    def consume_action(self) -> str | None:
        """Видати жест одноразово після завершення утримання."""

        if not self.ready or self.fired:
            return None
        self.fired = True
        return self.held
