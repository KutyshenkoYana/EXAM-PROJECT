"""Малювання скелета руки, індикатора утримання та палітри."""

import cv2
import mediapipe as mp
import numpy as np

from config.config import DisplaySettings


def draw_hand(image: np.ndarray, points: np.ndarray, settings: DisplaySettings) -> None:
    """Намалювати з'єднання й точки кисті поверх прев'ю камери."""

    integer_points = points.astype(int)
    for start, end in mp.solutions.hands.HAND_CONNECTIONS:
        cv2.line(image, tuple(integer_points[start]), tuple(integer_points[end]),
                 settings.skeleton_color, settings.skeleton_thickness, cv2.LINE_AA)
    for point in integer_points:
        center = tuple(point)
        cv2.circle(image, center, settings.landmark_outer_radius,
                   settings.skeleton_color, -1, cv2.LINE_AA)
        cv2.circle(image, center, settings.landmark_inner_radius,
                   settings.landmark_inner_color, -1, cv2.LINE_AA)


def draw_status(
    image: np.ndarray,
    tip: np.ndarray | None,
    active_color: tuple[int, int, int],
    palette: tuple[tuple[int, int, int], ...],
    progress: float,
    settings: DisplaySettings,
) -> None:
    """Відобразити прогрес активації жесту й доступні кольори малювання."""

    if tip is not None:
        center = tuple(tip.astype(int))
        cv2.circle(image, center, settings.status_radius, settings.skeleton_color,
                   settings.status_border_thickness, cv2.LINE_AA)
        cv2.ellipse(image, center, (settings.status_radius, settings.status_radius),
                    -90, 0, 360 * progress, active_color,
                    settings.status_progress_thickness, cv2.LINE_AA)

    y = image.shape[0] - settings.palette_bottom_margin
    for index, color in enumerate(palette):
        active = color == active_color
        center = (settings.palette_x + index * settings.palette_spacing, y)
        outer = settings.palette_active_radius if active else settings.palette_inactive_radius
        inner = settings.palette_inner_active_radius if active else settings.palette_inner_inactive_radius
        cv2.circle(image, center, outer, settings.skeleton_color, -1, cv2.LINE_AA)
        cv2.circle(image, center, inner, color, -1, cv2.LINE_AA)
