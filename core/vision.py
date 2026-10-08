"""Робота з MediaPipe Hands та координатами курсора."""

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from config.config import Settings
from core.gesture import finger_pattern


@dataclass
class HandObservation:
    """Дані руки, необхідні для жестів та відображення."""

    gesture: str | None = None
    slide_point: np.ndarray | None = None
    camera_points: np.ndarray | None = None
    camera_tip: np.ndarray | None = None


class HandVision:
    """Знаходити руку у відеокадрі та визначати її жест і позицію."""

    def __init__(self, settings: Settings) -> None:
        """Створити модель MediaPipe із параметрами конфігурації."""

        self.settings = settings
        self.hands_module = mp.solutions.hands
        self.detector = self.hands_module.Hands(
            max_num_hands=settings.vision.max_num_hands,
            min_detection_confidence=settings.vision.detection_confidence,
            min_tracking_confidence=settings.vision.tracking_confidence,
        )
        self.previous_point: np.ndarray | None = None

    def __enter__(self) -> "HandVision":
        """Дозволити використання детектора в контекстному менеджері."""

        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        """Звільнити ресурси моделі."""

        self.detector.close()

    def observe(self, frame: np.ndarray, slide_size: tuple[int, int]) -> HandObservation:
        """Визначити жест, позицію на слайді та точки руки для прев'ю."""

        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)
        if not result.multi_hand_landmarks:
            self.previous_point = None
            return HandObservation()

        landmarks = result.multi_hand_landmarks[0].landmark
        points = np.array([(p.x * frame_w, p.y * frame_h) for p in landmarks], dtype=float)
        pattern = finger_pattern(points, self.settings.vision)
        gesture = self.settings.gestures.patterns.get(pattern)

        pointer = self.settings.pointer
        normalized = points[pointer.landmark_index] / np.array([frame_w, frame_h])
        target = np.clip((normalized - pointer.crop_offset) / pointer.crop_width, 0, 1)
        target *= np.array([slide_size[0] - 1, slide_size[1] - 1])

        if self.previous_point is None:
            slide_point = target
        else:
            slide_point = self.previous_point * (1 - pointer.smoothing) + target * pointer.smoothing
        self.previous_point = slide_point

        preview_points = points * (self.settings.camera.preview_width / frame_w)
        return HandObservation(
            gesture=gesture,
            slide_point=slide_point,
            camera_points=preview_points,
            camera_tip=preview_points[pointer.landmark_index],
        )
