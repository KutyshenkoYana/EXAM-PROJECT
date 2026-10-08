"""Юніт-тести налаштувань, жестів і відображення без вебкамери."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from config import load_config
from core.gesture import GestureTracker, finger_pattern
from core.presentation import Presentation


class ConfigTests(unittest.TestCase):
    """Перевірки перетворення YAML-налаштувань."""

    def test_default_config(self):
        """Базові шість жестів і параметри є в конфігурації."""

        config = load_config()
        self.assertEqual(config.gestures.patterns[(0, 1, 1, 0, 0)], "draw")
        self.assertEqual(config.vision.finger_tips, (8, 12, 16, 20))
        self.assertEqual(len(config.drawing.colors), 3)


class GestureTests(unittest.TestCase):
    """Перевірки таймера розпізнавання та пальців."""

    def test_hold_and_single_fire(self):
        """Команда має спрацювати один раз після утримання."""

        tracker = GestureTracker(load_config().gestures)
        self.assertFalse(tracker.update("next", 10.0))
        self.assertTrue(tracker.update("next", 10.21))
        self.assertFalse(tracker.ready)
        tracker.update("next", 11.0)
        self.assertTrue(tracker.ready)
        self.assertEqual(tracker.consume_action(), "next")
        self.assertIsNone(tracker.consume_action())
        self.assertFalse(tracker.update(None, 11.1))
        self.assertTrue(tracker.update(None, 11.31))
        self.assertFalse(tracker.ready)

    def test_finger_pattern_shape(self):
        """Функція завжди повертає п'ять значень 0/1."""

        points = np.zeros((21, 2), dtype=float)
        pattern = finger_pattern(points, load_config().vision)
        self.assertEqual(pattern, (0, 0, 0, 0, 0))


class PresentationTests(unittest.TestCase):
    """Перевірки слайдів і шарів малювання."""

    def test_draw_erase_and_restore_slide(self):
        """Малювання змінює пікселі, а гумка повертає оригінал."""

        with tempfile.TemporaryDirectory() as folder:
            image = np.full((100, 100, 3), 180, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(Path(folder) / "01.png"), image))
            settings = load_config()
            settings = replace(settings, slides=replace(settings.slides, directory=Path(folder), scale=1.0))
            deck = Presentation(settings)
            deck.handle_gesture("start")
            original = deck.slides[0].copy()
            deck.render(np.array([50., 50.]), "draw", True)
            self.assertFalse(np.array_equal(deck.render(None, None, False), original))
            deck.render(np.array([50., 50.]), "erase", True)
            self.assertTrue(np.array_equal(deck.render(None, None, False), original))

    def test_missing_slides(self):
        """Програма пояснює, якщо презентація порожня."""

        with tempfile.TemporaryDirectory() as folder:
            settings = load_config()
            settings = replace(settings, slides=replace(settings.slides, directory=Path(folder)))
            with self.assertRaises(FileNotFoundError):
                Presentation(settings)


if __name__ == "__main__":
    unittest.main()
