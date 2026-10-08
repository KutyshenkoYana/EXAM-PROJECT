"""Точка входу в презентацію з керуванням жестами руки."""

import time

import cv2

from config import Settings, load_config
from core.gesture import GestureTracker
from core.presentation import Presentation
from core.vision import HandObservation, HandVision
from core.visualize import draw_hand, draw_status


def prepare_frame(frame, settings: Settings):
    """Віддзеркалити та згладити відеокадр перед розпізнаванням."""

    if settings.camera.mirror:
        frame = cv2.flip(frame, 1)
    return cv2.medianBlur(frame, settings.camera.median_blur_kernel)


def handle_gesture(observation: HandObservation, tracker: GestureTracker,
                   deck: Presentation) -> None:
    """Оновити таймер утримання і передати одноразові команди презентації."""

    changed = tracker.update(observation.gesture, time.monotonic())
    if changed:
        deck.reset_stroke()
    # Не виконуємо команду, якщо руку вже прибрано або жест змінюється.
    if observation.gesture == tracker.held:
        action = tracker.consume_action()
        if action in ("start", "next", "prev", "color"):
            deck.handle_gesture(action)


def visualize(frame, observation: HandObservation, tracker: GestureTracker,
              deck: Presentation, settings: Settings) -> None:
    """Показати слайди, позицію курсора, модель руки та прогрес жесту."""

    height, width = frame.shape[:2]
    cam_width = settings.camera.preview_width
    camera_image = cv2.resize(frame, (cam_width, cam_width * height // width))

    if observation.camera_points is not None:
        draw_hand(camera_image, observation.camera_points, settings.display)
    draw_status(camera_image, observation.camera_tip, deck.current_color,
                settings.drawing.colors, tracker.progress, settings.display)

    # Під час зміни жесту не домальовуємо лінію старим інструментом.
    active_gesture = tracker.held if tracker.held == observation.gesture else None
    stage = deck.render(observation.slide_point, active_gesture,
                        tracker.ready and active_gesture is not None)
    cv2.imshow(settings.display.stage_title, stage)
    cv2.imshow(settings.display.camera_title, camera_image)


def should_stop(settings: Settings) -> bool:
    """Перевірити натиснення Escape/q або закриття одного з вікон."""

    key = cv2.waitKey(settings.display.wait_key_ms) & 0xFF
    if key in settings.display.exit_key_codes:
        return True

    try:
        for title in (settings.display.stage_title, settings.display.camera_title):
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return True
    except cv2.error:
        return True
    return False


def run(camera: cv2.VideoCapture, deck: Presentation, settings: Settings) -> None:
    """Запустити цикл захоплення кадрів, розпізнавання жестів і показу слайдів."""

    cv2.namedWindow(settings.display.stage_title)
    cv2.namedWindow(settings.display.camera_title)
    cv2.moveWindow(settings.display.stage_title, settings.display.stage_x,
                   settings.display.stage_y)
    cv2.moveWindow(settings.display.camera_title, deck.width + settings.display.camera_gap_x,
                   settings.display.camera_y)

    tracker = GestureTracker(settings.gestures)
    with HandVision(settings) as vision:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame = prepare_frame(frame, settings)
            observation = vision.observe(frame, (deck.width, deck.height))
            handle_gesture(observation, tracker, deck)
            visualize(frame, observation, tracker, deck, settings)
            if should_stop(settings):
                break


def main() -> None:
    """Завантажити налаштування й слайди, відкрити камеру та прибрати ресурси."""

    settings = load_config()
    deck = Presentation(settings)
    camera = cv2.VideoCapture(settings.camera.index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Не вдалося відкрити камеру #{settings.camera.index}. "
            "Перевірте підключення та дозволи для камери."
        )
    try:
        run(camera, deck, settings)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
