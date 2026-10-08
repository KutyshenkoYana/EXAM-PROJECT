"""Завантаження та типізований доступ до налаштувань з config.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SlideSettings:
    """Параметри завантаження зображень слайдів."""

    directory: Path
    pattern: str
    scale: float


@dataclass(frozen=True)
class CameraSettings:
    """Параметри камери та попередньої обробки кадру."""

    index: int
    preview_width: int
    mirror: bool
    median_blur_kernel: int


@dataclass(frozen=True)
class VisionSettings:
    """Параметри MediaPipe і визначення випрямлених пальців."""

    max_num_hands: int
    detection_confidence: float
    tracking_confidence: float
    thumb_tip: int
    thumb_base: int
    thumb_reference_start: int
    thumb_reference_end: int
    thumb_open_ratio: float
    finger_tips: tuple[int, ...]
    finger_joint_offset: int
    wrist_index: int


@dataclass(frozen=True)
class PointerSettings:
    """Перетворення координат руки на координати презентації."""

    landmark_index: int
    crop_offset: float
    crop_width: float
    smoothing: float


@dataclass(frozen=True)
class GestureSettings:
    """Шаблони жестів і часові пороги для їх активації."""

    hold_seconds: float
    debounce_seconds: float
    patterns: dict[tuple[int, ...], str]


@dataclass(frozen=True)
class DrawingSettings:
    """Кольори, товщина ліній і розмір гумки."""

    colors: tuple[tuple[int, int, int], ...]
    brush_size: int
    eraser_radius: int
    eraser_stroke_multiplier: int
    erased_color: tuple[int, int, int]
    cursor_radius_extra: int
    cursor_line_width: int


@dataclass(frozen=True)
class DisplaySettings:
    """Розміри, позиції вікон та оформлення інтерфейсу."""

    stage_title: str
    camera_title: str
    stage_x: int
    stage_y: int
    camera_gap_x: int
    camera_y: int
    background_value: int
    wait_key_ms: int
    exit_key_codes: tuple[int, ...]
    skeleton_color: tuple[int, int, int]
    skeleton_thickness: int
    landmark_outer_radius: int
    landmark_inner_radius: int
    landmark_inner_color: tuple[int, int, int]
    status_radius: int
    status_border_thickness: int
    status_progress_thickness: int
    palette_x: int
    palette_spacing: int
    palette_bottom_margin: int
    palette_active_radius: int
    palette_inactive_radius: int
    palette_inner_active_radius: int
    palette_inner_inactive_radius: int


@dataclass(frozen=True)
class Settings:
    """Усі налаштування застосунку."""

    slides: SlideSettings
    camera: CameraSettings
    vision: VisionSettings
    pointer: PointerSettings
    gestures: GestureSettings
    drawing: DrawingSettings
    display: DisplaySettings


def _bgr(value: list[int]) -> tuple[int, int, int]:
    """Перетворити YAML-список на OpenCV BGR-колір."""

    if len(value) != 3 or any(not isinstance(v, int) or not 0 <= v <= 255 for v in value):
        raise ValueError(f"Некоректний BGR-колір: {value}")
    return tuple(value)


def load_config(path: Path | None = None) -> Settings:
    """Прочитати YAML і повернути налаштування з перевіркою базових значень."""

    config_path = Path(path) if path is not None else Path(__file__).with_name("config.yaml")
    with config_path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)

    if not isinstance(data, dict):
        raise ValueError("config.yaml повинен містити об'єкт із налаштуваннями.")

    slide_data = data["slides"]
    slides_path = Path(slide_data["directory"])
    if not slides_path.is_absolute():
        slides_path = PROJECT_ROOT / slides_path
    slides = SlideSettings(slides_path.resolve(), slide_data["pattern"], float(slide_data["scale"]))

    vision_data = data["vision"].copy()
    vision_data["finger_tips"] = tuple(vision_data["finger_tips"])
    vision = VisionSettings(**vision_data)

    patterns = {tuple(pattern): name for name, pattern in data["gestures"]["patterns"].items()}
    if len(patterns) != len(data["gestures"]["patterns"]):
        raise ValueError("Різні жести не можуть мати однакові шаблони пальців.")
    if any(len(pattern) != 5 or any(bit not in (0, 1) for bit in pattern) for pattern in patterns):
        raise ValueError("Кожен жест повинен бути списком із п'яти 0/1.")
    gestures = GestureSettings(
        hold_seconds=float(data["gestures"]["hold_seconds"]),
        debounce_seconds=float(data["gestures"]["debounce_seconds"]),
        patterns=patterns,
    )

    drawing_data = data["drawing"].copy()
    drawing_data["colors"] = tuple(_bgr(color) for color in drawing_data["colors"])
    drawing_data["erased_color"] = _bgr(drawing_data["erased_color"])
    drawing = DrawingSettings(**drawing_data)

    display_data = data["display"].copy()
    display_data["skeleton_color"] = _bgr(display_data["skeleton_color"])
    display_data["landmark_inner_color"] = _bgr(display_data["landmark_inner_color"])
    display_data["exit_key_codes"] = tuple(display_data["exit_key_codes"])
    display = DisplaySettings(**display_data)

    camera = CameraSettings(**data["camera"])
    pointer = PointerSettings(**data["pointer"])

    if not 0 <= pointer.smoothing <= 1:
        raise ValueError("pointer.smoothing має бути між 0 і 1.")
    if pointer.crop_width <= 0 or slides.scale <= 0:
        raise ValueError("Ширина області вказівника та масштаб слайдів мають бути > 0.")
    if camera.median_blur_kernel < 1 or camera.median_blur_kernel % 2 == 0:
        raise ValueError("camera.median_blur_kernel має бути непарним додатним числом.")
    if not drawing.colors:
        raise ValueError("drawing.colors не може бути порожнім.")
    if gestures.hold_seconds <= 0 or gestures.debounce_seconds < 0:
        raise ValueError("Час утримання має бути > 0, затримка стабілізації >= 0.")

    return Settings(slides, camera, vision, pointer, gestures, drawing, display)
