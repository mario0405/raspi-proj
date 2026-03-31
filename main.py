#!/usr/bin/env python3
"""
AI-powered bird deterrent for Raspberry Pi 5.

What this script does:
1. Captures camera frames using Picamera2 (required for Pi Camera Module 3 on Pi 5).
2. Runs TensorFlow Lite object detection (MobileNet SSD COCO).
3. Triggers a deterrent sound when a bird is detected above a confidence threshold.
4. Enforces an audio cooldown to avoid repeated overlapping playback.
5. Uses Astral sunrise/sunset calculations for Basel, Switzerland and only performs
   heavy AI work during daylight. At night it sleeps in a low-resource loop.

Designed to run continuously inside Docker with restart policy: unless-stopped.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
from astral import LocationInfo
from astral.sun import sun
from PIL import Image
from picamera2 import Picamera2

# Prefer tflite_runtime for low overhead on edge devices.
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback for environments where only full TensorFlow is available.
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore


# -----------------------------
# Runtime configuration
# -----------------------------
# Basel, Switzerland (Claragraben area).
DEFAULT_LAT = 47.5596
DEFAULT_LON = 7.5886
DEFAULT_TIMEZONE = "Europe/Zurich"

# Main behavior defaults. Override any of these with environment variables.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "15"))
INFERENCE_INTERVAL_SECONDS = float(os.getenv("INFERENCE_INTERVAL_SECONDS", "0.35"))
NIGHT_SLEEP_SECONDS = int(os.getenv("NIGHT_SLEEP_SECONDS", "60"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
SAVE_TRIGGER_SNAPSHOTS = os.getenv("SAVE_TRIGGER_SNAPSHOTS", "true").lower() == "true"
SNAPSHOT_JPEG_QUALITY = int(os.getenv("SNAPSHOT_JPEG_QUALITY", "85"))

# File paths inside the container.
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/detect.tflite"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", "/app/models/labelmap.txt"))
SCARE_SOUND_PATH = Path(os.getenv("SCARE_SOUND_PATH", "/app/assets/scare_sound.wav"))
SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", "/app/snapshots"))

# Optional ALSA device override for USB audio adapter, for example: "plughw:1,0".
ALSA_DEVICE = os.getenv("ALSA_DEVICE", "").strip()

# Global run flag that allows clean shutdown from SIGINT/SIGTERM.
RUNNING = True


@dataclass(frozen=True)
class AppLocation:
    """Represents the installation location for Astral day/night calculations."""

    name: str
    region: str
    timezone: str
    latitude: float
    longitude: float


def handle_signal(signum: int, _frame) -> None:
    """Signal handler so Docker stop signals can terminate the loop cleanly."""
    global RUNNING
    print(f"[signal] Received signal {signum}, shutting down...")
    RUNNING = False


def load_labels(path: Path) -> Dict[int, str]:
    """
    Load labels from a TFLite label file.

    Supported formats:
    - "0 person"
    - "person" (index inferred by line number)
    """
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    labels: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            clean = line.strip()
            if not clean:
                continue

            parts = clean.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                class_id = int(parts[0])
                class_name = parts[1].strip().lower()
            else:
                class_id = idx
                class_name = clean.lower()

            labels[class_id] = class_name

    if not labels:
        raise ValueError(f"Label file is empty: {path}")

    return labels


def find_bird_class_ids(labels: Dict[int, str]) -> Set[int]:
    """
    Find all class IDs mapped to bird labels.

    We match exact "bird" and variants like "bird (flying)" just in case.
    If no explicit label is found, we fall back to COCO's common class ID 16.
    """
    bird_ids = {cid for cid, name in labels.items() if name == "bird" or name.startswith("bird ")}

    if not bird_ids:
        bird_ids = {16}
        print("[warn] No explicit 'bird' label found. Falling back to COCO class ID 16.")

    return bird_ids


def is_daylight(now_local: datetime, loc: AppLocation) -> Tuple[bool, datetime, datetime]:
    """
    Return daylight status and today's sunrise/sunset in local timezone.

    Heavy camera processing/inference only runs when now is between sunrise and sunset.
    """
    info = LocationInfo(loc.name, loc.region, loc.timezone, loc.latitude, loc.longitude)
    s = sun(info.observer, date=now_local.date(), tzinfo=loc.timezone)
    sunrise = s["sunrise"]
    sunset = s["sunset"]
    return sunrise <= now_local <= sunset, sunrise, sunset


def sleep_until_daytime(loc: AppLocation, night_sleep_seconds: int) -> None:
    """
    Low-resource wait loop used overnight.

    Instead of stopping the container, we keep a tiny loop alive and periodically
    check if daylight started. This keeps Docker healthy while minimizing CPU load.
    """
    while RUNNING:
        now_local = datetime.now().astimezone()
        daylight, sunrise, sunset = is_daylight(now_local, loc)

        if daylight:
            print("[cycle] Daylight detected. Resuming camera + AI inference.")
            return

        if now_local < sunrise:
            next_event = sunrise
            event_name = "sunrise"
        else:
            next_event = sunrise + timedelta(days=1)
            event_name = "next sunrise"

        seconds_to_event = int((next_event - now_local).total_seconds())
        sleep_for = max(5, min(night_sleep_seconds, seconds_to_event))

        print(
            f"[cycle] Night mode. Waiting for {event_name} "
            f"at {next_event.isoformat()} (sleep {sleep_for}s)."
        )
        time.sleep(sleep_for)


def init_camera(width: int, height: int) -> Picamera2:
    """Initialize Picamera2 with a small RGB stream for efficient inference."""
    picam2 = Picamera2()

    # RGB888 avoids repeated color conversion before feeding the model.
    config = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    # Short sensor warm-up helps stabilize first frames.
    time.sleep(1.0)
    print(f"[camera] Started at {width}x{height} RGB888")
    return picam2


def init_interpreter(model_path: Path) -> tuple[Interpreter, Tuple[int, int], bool]:
    """
    Load TFLite model and return:
    - interpreter
    - expected input size as (width, height)
    - whether model expects float32 input
    """
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    interpreter = Interpreter(model_path=str(model_path), num_threads=2)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    _, in_h, in_w, _ = input_details["shape"]
    is_float_model = input_details["dtype"] == np.float32

    print(
        f"[model] Loaded {model_path.name} | input={in_w}x{in_h} "
        f"| float_input={is_float_model}"
    )
    return interpreter, (in_w, in_h), is_float_model


def preprocess_frame(frame_rgb: np.ndarray, target_size: Tuple[int, int], is_float_model: bool) -> np.ndarray:
    """
    Resize camera frame to the model input size and apply normalization if needed.

    Returns tensor with shape [1, H, W, 3].
    """
    target_w, target_h = target_size

    # Pillow resize is simple and reliable inside minimal containers.
    img = Image.fromarray(frame_rgb)
    img = img.resize((target_w, target_h), resample=Image.BILINEAR)
    arr = np.asarray(img)

    if is_float_model:
        arr = (arr.astype(np.float32) - 127.5) / 127.5

    return np.expand_dims(arr, axis=0)


def detect_bird(
    interpreter: Interpreter,
    input_tensor: np.ndarray,
    labels: Dict[int, str],
    bird_class_ids: Set[int],
    confidence_threshold: float,
) -> Tuple[bool, float, int, str]:
    """
    Run one inference and check if a bird above threshold is present.

    Returns: (bird_found, best_score, class_id, class_name)
    """
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()

    output_details = interpreter.get_output_details()

    # Typical SSD order:
    # 0: boxes [1, N, 4], 1: classes [1, N], 2: scores [1, N], 3: count [1]
    classes = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]

    best_score = 0.0
    best_class = -1

    for class_id_float, score in zip(classes, scores):
        cid = int(class_id_float)
        score_f = float(score)

        if score_f < confidence_threshold:
            continue

        if cid in bird_class_ids:
            if score_f > best_score:
                best_score = score_f
                best_class = cid

    if best_class >= 0:
        return True, best_score, best_class, labels.get(best_class, "bird")

    return False, 0.0, -1, ""


def play_deterrent_sound(sound_path: Path, alsa_device: str) -> bool:
    """
    Play deterrent sound using 'aplay' to keep dependencies lightweight.

    If ALSA_DEVICE is set (for USB audio adapter routing), it is passed as:
    aplay -D <device>
    """
    if not sound_path.exists():
        print(f"[audio][error] Sound file not found: {sound_path}")
        return False

    cmd = ["aplay", "-q"]
    if alsa_device:
        cmd.extend(["-D", alsa_device])
    cmd.append(str(sound_path))

    try:
        subprocess.run(cmd, check=True)
        print(f"[audio] Played deterrent sound: {sound_path.name}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[audio][error] aplay failed: {exc}")
        return False


def save_trigger_snapshot(
    frame_rgb: np.ndarray,
    snapshot_dir: Path,
    score: float,
    class_name: str,
    jpeg_quality: int,
) -> Path | None:
    """Save a JPEG snapshot for later review when a trigger event occurs."""
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        score_pct = int(score * 100)
        safe_class = class_name.replace(" ", "_").replace("/", "_")
        filename = f"trigger_{ts}_{safe_class}_{score_pct:02d}.jpg"
        out_path = snapshot_dir / filename

        Image.fromarray(frame_rgb).save(
            out_path,
            format="JPEG",
            quality=max(20, min(jpeg_quality, 95)),
            optimize=True,
        )
        return out_path
    except Exception as exc:  # noqa: BLE001 - snapshot failure must not crash service
        print(f"[snapshot][error] Failed to save trigger snapshot: {exc}")
        return None


def main() -> None:
    """Main execution loop."""
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    location = AppLocation(
        name="Basel",
        region="Switzerland",
        timezone=os.getenv("TIMEZONE", DEFAULT_TIMEZONE),
        latitude=float(os.getenv("LATITUDE", str(DEFAULT_LAT))),
        longitude=float(os.getenv("LONGITUDE", str(DEFAULT_LON))),
    )

    print("[boot] Starting AI bird deterrent...")
    print(
        "[boot] Config | "
        f"threshold={CONFIDENCE_THRESHOLD:.2f} | "
        f"cooldown={COOLDOWN_SECONDS}s | "
        f"interval={INFERENCE_INTERVAL_SECONDS}s"
    )
    print(
        "[boot] Snapshot config | "
        f"enabled={SAVE_TRIGGER_SNAPSHOTS} | "
        f"dir={SNAPSHOT_DIR} | "
        f"jpeg_quality={SNAPSHOT_JPEG_QUALITY}"
    )

    labels = load_labels(LABELS_PATH)
    bird_class_ids = find_bird_class_ids(labels)
    print(f"[model] Bird class IDs: {sorted(bird_class_ids)}")

    interpreter, model_size, is_float_model = init_interpreter(MODEL_PATH)

    # Camera startup is delayed until daylight to reduce wear/heat overnight.
    picam2: Picamera2 | None = None

    next_allowed_trigger_ts = 0.0

    while RUNNING:
        now_local = datetime.now().astimezone()
        daylight, sunrise, sunset = is_daylight(now_local, location)

        if not daylight:
            if picam2 is not None:
                print("[camera] Night mode reached. Stopping camera to reduce resource usage.")
                picam2.stop()
                picam2.close()
                picam2 = None

            print(
                f"[cycle] Night mode active (sunrise: {sunrise.time()}, sunset: {sunset.time()})."
            )
            sleep_until_daytime(location, NIGHT_SLEEP_SECONDS)
            continue

        if picam2 is None:
            picam2 = init_camera(CAMERA_WIDTH, CAMERA_HEIGHT)

        # Capture one frame and run detection.
        frame = picam2.capture_array()
        input_tensor = preprocess_frame(frame, model_size, is_float_model)

        bird_found, score, class_id, class_name = detect_bird(
            interpreter=interpreter,
            input_tensor=input_tensor,
            labels=labels,
            bird_class_ids=bird_class_ids,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )

        now_ts = time.time()
        in_cooldown = now_ts < next_allowed_trigger_ts

        if bird_found:
            print(
                f"[detect] Bird detected: class={class_name}({class_id}) "
                f"score={score:.2f} cooldown_active={in_cooldown}"
            )

            if not in_cooldown:
                played = play_deterrent_sound(SCARE_SOUND_PATH, ALSA_DEVICE)
                if played:
                    if SAVE_TRIGGER_SNAPSHOTS:
                        snap = save_trigger_snapshot(
                            frame_rgb=frame,
                            snapshot_dir=SNAPSHOT_DIR,
                            score=score,
                            class_name=class_name,
                            jpeg_quality=SNAPSHOT_JPEG_QUALITY,
                        )
                        if snap is not None:
                            print(f"[snapshot] Saved trigger snapshot: {snap}")

                    next_allowed_trigger_ts = time.time() + COOLDOWN_SECONDS
                    print(f"[cooldown] Audio triggered. Cooling down for {COOLDOWN_SECONDS}s.")

        # Small fixed delay to cap CPU usage and heat.
        time.sleep(max(0.05, INFERENCE_INTERVAL_SECONDS))

    if picam2 is not None:
        picam2.stop()
        picam2.close()

    print("[shutdown] Bird deterrent stopped cleanly.")


if __name__ == "__main__":
    main()
