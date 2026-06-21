#!/usr/bin/env python3
"""
Desk Hero – combined face detection + robot patrol + Red Bull handoff.

Single process, single camera (robot wrist cam).
Sweeps left↔right, runs face/fatigue detection on the robot camera feed,
freezes when a human is seen, and triggers a Red Bull handoff when tired.

Usage:
    QT_QPA_PLATFORM=xcb lerobot/.venv/bin/python scripted_redbull.py
"""

import os
import sys
import time
import math
import json
import threading
import signal
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe import Image, ImageFormat
from ultralytics import YOLO

_lerobot_root = Path(__file__).resolve().parent / "lerobot" / "src"
if str(_lerobot_root) not in sys.path:
    sys.path.insert(0, str(_lerobot_root))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.configs import Cv2Backends

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDBULL_CLASS_ID = 39
YOLO_CONF = 0.5

ROBOT_PORT = "/dev/ttyACM0"
CAMERA_INDEX = "/dev/video4"
W, H, FPS = 640, 480, 30

REDBULL_CALIB_FILE = Path.home() / ".cache" / "lerobot_redbull_calib.json"
SWEEP_CALIB_FILE = Path.home() / ".cache" / "lerobot_sweep_calib.json"

GRIPPER_OPEN = 40.0
GRIPPER_CLOSED = 60.0

SWEEP_SPEED = 0.03

# Face landmarker model
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_PATH = Path.home() / ".cache" / "mediapipe" / "face_landmarker.task"

# Fatigue thresholds
EAR_CLOSED_THRESHOLD = 0.20
MAR_YAWN_THRESHOLD = 0.55
TIRED_SECONDS = 1.0
VERY_TIRED_SECONDS = 2.0
MICROSLEEP_SECONDS = 4.0
SLEEPING_LIKE_SECONDS = 8.0
SLEEPING_100_SECONDS = 10.0
YAWN_MIN_SECONDS = 0.7

# Red Bull trigger
ACTION_COOLDOWN_SECONDS = 20

REDBULL_TRIGGER_STATES = [
    "TIRED", "VERY_TIRED", "MICROSLEEP_RISK",
    "SLEEPING_LIKE", "SLEEPING_100_DEMO", "SAD_LIKE",
]

# Delivery offsets
DELIVERY_OFFSETS = {
    "left":  {"shoulder_pan": 25, "shoulder_lift": -5},
    "center":{"shoulder_pan": 0,  "shoulder_lift": -10},
    "right": {"shoulder_pan": -25,"shoulder_lift": -5},
}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_running = True
_bus_lock = threading.Lock()

_robot = None
_yolo = None
_face_landmarker = None
_connected = False

_sweep_active = False
_sweep_paused = threading.Event()
_sweep_paused.set()

_sweep_calib = None
_redbull_calib = None

# Red Bull state
_can_is_full = True
_last_redbull_time = 0.0
_redbull_running = False


def _sig_handler(sig, frame):
    global _running
    _running = False


signal.signal(signal.SIGINT, _sig_handler)


# ---------------------------------------------------------------------------
# MediaPipe model
# ---------------------------------------------------------------------------

def _ensure_face_model():
    if not FACE_LANDMARKER_PATH.exists():
        FACE_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[MODEL] Downloading face_landmarker.task ...")
        urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
        print("[MODEL] Done.")
    return str(FACE_LANDMARKER_PATH)


def _create_face_landmarker():
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_face_model()),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
    )


# ---------------------------------------------------------------------------
# Face / fatigue helpers (ported from emotion.py)
# ---------------------------------------------------------------------------

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _dist(p1, p2):
    return math.dist(p1, p2)


def _point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def _eye_aspect_ratio(landmarks, eye_indices, w, h):
    p1 = _point(landmarks, eye_indices[0], w, h)
    p2 = _point(landmarks, eye_indices[1], w, h)
    p3 = _point(landmarks, eye_indices[2], w, h)
    p4 = _point(landmarks, eye_indices[3], w, h)
    p5 = _point(landmarks, eye_indices[4], w, h)
    p6 = _point(landmarks, eye_indices[5], w, h)
    v1 = _dist(p2, p6)
    v2 = _dist(p3, p5)
    hz = _dist(p1, p4)
    if hz == 0:
        return 0.0
    return (v1 + v2) / (2.0 * hz)


def _mouth_aspect_ratio(landmarks, w, h):
    top = _point(landmarks, 13, w, h)
    bot = _point(landmarks, 14, w, h)
    left = _point(landmarks, 61, w, h)
    right = _point(landmarks, 291, w, h)
    v = _dist(top, bot)
    hz = _dist(left, right)
    if hz == 0:
        return 0.0
    return v / hz


def _mouth_expression_metrics(landmarks, w, h):
    lc = _point(landmarks, 61, w, h)
    rc = _point(landmarks, 291, w, h)
    ul = _point(landmarks, 13, w, h)
    ll = _point(landmarks, 14, w, h)
    lf = _point(landmarks, 234, w, h)
    rf = _point(landmarks, 454, w, h)
    face_w = _dist(lf, rf)
    mouth_w = _dist(lc, rc)
    if face_w == 0:
        return 0.0, 0.0
    smile = mouth_w / face_w
    lip_cy = (ul[1] + ll[1]) / 2
    corner_cy = (lc[1] + rc[1]) / 2
    droop = (corner_cy - lip_cy) / face_w
    return smile, droop


def _classify_expression(smile_ratio, corner_droop, mar, ear):
    if mar > 0.55 and ear > 0.25:
        return "SURPRISED_OR_BIG_YAWN"
    if mar > 0.45:
        return "MOUTH_OPEN_OR_YAWN"
    if smile_ratio > 0.42 and corner_droop < 0.03:
        return "POSITIVE_SMILE_LIKE"
    if smile_ratio < 0.38 and corner_droop > 0.015:
        return "SAD_LIKE"
    return "NEUTRAL_FOCUSED"


def _face_position(landmarks, w):
    """left / center / right based on face centre in frame."""
    xs = [lm.x * w for lm in landmarks]
    cx = sum(xs) / len(xs)
    if cx < w / 3:
        return "left"
    if cx > 2 * w / 3:
        return "right"
    return "center"


def _draw_face_overlay(frame, landmarks, w, h, color):
    # Face bbox
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    x1, y1 = max(0, min(xs) - 20), max(0, min(ys) - 20)
    x2, y2 = min(w - 1, max(xs) + 20), min(h - 1, max(ys) + 20)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Eyes
    for eye in (LEFT_EYE, RIGHT_EYE):
        pts = np.array([_point(landmarks, i, w, h) for i in eye], dtype=np.int32)
        rx, ry, rw, rh = cv2.boundingRect(pts)
        cv2.circle(frame, (rx + rw // 2, ry + rh // 2), max(rw, rh) // 2 + 4, color, 2)


# ---------------------------------------------------------------------------
# Robot primitives
# ---------------------------------------------------------------------------

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _get_joints():
    with _bus_lock:
        obs = _robot.get_observation()
    return {k: v for k, v in obs.items() if k.endswith(".pos")}


def _send_joints(joints, duration=0.0, interpolate=False):
    if interpolate and duration > 0:
        start = _get_joints()
        steps = max(1, int(duration * 30))
        for i in range(1, steps + 1):
            if not _running:
                return
            t = i / steps
            interp = {}
            for key in joints:
                if key in start:
                    interp[key] = start[key] + (joints[key] - start[key]) * t
                else:
                    interp[key] = joints[key]
            with _bus_lock:
                _robot.send_action(interp)
            time.sleep(duration / steps)
    else:
        with _bus_lock:
            _robot.send_action(joints)
        if duration > 0:
            t0 = time.time()
            while time.time() - t0 < duration and _running:
                time.sleep(0.05)


def _detect_redbull(frame):
    results = _yolo(frame, verbose=False)[0]
    detections = []
    for det in results.boxes:
        if int(det.cls[0]) == REDBULL_CLASS_ID and float(det.conf[0]) >= YOLO_CONF:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            detections.append(((x1 + x2) // 2, (y1 + y2) // 2, (x1, y1, x2, y2)))
    return detections


def _interpolate_hover(pixel, hover_points):
    cx, cy = pixel
    best = min(hover_points, key=lambda p: (p["pixel"][0] - cx) ** 2 + (p["pixel"][1] - cy) ** 2)
    return best["joints"].copy()


# ---------------------------------------------------------------------------
# Sweep calibration (auto-gen from Red Bull calib)
# ---------------------------------------------------------------------------

def _build_sweep_calib():
    global _sweep_calib

    if _sweep_calib is not None:
        return True

    if SWEEP_CALIB_FILE.exists():
        _sweep_calib = _load_json(SWEEP_CALIB_FILE)
        return True

    if REDBULL_CALIB_FILE.exists():
        rb = _load_json(REDBULL_CALIB_FILE)
        base = rb["hover_points"][0]["joints"].copy()
        centre = {k: v for k, v in base.items()}
        left = {k: v for k, v in base.items()}
        right = {k: v for k, v in base.items()}

        for pose in (left, centre, right):
            if "shoulder_lift.pos" in pose:
                pose["shoulder_lift.pos"] = 0
            if "elbow_flex.pos" in pose:
                pose["elbow_flex.pos"] = 0
            if "wrist_flex.pos" in pose:
                pose["wrist_flex.pos"] = -30

        if "shoulder_pan.pos" in left:
            left["shoulder_pan.pos"] += 55
        if "shoulder_pan.pos" in right:
            right["shoulder_pan.pos"] -= 55

        _sweep_calib = {"left": left, "centre": centre, "right": right}
        print("[SWEEP] Auto-generated sweep from Red Bull calib.")
        print("[SWEEP] Run le_robot_controller.py mode 1 for proper calibration.")
        return True

    print("[SWEEP] No sweep or Red Bull calibration found. Sweep disabled.")
    return False


# ---------------------------------------------------------------------------
# Red Bull handoff
# ---------------------------------------------------------------------------

def _do_redbull_handoff(person_position):
    global _redbull_running, _redbull_calib, _can_is_full, _last_redbull_time

    now = time.time()
    if _redbull_running:
        return
    if now - _last_redbull_time < ACTION_COOLDOWN_SECONDS:
        return

    _last_redbull_time = now
    _redbull_running = True
    _pause_sweep()
    time.sleep(0.2)

    try:
        if _redbull_calib is None:
            _redbull_calib = _load_json(REDBULL_CALIB_FILE) if REDBULL_CALIB_FILE.exists() else None

        if _redbull_calib is None:
            print("[REDBULL] No calibration.")
            return

        hover_points = _redbull_calib["hover_points"]
        delta = _redbull_calib["grasp_delta"]
        lift_pos = _redbull_calib["lift_position"]
        neutral = hover_points[0]["joints"].copy()
        neutral["gripper.pos"] = GRIPPER_OPEN

        # 1. Move to neutral
        _send_joints(neutral, duration=1.5, interpolate=True)
        if not _running: return

        # 2. Detect can
        with _bus_lock:
            frame = _robot.get_observation()["front"].copy()
        detections = _detect_redbull(frame)
        if not detections:
            print("[REDBULL] No can detected.")
            return

        cx, cy, _ = detections[0]
        hover = _interpolate_hover((cx, cy), hover_points)
        hover["gripper.pos"] = GRIPPER_OPEN
        _send_joints(hover, duration=1.0, interpolate=True)
        if not _running: return

        # 3. Grasp
        grasp = _get_joints()
        for key in delta:
            if key != "gripper.pos":
                grasp[key] = grasp.get(key, 0) + delta[key]
        grasp["gripper.pos"] = GRIPPER_OPEN
        _send_joints(grasp, duration=1.0, interpolate=True)
        if not _running: return

        grasp["gripper.pos"] = GRIPPER_CLOSED
        _send_joints(grasp, duration=0.6)

        # 4. Lift
        _send_joints(lift_pos, duration=1.0, interpolate=True)
        if not _running: return

        print("[REDBULL] Can grabbed, delivering...")

        # 5. Deliver toward person
        delivery = _get_joints()
        offset = DELIVERY_OFFSETS.get(person_position, DELIVERY_OFFSETS["center"])
        for key, val in offset.items():
            jk = f"{key}.pos"
            if jk in delivery:
                delivery[jk] += val
        _send_joints(delivery, duration=1.5, interpolate=True)
        if not _running: return

        # 6. Release
        delivery["gripper.pos"] = GRIPPER_OPEN
        _send_joints(delivery, duration=0.8)
        print(f"[REDBULL] Delivered to {person_position}!")
        _can_is_full = False

        # 7. Return to neutral
        _send_joints(neutral, duration=1.5, interpolate=True)

    finally:
        _redbull_running = False
        _resume_sweep()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    global _running, _robot, _yolo, _face_landmarker, _connected, _can_is_full

    # ── Robot connect ────────────────────────────────────────────────
    robot_cfg = SO100FollowerConfig(
        port=ROBOT_PORT,
        id="scripted_redbull",
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=CAMERA_INDEX,
                width=W, height=H, fps=FPS,
                backend=Cv2Backends.V4L2,
            ),
        },
        use_degrees=True,
        max_relative_target=30.0,
    )
    _robot = SO100Follower(robot_cfg)
    _robot.connect()
    _yolo = YOLO("yolov8n.pt")
    _face_landmarker = _create_face_landmarker()
    _connected = True

    # Warm up camera
    for _ in range(10):
        with _bus_lock:
            _robot.get_observation()

    # ── Sweep setup ──────────────────────────────────────────────────
    if not _build_sweep_calib():
        _disconnect()
        return

    left = _sweep_calib["left"].copy()
    right = _sweep_calib["right"].copy()

    current = _get_joints()
    target = left.copy()
    direction = 1
    wait_frames = 0

    # ── Fatigue tracking state ───────────────────────────────────────
    perclos_window = deque(maxlen=180)
    blink_times = deque(maxlen=100)
    yawn_times = deque(maxlen=30)
    eyes_closed_start = None
    yawn_start = None
    was_eyes_closed = False
    app_start = time.time()

    last_expression = "UNKNOWN"
    expr_candidate = "UNKNOWN"
    expr_candidate_start = None

    confirmed_state = "IDLE"
    state_candidate = "IDLE"
    state_candidate_start = None

    CONFIRM_S = 0.4
    face_watching = False
    face_lost_time = None
    FACE_LOST_TIMEOUT = 1.5

    global _sweep_active
    _sweep_active = True

    frame_count = 0

    print("\n=== Desk Hero ===")
    print("Sweeping + face detection on camera 4.")
    print("F = toggle can  |  R = manual Red Bull  |  Q = quit\n")

    try:
        while _running:
            # ── Get robot observation (joints + camera frame) ─────
            with _bus_lock:
                obs = _robot.get_observation()
            frame = obs["front"].copy()
            now = time.time()
            runtime = now - app_start

            # ── Face detection ────────────────────────────────────
            mp_image = Image(image_format=ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = _face_landmarker.detect(mp_image)

            face_detected = False
            landmarks = None
            person_position = "center"
            ear = mar = smile_ratio = corner_droop = None
            fatigue_status = "NO_FACE"
            fatigue_score = 0
            sleep_confidence = 0
            expression_cue = "UNKNOWN"
            raw_state = "IDLE"
            eyes_closed_duration = 0.0

            if result.face_landmarks:
                face_detected = True
                landmarks = result.face_landmarks[0]
                person_position = _face_position(landmarks, W)
                face_lost_time = None

                # EAR / MAR
                le = _eye_aspect_ratio(landmarks, LEFT_EYE, W, H)
                re = _eye_aspect_ratio(landmarks, RIGHT_EYE, W, H)
                ear = (le + re) / 2.0
                mar = _mouth_aspect_ratio(landmarks, W, H)

                eyes_closed = ear < EAR_CLOSED_THRESHOLD
                mouth_open = mar > MAR_YAWN_THRESHOLD

                # Eye closure tracking
                if eyes_closed:
                    if eyes_closed_start is None:
                        eyes_closed_start = now
                    eyes_closed_duration = now - eyes_closed_start
                else:
                    if was_eyes_closed and eyes_closed_start is not None:
                        dur = now - eyes_closed_start
                        if 0.05 <= dur <= 0.7:
                            blink_times.append(now)
                    eyes_closed_start = None
                    eyes_closed_duration = 0.0
                was_eyes_closed = eyes_closed

                # Yawn tracking
                if mouth_open:
                    if yawn_start is None:
                        yawn_start = now
                else:
                    if yawn_start is not None:
                        if now - yawn_start >= YAWN_MIN_SECONDS:
                            yawn_times.append(now)
                    yawn_start = None

                # PERCLOS
                perclos_window.append(1 if eyes_closed else 0)
                perclos = sum(perclos_window) / max(1, len(perclos_window))

                # Clean old events
                while blink_times and now - blink_times[0] > 60:
                    blink_times.popleft()
                while yawn_times and now - yawn_times[0] > 60:
                    yawn_times.popleft()

                blink_rate = (len(blink_times) * 60.0 / min(60.0, max(1.0, runtime))
                              if runtime > 10 else 0)
                yawns_last_min = len(yawn_times)

                # Expression
                smile_ratio, corner_droop = _mouth_expression_metrics(landmarks, W, H)
                raw_expr = _classify_expression(smile_ratio, corner_droop, mar, ear)

                if raw_expr != expr_candidate:
                    expr_candidate = raw_expr
                    expr_candidate_start = now
                if expr_candidate_start and now - expr_candidate_start >= CONFIRM_S:
                    last_expression = expr_candidate
                expression_cue = last_expression

                # Fatigue score
                score = 0
                cd = eyes_closed_duration
                if cd >= SLEEPING_100_SECONDS:   score = 100
                elif cd >= SLEEPING_LIKE_SECONDS: score += 95
                elif cd >= MICROSLEEP_SECONDS:    score += 85
                elif cd >= VERY_TIRED_SECONDS:    score += 70
                elif cd >= TIRED_SECONDS:         score += 45

                if perclos > 0.80:      score += 60
                elif perclos > 0.45:    score += 45
                elif perclos > 0.30:    score += 30
                elif perclos > 0.18:    score += 15
                if yawns_last_min >= 2: score += 30
                elif yawns_last_min == 1: score += 18
                if ear < 0.23:          score += 10
                if runtime > 30 and blink_rate < 5: score += 8
                fatigue_score = min(score, 100)

                # Status classification
                if cd >= SLEEPING_100_SECONDS:
                    fatigue_status = "SLEEPING_100_DEMO"
                    sleep_confidence = 100
                elif cd >= SLEEPING_LIKE_SECONDS:
                    fatigue_status = "SLEEPING_LIKE"
                    sleep_confidence = 90
                elif cd >= MICROSLEEP_SECONDS:
                    fatigue_status = "MICROSLEEP_RISK"
                    sleep_confidence = 70
                elif cd >= VERY_TIRED_SECONDS:
                    fatigue_status = "VERY_TIRED"
                    sleep_confidence = 40
                elif cd >= TIRED_SECONDS:
                    fatigue_status = "TIRED"
                    sleep_confidence = 20
                elif fatigue_score >= 70:
                    fatigue_status = "VERY_TIRED"
                elif fatigue_score >= 35:
                    fatigue_status = "TIRED"
                else:
                    fatigue_status = "NOT_TIRED"

                # Robot state
                if fatigue_status == "SLEEPING_100_DEMO":
                    raw_state = "SLEEPING_100_DEMO"
                elif fatigue_status == "SLEEPING_LIKE":
                    raw_state = "SLEEPING_LIKE"
                elif fatigue_status == "MICROSLEEP_RISK":
                    raw_state = "MICROSLEEP_RISK"
                elif fatigue_status == "VERY_TIRED":
                    raw_state = "VERY_TIRED"
                elif fatigue_status == "TIRED":
                    raw_state = "TIRED"
                elif expression_cue == "SAD_LIKE":
                    raw_state = "SAD_LIKE"
                else:
                    raw_state = "IDLE"

            else:
                # No face
                perclos = 0.0
                perclos_window.clear()
                was_eyes_closed = False
                eyes_closed_start = None
                yawn_start = None
                if face_lost_time is None:
                    face_lost_time = now

            # ── Stabilize state ────────────────────────────────────
            if raw_state != state_candidate:
                state_candidate = raw_state
                state_candidate_start = now
            if state_candidate_start and now - state_candidate_start >= CONFIRM_S:
                confirmed_state = state_candidate
            robot_state = confirmed_state

            # ── Face watch: freeze sweep when human is seen ────────
            if face_detected:
                face_watching = True
            elif face_lost_time and now - face_lost_time > FACE_LOST_TIMEOUT:
                face_watching = False

            # ── Sweep movement (skip if watching face or handoff) ──
            if not face_watching and not _redbull_running:
                keys = [k for k, v in current.items() if isinstance(v, (int, float))]
                for key in keys:
                    if key in target and key != "gripper.pos":
                        current[key] += (target[key] - current[key]) * SWEEP_SPEED
                current["gripper.pos"] = GRIPPER_OPEN
                with _bus_lock:
                    _robot.send_action(current)

                dist_sum = sum(abs(current.get(k, 0) - target.get(k, 0))
                               for k in keys if k != "gripper.pos")
                if dist_sum < 2.0:
                    wait_frames += 1
                    if wait_frames > 10:
                        wait_frames = 0
                        direction *= -1
                        target = right.copy() if direction == 1 else left.copy()

            # ── Red Bull trigger ───────────────────────────────────
            if (face_detected and _can_is_full
                    and robot_state in REDBULL_TRIGGER_STATES
                    and not _redbull_running):
                print(f"[SYSTEM] Triggering Red Bull handoff ({robot_state})")
                t = threading.Thread(target=_do_redbull_handoff,
                                     args=(person_position,), daemon=True)
                t.start()

            # ── Draw overlay ───────────────────────────────────────
            if face_detected and landmarks:
                if robot_state in ("SLEEPING_100_DEMO", "SLEEPING_LIKE"):
                    color = (0, 0, 255)
                elif robot_state in ("MICROSLEEP_RISK", "VERY_TIRED", "TIRED"):
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)
                _draw_face_overlay(frame, landmarks, W, H, color)

            # Status text
            status = "WATCHING" if face_watching else "SWEEPING"
            if _redbull_running:
                status = "HANDOFF"
            cv2.putText(frame, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if _redbull_running else
                        (0, 255, 0) if face_watching else (255, 255, 255), 2)

            cv2.putText(frame, f"State: {robot_state} | Fatigue: {fatigue_status} {fatigue_score}/100",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if ear is not None:
                cv2.putText(frame, f"EAR: {ear:.3f} | MAR: {mar:.3f} | PERCLOS: {perclos:.2f}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.putText(frame, f"Eyes closed: {eyes_closed_duration:.1f}s | Can: {'FULL' if _can_is_full else 'EMPTY'}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.putText(frame, "F:can R:handoff Q:quit", (10, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow("Desk Hero", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("f"):
                _can_is_full = not _can_is_full
                print(f"[SYSTEM] can_is_full = {_can_is_full}")
            elif key == ord("r"):
                t = threading.Thread(target=_do_redbull_handoff,
                                     args=(person_position,), daemon=True)
                t.start()

            time.sleep(1.0 / 30)

    except KeyboardInterrupt:
        pass
    finally:
        _disconnect()


def _disconnect():
    global _connected, _sweep_active
    _sweep_active = False
    _sweep_paused.set()
    if _face_landmarker:
        try:
            _face_landmarker.close()
        except Exception:
            pass
    if _connected and _robot is not None:
        try:
            _robot.disconnect()
        except Exception:
            pass
        _connected = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
