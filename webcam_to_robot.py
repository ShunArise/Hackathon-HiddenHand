#!/usr/bin/env python3
"""
Laptop webcam → SO100 robot Red Bull handoff.

Uses laptop camera (index 0) for face/fatigue/expression detection.
When tired states are detected, triggers the SO100 robot arm to
grab a Red Bull can and deliver it toward the person's position.

Usage:
    QT_QPA_PLATFORM=xcb lerobot/.venv/bin/python webcam_to_robot.py
"""
import os
import sys
import time
import math
import json
import threading
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
# Config
# ---------------------------------------------------------------------------

CAMERA_INDEX = 1                    # laptop webcam
ROBOT_PORT = "/dev/ttyACM0"
ROBOT_CAMERA = "/dev/video"         # robot wrist cam (for can detection)
REDBULL_CALIB_FILE = Path.home() / ".cache" / "lerobot_redbull_calib.json"
REDBULL_CLASS_ID = 39
YOLO_CONF = 0.5

GRIPPER_OPEN = 40.0
GRIPPER_CLOSED = 60.0

ACTION_COOLDOWN_SECONDS = 20

REDBULL_TRIGGER_STATES = [
    "TIRED", "VERY_TIRED", "MICROSLEEP_RISK",
    "SLEEPING_LIKE", "SLEEPING_100_DEMO", "SAD_LIKE",
]

DELIVERY_OFFSETS = {
    "left":   {"shoulder_pan": 25, "shoulder_lift": -5},
    "center": {"shoulder_pan": 0,  "shoulder_lift": -10},
    "right":  {"shoulder_pan": -25,"shoulder_lift": -5},
}

# MediaPipe face model
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_PATH = Path.home() / ".cache" / "mediapipe" / "face_landmarker.task"

# Fatigue thresholds (same as emotion.py)
EAR_CLOSED = 0.20
MAR_YAWN = 0.55
TIRED_S = 1.0
VERY_TIRED_S = 2.0
MICROSLEEP_S = 4.0
SLEEPING_LIKE_S = 8.0
SLEEPING_100_S = 10.0
YAWN_MIN_S = 0.7

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_running = True
_can_is_full = True
_last_redbull_time = 0.0
_redbull_running = False
_robot_connected = False
_robot = None
_yolo = None

# ---------------------------------------------------------------------------
# Face landmarker
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
# Face / fatigue helpers
# ---------------------------------------------------------------------------

def _dist(p1, p2):
    return math.dist(p1, p2)


def _point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def _ear(landmarks, eye_idx, w, h):
    p = [_point(landmarks, i, w, h) for i in eye_idx]
    v1 = _dist(p[1], p[5])
    v2 = _dist(p[2], p[4])
    hz = _dist(p[0], p[3])
    if hz == 0:
        return 0.0
    return (v1 + v2) / (2.0 * hz)


def _mar(landmarks, w, h):
    top = _point(landmarks, 13, w, h)
    bot = _point(landmarks, 14, w, h)
    left = _point(landmarks, 61, w, h)
    right = _point(landmarks, 291, w, h)
    v = _dist(top, bot)
    hz = _dist(left, right)
    if hz == 0:
        return 0.0
    return v / hz


def _mouth_metrics(landmarks, w, h):
    lc = _point(landmarks, 61, w, h)
    rc = _point(landmarks, 291, w, h)
    ul = _point(landmarks, 13, w, h)
    ll = _point(landmarks, 14, w, h)
    lf = _point(landmarks, 234, w, h)
    rf = _point(landmarks, 454, w, h)
    fw = _dist(lf, rf)
    mw = _dist(lc, rc)
    if fw == 0:
        return 0.0, 0.0
    smile = mw / fw
    lip_cy = (ul[1] + ll[1]) / 2
    corner_cy = (lc[1] + rc[1]) / 2
    droop = (corner_cy - lip_cy) / fw
    return smile, droop


def _classify(smile, droop, mar, ear):
    if mar > 0.55 and ear > 0.25:
        return "SURPRISED_OR_BIG_YAWN"
    if mar > 0.45:
        return "MOUTH_OPEN_OR_YAWN"
    if smile > 0.42 and droop < 0.03:
        return "POSITIVE_SMILE_LIKE"
    if smile < 0.38 and droop > 0.015:
        return "SAD_LIKE"
    return "NEUTRAL_FOCUSED"


def _face_position(landmarks, w):
    xs = [lm.x * w for lm in landmarks]
    cx = sum(xs) / len(xs)
    if cx < w / 3:
        return "left"
    if cx > 2 * w / 3:
        return "right"
    return "center"

# ---------------------------------------------------------------------------
# Robot Red Bull handoff
# ---------------------------------------------------------------------------

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _robot_cfg():
    return SO100FollowerConfig(
        port=ROBOT_PORT,
        id="webcam_robot",
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=ROBOT_CAMERA,
                width=640, height=480, fps=30,
                backend=Cv2Backends.V4L2,
            ),
        },
        use_degrees=True,
        max_relative_target=30.0,
    )


def _do_redbull_handoff(person_position):
    global _redbull_running, _can_is_full, _last_redbull_time

    now = time.time()
    if _redbull_running:
        return
    if now - _last_redbull_time < ACTION_COOLDOWN_SECONDS:
        return
    _last_redbull_time = now
    _redbull_running = True

    print(f"[ROBOT] Red Bull handoff → {person_position}")

    try:
        if not REDBULL_CALIB_FILE.exists():
            print("[ROBOT] No calibration file. Skipping.")
            return

        calib = _load_json(REDBULL_CALIB_FILE)
        hover_points = calib["hover_points"]
        delta = calib["grasp_delta"]
        lift_pos = calib["lift_position"]
        neutral = hover_points[0]["joints"].copy()
        neutral["gripper.pos"] = GRIPPER_OPEN

        robot = SO100Follower(_robot_cfg())
        robot.connect()
        print("[ROBOT] Connected.")

        def _act(joints, dur=0):
            robot.send_action(joints)
            if dur > 0:
                time.sleep(dur)

        def _get_j():
            obs = robot.get_observation()
            return {k: v for k, v in obs.items() if k.endswith(".pos")}

        def _interp(target, dur):
            start = _get_j()
            steps = max(1, int(dur * 30))
            for i in range(1, steps + 1):
                if not _running:
                    return
                t = i / steps
                interp = {}
                for k in target:
                    interp[k] = start.get(k, target[k]) + (target[k] - start.get(k, target[k])) * t
                robot.send_action(interp)
                time.sleep(dur / steps)

        # 1. Neutral
        _interp(neutral, 1.5)
        if not _running: return

        # 2. Detect Red Bull on robot cam
        obs = robot.get_observation()
        frame = obs["front"].copy()
        results = _yolo(frame, verbose=False)[0]
        detections = []
        for det in results.boxes:
            if int(det.cls[0]) == REDBULL_CLASS_ID and float(det.conf[0]) >= YOLO_CONF:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                detections.append(((x1 + x2) // 2, (y1 + y2) // 2))

        if not detections:
            print("[ROBOT] No Red Bull can detected.")
            robot.disconnect()
            return

        cx, cy = detections[0]
        # nearest hover point
        best = min(hover_points, key=lambda p: (p["pixel"][0]-cx)**2 + (p["pixel"][1]-cy)**2)
        hover = best["joints"].copy()
        hover["gripper.pos"] = GRIPPER_OPEN

        # 3. Hover above can
        _interp(hover, 1.0)
        if not _running: return

        # 4. Grasp
        grasp = _get_j()
        for k, v in delta.items():
            if k != "gripper.pos":
                grasp[k] = grasp.get(k, 0) + v
        grasp["gripper.pos"] = GRIPPER_OPEN
        _interp(grasp, 1.0)
        if not _running: return

        grasp["gripper.pos"] = GRIPPER_CLOSED
        _act(grasp, 0.6)
        if not _running: return

        # 5. Lift
        _interp(lift_pos, 1.0)
        if not _running: return

        print("[ROBOT] Can grabbed. Delivering...")

        # 6. Deliver
        delivery = _get_j()
        offsets = DELIVERY_OFFSETS.get(person_position, DELIVERY_OFFSETS["center"])
        for k, v in offsets.items():
            jk = f"{k}.pos"
            if jk in delivery:
                delivery[jk] += v
        _interp(delivery, 1.5)
        if not _running: return

        # 7. Release
        delivery["gripper.pos"] = GRIPPER_OPEN
        _act(delivery, 0.8)

        print(f"[ROBOT] Delivered to {person_position}!")
        _can_is_full = False

        # 8. Return to neutral
        _interp(neutral, 1.5)

        robot.disconnect()
        print("[ROBOT] Disconnected.")

    except Exception as e:
        print(f"[ROBOT] Error: {e}")
    finally:
        _redbull_running = False

# ---------------------------------------------------------------------------
# Main webcam loop
# ---------------------------------------------------------------------------

def main():
    global _running, _can_is_full, _yolo

    print("=== Laptop Webcam → Robot ===")
    print("F = toggle can full/empty")
    print("R = manual Red Bull test")
    print("Q = quit\n")

    # ── Open laptop webcam ──
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Camera {CAMERA_INDEX} not found.")
        return
    print(f"[CAM] Opened camera {CAMERA_INDEX}")

    # ── Load models ──
    face_landmarker = _create_face_landmarker()
    _yolo = YOLO("yolov8n.pt")

    # ── Fatigue state ──
    perclos_window = deque(maxlen=180)
    blink_times = deque(maxlen=100)
    yawn_times = deque(maxlen=30)
    eyes_closed_start = None
    yawn_start = None
    was_eyes_closed = False
    app_start = time.time()

    last_expr = "UNKNOWN"
    expr_candidate = "UNKNOWN"
    expr_candidate_start = None

    confirmed_state = "IDLE"
    state_candidate = "IDLE"
    state_candidate_start = None
    CONFIRM_S = 0.4

    person_position = "center"

    try:
        while _running:
            ret, frame = cap.read()
            if not ret:
                print("Lost camera frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            now = time.time()
            runtime = now - app_start

            # ── MediaPipe ──
            mp_img = Image(image_format=ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = face_landmarker.detect(mp_img)

            face_detected = False
            landmarks = None
            ear = mar = smile = droop = None
            fatigue_status = "NO_FACE"
            fatigue_score = 0
            expr_cue = "UNKNOWN"
            raw_state = "IDLE"
            eyes_closed_dur = 0.0
            perclos = 0.0

            if result.face_landmarks:
                face_detected = True
                landmarks = result.face_landmarks[0]
                person_position = _face_position(landmarks, w)

                le = _ear(landmarks, LEFT_EYE, w, h)
                re = _ear(landmarks, RIGHT_EYE, w, h)
                ear = (le + re) / 2.0
                mar = _mar(landmarks, w, h)

                eyes_closed = ear < EAR_CLOSED
                mouth_open = mar > MAR_YAWN

                # Eye closure
                if eyes_closed:
                    if eyes_closed_start is None:
                        eyes_closed_start = now
                    eyes_closed_dur = now - eyes_closed_start
                else:
                    if was_eyes_closed and eyes_closed_start is not None:
                        d = now - eyes_closed_start
                        if 0.05 <= d <= 0.7:
                            blink_times.append(now)
                    eyes_closed_start = None
                    eyes_closed_dur = 0.0
                was_eyes_closed = eyes_closed

                # Yawn
                if mouth_open:
                    if yawn_start is None:
                        yawn_start = now
                else:
                    if yawn_start is not None:
                        if now - yawn_start >= YAWN_MIN_S:
                            yawn_times.append(now)
                    yawn_start = None

                # PERCLOS
                perclos_window.append(1 if eyes_closed else 0)
                perclos = sum(perclos_window) / max(1, len(perclos_window))

                while blink_times and now - blink_times[0] > 60:
                    blink_times.popleft()
                while yawn_times and now - yawn_times[0] > 60:
                    yawn_times.popleft()

                blink_rate = (len(blink_times) * 60.0 / min(60.0, max(1.0, runtime))
                              if runtime > 10 else 0)
                yawns_last_min = len(yawn_times)

                # Expression
                smile, droop = _mouth_metrics(landmarks, w, h)
                raw_expr = _classify(smile, droop, mar, ear)
                if raw_expr != expr_candidate:
                    expr_candidate = raw_expr
                    expr_candidate_start = now
                if expr_candidate_start and now - expr_candidate_start >= CONFIRM_S:
                    last_expr = expr_candidate
                expr_cue = last_expr

                # Fatigue score
                score = 0
                cd = eyes_closed_dur
                if cd >= SLEEPING_100_S:   score = 100
                elif cd >= SLEEPING_LIKE_S: score += 95
                elif cd >= MICROSLEEP_S:    score += 85
                elif cd >= VERY_TIRED_S:    score += 70
                elif cd >= TIRED_S:         score += 45

                if perclos > 0.80:      score += 60
                elif perclos > 0.45:    score += 45
                elif perclos > 0.30:    score += 30
                elif perclos > 0.18:    score += 15
                if yawns_last_min >= 2: score += 30
                elif yawns_last_min == 1: score += 18
                if ear < 0.23:          score += 10
                if runtime > 30 and blink_rate < 5: score += 8

                fatigue_score = min(score, 100)

                # Classification
                if cd >= SLEEPING_100_S:
                    fatigue_status = "SLEEPING_100_DEMO"
                elif cd >= SLEEPING_LIKE_S:
                    fatigue_status = "SLEEPING_LIKE"
                elif cd >= MICROSLEEP_S:
                    fatigue_status = "MICROSLEEP_RISK"
                elif cd >= VERY_TIRED_S:
                    fatigue_status = "VERY_TIRED"
                elif cd >= TIRED_S:
                    fatigue_status = "TIRED"
                elif fatigue_score >= 70:
                    fatigue_status = "VERY_TIRED"
                elif fatigue_score >= 35:
                    fatigue_status = "TIRED"
                else:
                    fatigue_status = "NOT_TIRED"

                # Robot state
                if fatigue_status in ("SLEEPING_100_DEMO",):
                    raw_state = "SLEEPING_100_DEMO"
                elif fatigue_status in ("SLEEPING_LIKE",):
                    raw_state = "SLEEPING_LIKE"
                elif fatigue_status in ("MICROSLEEP_RISK",):
                    raw_state = "MICROSLEEP_RISK"
                elif fatigue_status in ("VERY_TIRED",):
                    raw_state = "VERY_TIRED"
                elif fatigue_status in ("TIRED",):
                    raw_state = "TIRED"
                elif expr_cue == "SAD_LIKE":
                    raw_state = "SAD_LIKE"
                elif expr_cue == "POSITIVE_SMILE_LIKE":
                    raw_state = "POSITIVE"
                else:
                    raw_state = "IDLE"

            # ── Stabilize state ──
            if raw_state != state_candidate:
                state_candidate = raw_state
                state_candidate_start = now
            if state_candidate_start and now - state_candidate_start >= CONFIRM_S:
                confirmed_state = state_candidate
            robot_state = confirmed_state

            # ── Trigger Red Bull ──
            if (face_detected and _can_is_full
                    and robot_state in REDBULL_TRIGGER_STATES
                    and not _redbull_running):
                print(f"[TRIGGER] {robot_state} → Red Bull handoff ({person_position})")
                t = threading.Thread(target=_do_redbull_handoff,
                                     args=(person_position,), daemon=True)
                t.start()

            # ── Save JSON state (for other consumers) ──
            state = {
                "timestamp": now,
                "robot_state": robot_state,
                "fatigue_status": fatigue_status,
                "fatigue_score": fatigue_score,
                "expression_cue": expr_cue,
                "face_detected": face_detected,
                "person_position": person_position,
                "ear": ear, "mar": mar,
                "perclos": perclos,
                "eyes_closed_duration": eyes_closed_dur,
                "can_is_full": _can_is_full,
                "redbull_running": _redbull_running,
            }
            with open("emotion_detection.json", "w") as f:
                json.dump(state, f, indent=2)

            # ── Overlay ──
            if robot_state in ("SLEEPING_100_DEMO", "SLEEPING_LIKE", "VERY_TIRED", "SAD_LIKE"):
                color = (0, 0, 255)
            elif robot_state in ("MICROSLEEP_RISK", "TIRED"):
                color = (0, 165, 255)
            elif robot_state == "POSITIVE":
                color = (0, 255, 0)
            else:
                color = (255, 255, 255)

            if face_detected and landmarks:
                xs = [int(lm.x * w) for lm in landmarks]
                ys = [int(lm.y * h) for lm in landmarks]
                x1, y1 = max(0, min(xs)-20), max(0, min(ys)-20)
                x2, y2 = min(w-1, max(xs)+20), min(h-1, max(ys)+20)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"State: {robot_state}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
            cv2.putText(frame, f"Fatigue: {fatigue_status} | Score: {fatigue_score}/100",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if fatigue_status in ("SLEEPING_100_DEMO","SLEEPING_LIKE","VERY_TIRED") else (255,255,255), 2)
            cv2.putText(frame, f"Expression: {expr_cue}",
                        (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if ear is not None:
                cv2.putText(frame, f"EAR: {ear:.3f} | MAR: {mar:.3f}",
                            (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Eyes closed: {eyes_closed_dur:.1f}s | PERCLOS: {perclos:.2f}",
                        (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Can: {'FULL' if _can_is_full else 'EMPTY'} | Handoff: {person_position}",
                        (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0) if _can_is_full else (255,255,255), 2)
            h_status = "HANDOFF" if _redbull_running else "READY"
            cv2.putText(frame, f"Robot: {h_status}",
                        (30, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,0,255) if _redbull_running else (0,255,0), 2)
            cv2.putText(frame, "F:can toggle | R:manual handoff | Q:quit",
                        (30, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,180,0), 2)

            cv2.imshow("Laptop Webcam → Robot", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("f"):
                _can_is_full = not _can_is_full
                print(f"[SYSTEM] can_is_full = {_can_is_full}")
            elif key == ord("r"):
                if not _redbull_running:
                    t = threading.Thread(target=_do_redbull_handoff,
                                         args=(person_position,), daemon=True)
                    t.start()

    except KeyboardInterrupt:
        pass
    finally:
        face_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\nBye!")


if __name__ == "__main__":
    main()
