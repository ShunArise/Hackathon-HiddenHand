import cv2
import time
import math
import json
import os
import platform
import sys
import select
import subprocess
import threading
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe import Image, ImageFormat

try:
    import pygame
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

try:
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
    ROBOT_AVAILABLE = True
except Exception:
    ROBOT_AVAILABLE = False


# ============================================================
# MediaPipe face landmarker model (auto-download on first run)
# ============================================================

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_PATH = Path.home() / ".cache" / "mediapipe" / "face_landmarker.task"


def _ensure_face_model():
    if not FACE_LANDMARKER_PATH.exists():
        FACE_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[MODEL] Downloading face_landmarker.task → {FACE_LANDMARKER_PATH} ...")
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


# ============================================================
# Utility functions
# ============================================================

def dist(p1, p2):
    return math.dist(p1, p2)


def point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def get_face_bbox(landmarks, w, h, padding=20):
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(w - 1, max(xs) + padding)
    y2 = min(h - 1, max(ys) + padding)

    return x1, y1, x2, y2


def draw_face_box(frame, landmarks, w, h, color, thickness=2):
    x1, y1, x2, y2 = get_face_bbox(landmarks, w, h, padding=20)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cv2.putText(
        frame,
        "FACE",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def draw_eye_circles(frame, landmarks, w, h, color, thickness=2):
    left_eye_points = [33, 160, 158, 133, 153, 144]
    right_eye_points = [362, 385, 387, 263, 373, 380]

    left_pts = np.array([point(landmarks, idx, w, h) for idx in left_eye_points], dtype=np.int32)
    lx, ly, lw, lh = cv2.boundingRect(left_pts)
    left_center = (lx + lw // 2, ly + lh // 2)
    left_radius = max(lw, lh) // 2 + 4
    cv2.circle(frame, left_center, left_radius, color, thickness)

    right_pts = np.array([point(landmarks, idx, w, h) for idx in right_eye_points], dtype=np.int32)
    rx, ry, rw, rh = cv2.boundingRect(right_pts)
    right_center = (rx + rw // 2, ry + rh // 2)
    right_radius = max(rw, rh) // 2 + 4
    cv2.circle(frame, right_center, right_radius, color, thickness)


def draw_detection_overlay(frame, landmarks, w, h, color):
    draw_face_box(frame, landmarks, w, h, color, thickness=2)
    draw_eye_circles(frame, landmarks, w, h, color, thickness=2)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    p1 = point(landmarks, eye_indices[0], w, h)
    p2 = point(landmarks, eye_indices[1], w, h)
    p3 = point(landmarks, eye_indices[2], w, h)
    p4 = point(landmarks, eye_indices[3], w, h)
    p5 = point(landmarks, eye_indices[4], w, h)
    p6 = point(landmarks, eye_indices[5], w, h)

    vertical_1 = dist(p2, p6)
    vertical_2 = dist(p3, p5)
    horizontal = dist(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, w, h):
    top_lip = point(landmarks, 13, w, h)
    bottom_lip = point(landmarks, 14, w, h)
    left_mouth = point(landmarks, 61, w, h)
    right_mouth = point(landmarks, 291, w, h)

    vertical = dist(top_lip, bottom_lip)
    horizontal = dist(left_mouth, right_mouth)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


def mouth_expression_metrics(landmarks, w, h):
    left_corner = point(landmarks, 61, w, h)
    right_corner = point(landmarks, 291, w, h)

    upper_lip = point(landmarks, 13, w, h)
    lower_lip = point(landmarks, 14, w, h)

    left_face = point(landmarks, 234, w, h)
    right_face = point(landmarks, 454, w, h)

    face_width = dist(left_face, right_face)
    mouth_width = dist(left_corner, right_corner)

    if face_width == 0:
        return 0.0, 0.0

    smile_ratio = mouth_width / face_width

    lip_center_y = (upper_lip[1] + lower_lip[1]) / 2
    corner_center_y = (left_corner[1] + right_corner[1]) / 2

    # In images, y increases downward.
    # Positive value = mouth corners lower than lip center.
    corner_droop = (corner_center_y - lip_center_y) / face_width

    return smile_ratio, corner_droop


def classify_expression(smile_ratio, corner_droop, mar, ear):
    """
    Rule-based visible-expression cue.
    This is not true emotion detection.
    """
    if mar > 0.55 and ear > 0.25:
        return "SURPRISED_OR_BIG_YAWN"

    if mar > 0.45:
        return "MOUTH_OPEN_OR_YAWN"

    if smile_ratio > 0.42 and corner_droop < 0.03:
        return "POSITIVE_SMILE_LIKE"

    if smile_ratio < 0.38 and corner_droop > 0.015:
        return "SAD_LIKE"

    return "NEUTRAL_FOCUSED"


def estimate_head_pose(landmarks, w, h):
    """
    Approximate head pose using six face landmarks.
    Returns pitch, yaw, roll in degrees.
    """
    image_points = np.array([
        point(landmarks, 1, w, h),      # nose tip
        point(landmarks, 152, w, h),    # chin
        point(landmarks, 33, w, h),     # left eye outer corner
        point(landmarks, 263, w, h),    # right eye outer corner
        point(landmarks, 61, w, h),     # left mouth corner
        point(landmarks, 291, w, h),    # right mouth corner
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, -63.6, -12.5),     # chin
        (-43.3, 32.7, -26.0),    # left eye
        (43.3, 32.7, -26.0),     # right eye
        (-28.9, -28.9, -24.1),   # left mouth
        (28.9, -28.9, -24.1),    # right mouth
    ], dtype=np.float64)

    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    try:
        success, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None, None, None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        return pitch, yaw, roll

    except Exception:
        return None, None, None


# ============================================================
# Audio functions
# ============================================================

def init_audio():
    if not AUDIO_AVAILABLE:
        print("[AUDIO] pygame is not installed. Music disabled.")
        return False

    try:
        pygame.mixer.init()
        print("[AUDIO] Audio initialized.")
        return True
    except Exception as e:
        print("[AUDIO] Could not initialize audio:", e)
        return False


def start_music(music_file):
    if not AUDIO_AVAILABLE:
        return

    if not os.path.exists(music_file):
        print(f"[AUDIO] Music file not found: {music_file}")
        return

    try:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(music_file)
            pygame.mixer.music.play(-1)
            print("[AUDIO] Sleep / very tired music started.")
    except Exception as e:
        print("[AUDIO] Could not play music:", e)


def stop_music():
    if not AUDIO_AVAILABLE:
        return

    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            print("[AUDIO] Music stopped.")
    except Exception as e:
        print("[AUDIO] Could not stop music:", e)


# ============================================================
# ACT model launcher
# ============================================================

_ACT_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "lerobot"
_ACT_SCRIPT_PATH = _ACT_SCRIPT_DIR / "run_act_model.py"

# Shared mutable state for the mouse callback (set by main, read by callback)
_btn_rect = [0, 0, 0, 0]   # x, y, w, h
_btn_clicked = False
_act_process = None


def _mouse_callback(event, x, y, flags, param):
    global _btn_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = _btn_rect
        if bx <= x <= bx + bw and by <= y <= by + bh:
            _btn_clicked = True


def launch_act_model():
    global _act_process
    if _act_process is not None and _act_process.poll() is None:
        print("[ACT] ACT model is already running.")
        return
    if not _ACT_SCRIPT_PATH.exists():
        print(f"[ACT] Script not found: {_ACT_SCRIPT_PATH}")
        return
    print(f"[ACT] Launching: {_ACT_SCRIPT_PATH}")
    _act_process = subprocess.Popen(
        [sys.executable, str(_ACT_SCRIPT_PATH)],
        cwd=str(_ACT_SCRIPT_DIR),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


# ============================================================
# Main app
# ============================================================

def main():
    global _btn_clicked

    while True:
        _btn_clicked = False
        _restart = False

        MUSIC_FILE = "very_tired_music.mp3"
        STATE_FILE = "emotion_detection.json"

        audio_ready = init_audio()
        music_is_playing = False

        robot = None
        tired_pos_sent = False
        if ROBOT_AVAILABLE:
            try:
                robot_cfg = SO100FollowerConfig(port="/dev/ttyACM0")
                robot = SO100Follower(robot_cfg)
                robot.connect()
                print("[ROBOT] Connected.")
            except Exception as e:
                print(f"[ROBOT] Could not connect: {e}")
                robot = None

        TIRED_POSITION = None
        TIRED_POSITION_FILE = "tired_position.json"

        if os.path.exists(TIRED_POSITION_FILE):
            with open(TIRED_POSITION_FILE) as f:
                TIRED_POSITION = json.load(f)
            print(f"[ROBOT] Loaded tired position from {TIRED_POSITION_FILE}")
        elif robot is not None:
            print("[ROBOT] No tired pose saved - type f+Enter to free arm, move it, then s+Enter to save")

        # Auto-detect first available camera
        cap = None
        for cam_idx in [3, 0, 1, 5]:
            if platform.system() == "Windows":
                c = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            else:
                c = cv2.VideoCapture(cam_idx)
            if c.isOpened():
                cap = c
                print(f"[CAMERA] Using camera index {cam_idx}")
                break
            c.release()

        if cap is None or not cap.isOpened():
            print("No camera found. Tried indices: 0, 1, 3, 4, 5")
            return


        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        # Thresholds
        EAR_CLOSED_THRESHOLD = 0.20
        MAR_YAWN_THRESHOLD = 0.55

        # Time thresholds
        # These are behavioral webcam states, not medical sleep stages.
        TIRED_SECONDS = 1.0
        VERY_TIRED_SECONDS = 2.0
        MICROSLEEP_SECONDS = 4.0
        SLEEPING_LIKE_SECONDS = 8.0
        SLEEPING_100_SECONDS = 10.0
        YAWN_MIN_SECONDS = 0.7

        # Rolling metrics
        perclos_window = deque(maxlen=180)
        blink_times = deque(maxlen=100)
        yawn_times = deque(maxlen=30)

        eyes_closed_start = None
        yawn_start = None
        was_eyes_closed = False

        app_start = time.time()

        # Stabilization
        last_expression = "UNKNOWN"
        expression_candidate = "UNKNOWN"
        expression_candidate_start = None
        EXPRESSION_CONFIRM_SECONDS = 0.4

        confirmed_robot_state = "IDLE"
        robot_candidate = "IDLE"
        robot_candidate_start = None
        ROBOT_CONFIRM_SECONDS = 0.4

        # Colors BGR
        WHITE = (255, 255, 255)
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        YELLOW = (0, 255, 255)
        BLUE = (255, 180, 0)
        ORANGE = (0, 165, 255)

        face_landmarker = _create_face_landmarker()

        cv2.namedWindow("Desk Hero V4 - Fatigue / Microsleep / Sleeping-like Detection")
        cv2.setMouseCallback("Desk Hero V4 - Fatigue / Microsleep / Sleeping-like Detection", _mouse_callback)

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Could not read camera frame.")
                    break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            mp_image = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = face_landmarker.detect(mp_image)

            now = time.time()
            runtime = now - app_start

            face_detected = False
            landmarks_for_drawing = None

            ear = None
            mar = None
            smile_ratio = None
            corner_droop = None

            pitch = None
            yaw = None
            roll = None

            perclos = 0.0
            blink_rate_per_min = 0.0
            yawns_last_min = 0

            fatigue_score = 0
            sleep_confidence = 0
            fatigue_status = "NO_FACE"
            expression_cue = "UNKNOWN"
            raw_robot_state = "IDLE"

            eyes_closed_duration = 0.0
            yawn_duration = 0.0

            if results.face_landmarks:
                face_detected = True
                landmarks = results.face_landmarks[0]
                landmarks_for_drawing = landmarks

                # -------------------------
                # EAR / MAR
                # -------------------------
                left_ear = eye_aspect_ratio(landmarks, left_eye, w, h)
                right_ear = eye_aspect_ratio(landmarks, right_eye, w, h)
                ear = (left_ear + right_ear) / 2.0

                mar = mouth_aspect_ratio(landmarks, w, h)

                eyes_closed = ear < EAR_CLOSED_THRESHOLD
                mouth_open = mar > MAR_YAWN_THRESHOLD

                # -------------------------
                # Eye closure duration + blink count
                # -------------------------
                if eyes_closed:
                    if eyes_closed_start is None:
                        eyes_closed_start = now
                    eyes_closed_duration = now - eyes_closed_start
                else:
                    if was_eyes_closed and eyes_closed_start is not None:
                        closed_duration = now - eyes_closed_start

                        # Normal blink: short closure
                        if 0.05 <= closed_duration <= 0.7:
                            blink_times.append(now)

                    eyes_closed_start = None
                    eyes_closed_duration = 0.0

                was_eyes_closed = eyes_closed

                # -------------------------
                # Yawn detection
                # -------------------------
                if mouth_open:
                    if yawn_start is None:
                        yawn_start = now
                    yawn_duration = now - yawn_start
                else:
                    if yawn_start is not None:
                        duration = now - yawn_start
                        if duration >= YAWN_MIN_SECONDS:
                            yawn_times.append(now)

                    yawn_start = None
                    yawn_duration = 0.0

                # -------------------------
                # PERCLOS
                # -------------------------
                perclos_window.append(1 if eyes_closed else 0)

                if len(perclos_window) > 0:
                    perclos = sum(perclos_window) / len(perclos_window)

                # Clean old blink/yawn events
                while blink_times and now - blink_times[0] > 60:
                    blink_times.popleft()

                while yawn_times and now - yawn_times[0] > 60:
                    yawn_times.popleft()

                if runtime >= 10:
                    blink_rate_per_min = len(blink_times) * (60.0 / min(60.0, runtime))

                yawns_last_min = len(yawn_times)

                # -------------------------
                # Head pose
                # -------------------------
                pitch, yaw, roll = estimate_head_pose(landmarks, w, h)

                looking_down = False
                looking_away = False

                if pitch is not None:
                    looking_down = pitch < -12 or pitch > 18

                if yaw is not None:
                    looking_away = abs(yaw) > 25

                # -------------------------
                # Expression cue
                # -------------------------
                smile_ratio, corner_droop = mouth_expression_metrics(landmarks, w, h)
                raw_expression = classify_expression(smile_ratio, corner_droop, mar, ear)

                if raw_expression != expression_candidate:
                    expression_candidate = raw_expression
                    expression_candidate_start = now

                if expression_candidate_start is not None:
                    if now - expression_candidate_start >= EXPRESSION_CONFIRM_SECONDS:
                        last_expression = expression_candidate

                expression_cue = last_expression

                # -------------------------
                # Sleeping / microsleep confidence
                # -------------------------
                # This is a demo-friendly behavioral scale.
                # It does not diagnose real sleep stages or deep sleep.
                if eyes_closed_duration >= SLEEPING_100_SECONDS:
                    sleep_confidence = 100
                elif eyes_closed_duration >= SLEEPING_LIKE_SECONDS:
                    sleep_confidence = 90
                elif eyes_closed_duration >= MICROSLEEP_SECONDS:
                    sleep_confidence = 70
                elif eyes_closed_duration >= VERY_TIRED_SECONDS:
                    sleep_confidence = 40
                elif eyes_closed_duration >= TIRED_SECONDS:
                    sleep_confidence = 20
                elif perclos > 0.80 and len(perclos_window) >= 90:
                    sleep_confidence = 90
                else:
                    sleep_confidence = 0

                # -------------------------
                # Advanced fatigue score
                # -------------------------
                fatigue_score = 0

                # Eye-closure duration is the strongest signal.
                if eyes_closed_duration >= SLEEPING_100_SECONDS:
                    fatigue_score = 100
                elif eyes_closed_duration >= SLEEPING_LIKE_SECONDS:
                    fatigue_score += 95
                elif eyes_closed_duration >= MICROSLEEP_SECONDS:
                    fatigue_score += 85
                elif eyes_closed_duration >= VERY_TIRED_SECONDS:
                    fatigue_score += 70
                elif eyes_closed_duration >= TIRED_SECONDS:
                    fatigue_score += 45

                # PERCLOS: percentage of recent frames where eyes are closed.
                if perclos > 0.80:
                    fatigue_score += 60
                elif perclos > 0.45:
                    fatigue_score += 45
                elif perclos > 0.30:
                    fatigue_score += 30
                elif perclos > 0.18:
                    fatigue_score += 15

                if yawns_last_min >= 2:
                    fatigue_score += 30
                elif yawns_last_min == 1:
                    fatigue_score += 18

                if ear < 0.23:
                    fatigue_score += 10

                if looking_down:
                    fatigue_score += 12

                if looking_away:
                    fatigue_score += 8

                if runtime >= 30 and blink_rate_per_min < 5:
                    fatigue_score += 8

                fatigue_score = min(fatigue_score, 100)

                # -------------------------
                # Status classification
                # -------------------------
                # Priority based on continuous eye closure first,
                # then fallback to combined fatigue score.
                if eyes_closed_duration >= SLEEPING_100_SECONDS:
                    fatigue_status = "SLEEPING_100_DEMO"
                    sleep_confidence = 100
                elif eyes_closed_duration >= SLEEPING_LIKE_SECONDS:
                    fatigue_status = "SLEEPING_LIKE"
                    sleep_confidence = max(sleep_confidence, 90)
                elif eyes_closed_duration >= MICROSLEEP_SECONDS:
                    fatigue_status = "MICROSLEEP_RISK"
                    sleep_confidence = max(sleep_confidence, 70)
                elif eyes_closed_duration >= VERY_TIRED_SECONDS:
                    fatigue_status = "VERY_TIRED"
                    sleep_confidence = max(sleep_confidence, 40)
                elif eyes_closed_duration >= TIRED_SECONDS:
                    fatigue_status = "TIRED"
                    sleep_confidence = max(sleep_confidence, 20)
                    print("need energieeee")
                elif fatigue_score >= 35:
                    fatigue_status = "TIRED"
                    print("need energieeee")
                else:
                    fatigue_status = "NOT_TIRED"
                    sleep_confidence = 0

                # -------------------------
                # Raw robot state priority
                # -------------------------
                if fatigue_status == "SLEEPING_100_DEMO":
                    raw_robot_state = "SLEEPING_100_DEMO"
                elif fatigue_status == "SLEEPING_LIKE":
                    raw_robot_state = "SLEEPING_LIKE"
                elif fatigue_status == "MICROSLEEP_RISK":
                    raw_robot_state = "MICROSLEEP_RISK"
                elif fatigue_status == "VERY_TIRED":
                    raw_robot_state = "VERY_TIRED"
                elif fatigue_status == "TIRED":
                    raw_robot_state = "TIRED"
                elif expression_cue == "SAD_LIKE":
                    raw_robot_state = "SAD_LIKE"
                elif expression_cue == "POSITIVE_SMILE_LIKE":
                    raw_robot_state = "POSITIVE"
                elif looking_away:
                    raw_robot_state = "DISTRACTED"
                else:
                    raw_robot_state = "IDLE"

            else:
                perclos_window.clear()
                raw_robot_state = "IDLE"
                fatigue_status = "NO_FACE"
                sleep_confidence = 0
                was_eyes_closed = False
                eyes_closed_start = None
                yawn_start = None

            # -------------------------
            # Stabilize robot state
            # -------------------------
            if raw_robot_state != robot_candidate:
                robot_candidate = raw_robot_state
                robot_candidate_start = now

            if robot_candidate_start is not None:
                if now - robot_candidate_start >= ROBOT_CONFIRM_SECONDS:
                    confirmed_robot_state = robot_candidate

            robot_state = confirmed_robot_state

            # -------------------------
            # Robot movement on TIRED
            # -------------------------
            if robot is not None:
                if robot_state == "TIRED" and not tired_pos_sent and TIRED_POSITION is not None:
                    try:
                        targets = dict(TIRED_POSITION)
                        targets["wrist_roll"] = robot.bus.read("Present_Position", "wrist_roll", normalize=True)
                        current_deg = robot.bus.sync_read("Present_Position", list(targets), normalize=True)

                        def _interp_move(motor, target_deg, steps=40, delay=0.08):
                            start = robot.bus.read("Present_Position", motor, normalize=True)
                            for i in range(1, steps + 1):
                                t = i / steps
                                robot.send_action({f"{motor}.pos": start + (target_deg - start) * t})
                                time.sleep(delay)

                        # 1. Lift shoulder_lift high first (safe approach from above)
                        safe_high = targets["shoulder_lift"] - 50.0
                        print(f"[ROBOT] Lifting shoulder_lift to {safe_high:.1f} deg (safe height)...")
                        _interp_move("shoulder_lift", safe_high)

                        # 2. Move remaining axes
                        for motor, target_deg in targets.items():
                            if motor == "shoulder_lift":
                                continue
                            print(f"[ROBOT] Moving {motor} to {target_deg:.1f} deg...")
                            _interp_move(motor, target_deg)

                        # 3. Lower shoulder_lift to target
                        print(f"[ROBOT] Lowering shoulder_lift to {targets['shoulder_lift']:.1f} deg...")
                        _interp_move("shoulder_lift", targets["shoulder_lift"])

                        # 4. Grab (inverted gripper: 0=closed 100=open)
                        print("[ROBOT] Grabbing...")
                        robot.bus.write("Torque_Limit", "gripper", 300, normalize=False)
                        grip_close = 10.0
                        _interp_move("gripper", grip_close, steps=20, delay=0.06)
                        robot.bus.write("Torque_Limit", "gripper", 1023, normalize=False)

                        # 5. Lift can up
                        print(f"[ROBOT] Lifting can to {safe_high:.1f} deg...")
                        _interp_move("shoulder_lift", safe_high)

                        # 6. Hold for 10s
                        print("[ROBOT] Holding for 10s...")
                        time.sleep(10)

                        # 7. Release (open gripper fully)
                        print("[ROBOT] Releasing...")
                        _interp_move("gripper", 100.0, steps=20, delay=0.06)

                        # 8. Return to home pose
                        print("[ROBOT] Returning to home...")
                        _interp_move("shoulder_lift", targets["shoulder_lift"])

                        print("[ROBOT] Tired pose + grab + release done")
                        tired_pos_sent = True
                    except Exception as e:
                        print(f"[ROBOT] Move failed: {e}")
                        try:
                            robot.bus.disable_torque(["gripper"])
                            robot.bus.enable_torque(["gripper"])
                            print("[ROBOT] Gripper torque reset")
                        except Exception:
                            pass
                elif robot_state != "TIRED":
                    tired_pos_sent = False

            # -------------------------
            # Music trigger
            # -------------------------
            DANGER_STATES = [
                #"VERY_TIRED",
                #"MICROSLEEP_RISK",
                "SLEEPING_LIKE",
                "SLEEPING_100_DEMO",
            ]

            if audio_ready and robot_state in DANGER_STATES and not music_is_playing:
                start_music(MUSIC_FILE)
                music_is_playing = True

            elif audio_ready and robot_state not in DANGER_STATES and music_is_playing:
                stop_music()
                music_is_playing = False

            # -------------------------
            # Save state for LeRobot
            # -------------------------
            state = {
                "timestamp": now,
                "version": "v4_risk_ladder",
                "face_detected": face_detected,
                "robot_state": robot_state,
                "raw_robot_state": raw_robot_state,
                "fatigue_status": fatigue_status,
                "sleep_confidence": sleep_confidence,
                "fatigue_score": fatigue_score,
                "expression_cue": expression_cue,
                "ear": ear,
                "mar": mar,
                "perclos": perclos,
                "blink_rate_per_min": blink_rate_per_min,
                "yawns_last_min": yawns_last_min,
                "eyes_closed_duration": eyes_closed_duration,
                "yawn_duration": yawn_duration,
                "smile_ratio": smile_ratio,
                "corner_droop": corner_droop,
                "head_pose": {
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll,
                },
                "music_playing": music_is_playing,
            }

            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)

            # -------------------------
            # Colors
            # -------------------------
            if robot_state == "SLEEPING_100_DEMO":
                state_color = RED
            elif robot_state == "SLEEPING_LIKE":
                state_color = RED
            elif robot_state == "MICROSLEEP_RISK":
                state_color = ORANGE
            elif robot_state == "VERY_TIRED":
                state_color = RED
            elif robot_state == "TIRED":
                state_color = ORANGE
            elif robot_state == "SAD_LIKE":
                state_color = RED
            elif robot_state == "POSITIVE":
                state_color = GREEN
            elif robot_state == "DISTRACTED":
                state_color = YELLOW
            else:
                state_color = WHITE

            if expression_cue == "SAD_LIKE":
                expression_color = RED
            elif expression_cue == "POSITIVE_SMILE_LIKE":
                expression_color = GREEN
            elif expression_cue in ["MOUTH_OPEN_OR_YAWN", "SURPRISED_OR_BIG_YAWN"]:
                expression_color = YELLOW
            else:
                expression_color = WHITE

            if fatigue_status in ["SLEEPING_100_DEMO", "SLEEPING_LIKE"]:
                fatigue_color = RED
            elif fatigue_status == "MICROSLEEP_RISK":
                fatigue_color = ORANGE
            elif fatigue_status == "VERY_TIRED":
                fatigue_color = RED
            elif fatigue_status == "TIRED":
                fatigue_color = ORANGE
            else:
                fatigue_color = WHITE

            music_color = RED if music_is_playing else WHITE

            if face_detected and landmarks_for_drawing is not None:
                draw_detection_overlay(frame, landmarks_for_drawing, w, h, state_color)

            # -------------------------
            # Overlay
            # -------------------------
            cv2.putText(
                frame,
                f"V3 Robot state: {robot_state}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                state_color,
                3,
            )

            cv2.putText(
                frame,
                f"Fatigue: {fatigue_status} | Score: {fatigue_score}/100",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                fatigue_color,
                2,
            )

            if robot_state == "SLEEPING_100_DEMO":
                cv2.putText(
                    frame,
                    "SLEEPING 100% DEMO",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    RED,
                    3,
                )
            elif robot_state == "SLEEPING_LIKE":
                cv2.putText(
                    frame,
                    "SLEEPING-LIKE STATE",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    RED,
                    3,
                )
            elif robot_state == "MICROSLEEP_RISK":
                cv2.putText(
                    frame,
                    "MICROSLEEP RISK",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    ORANGE,
                    3,
                )
            else:
                cv2.putText(
                    frame,
                    f"Sleep confidence: {sleep_confidence}%",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    WHITE,
                    2,
                )

            cv2.putText(
                frame,
                f"Expression cue: {expression_cue}",
                (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                expression_color,
                2,
            )

            cv2.putText(
                frame,
                f"Music: {'ON' if music_is_playing else 'OFF'}",
                (30, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                music_color,
                2,
            )

            if ear is not None and mar is not None:
                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f} | MAR: {mar:.3f}",
                    (30, 230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    WHITE,
                    2,
                )

            cv2.putText(
                frame,
                f"PERCLOS: {perclos:.2f} | Blinks/min: {blink_rate_per_min:.1f} | Yawns/min: {yawns_last_min}",
                (30, 265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                WHITE,
                2,
            )

            cv2.putText(
                frame,
                f"Eyes closed: {eyes_closed_duration:.1f}s",
                (30, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                RED if eyes_closed_duration >= SLEEPING_100_SECONDS else ORANGE if eyes_closed_duration >= MICROSLEEP_SECONDS else WHITE,
                2,
            )

            if pitch is not None and yaw is not None and roll is not None:
                cv2.putText(
                    frame,
                    f"Head pose pitch/yaw/roll: {pitch:.1f} / {yaw:.1f} / {roll:.1f}",
                    (30, 335),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    WHITE,
                    2,
                )

            if smile_ratio is not None and corner_droop is not None:
                cv2.putText(
                    frame,
                    f"Smile: {smile_ratio:.3f} | Droop: {corner_droop:.3f}",
                    (30, 370),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    WHITE,
                    2,
                )

            cv2.putText(
                frame,
                "Q=quit | f=free s=save (in terminal + Enter)",
                (30, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BLUE,
                2,
            )

            btn_w_px, btn_h_px = 180, 40
            btn_x = w - btn_w_px - 20
            btn_y = h - 60
            _btn_rect[:] = [btn_x, btn_y, btn_w_px, btn_h_px]

            is_running = _act_process is not None and _act_process.poll() is None
            btn_color = (0, 200, 0) if not is_running else (0, 120, 255)
            btn_text = "Run ACT Model" if not is_running else "ACT Running..."

            cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w_px, btn_y + btn_h_px), btn_color, -1)
            cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w_px, btn_y + btn_h_px), (255, 255, 255), 2)
            (tw, th), _ = cv2.getTextSize(btn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, btn_text,
                        (btn_x + (btn_w_px - tw) // 2, btn_y + (btn_h_px + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow("Desk Hero V4 - Fatigue / Microsleep / Sleeping-like Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if _btn_clicked:
                _btn_clicked = False
                _restart = True
                cv2.destroyAllWindows()
                launch_act_model()
                break

            if select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == "q":
                    break
                if cmd == "f" and robot is not None:
                    robot.bus.disable_torque()
                    print("[ROBOT] Torque OFF - move arm freely, then type s+Enter to save")
                if cmd == "s":
                    if robot is None:
                        print("[ROBOT] Cannot save - robot not connected")
                    else:
                        try:
                            robot.bus.enable_torque()
                            motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "gripper"]
                            pos = robot.bus.sync_read("Present_Position", motors, normalize=True)
                            TIRED_POSITION = dict(pos)
                            with open(TIRED_POSITION_FILE, "w") as f:
                                json.dump(TIRED_POSITION, f, indent=2)
                            print(f"[ROBOT] Saved tired position (deg): {TIRED_POSITION}")
                            tired_pos_sent = False
                        except Exception as e:
                            print(f"[ROBOT] Save failed: {e}")

    except KeyboardInterrupt:
        return
    finally:
        stop_music()
        if robot is not None:
            try:
                robot.disconnect()
                print("[ROBOT] Disconnected.")
            except Exception:
                pass
        face_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    if _restart and _act_process is not None:
        print("[ACT] Waiting for ACT model to finish...")
        _act_process.wait()
        _act_process = None
        print("[ACT] ACT model finished. Restarting emotion detection...")
    elif not _restart:
        break


if __name__ == "__main__":
    main()