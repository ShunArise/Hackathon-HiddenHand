#!/usr/bin/env python3
"""
Le Robot Controller:
  1. SWEEP  – swift left-to-right scanning for a human face
  2. WATCH  – face found, centre on it, run emotion recognition
  3. FETCH  – if the dude is tired → go to static Red Bull position → grab
  4. BRING  – deliver the can toward the corner of the human's direction
  5. CLEAN  – run ACT policy to grab the empty can (trash) and dispose of it

Usage (from hackathon root):
  /home/lennart/projects/hackathon/lerobot/.venv/bin/python le_robot_controller.py
"""
import signal
import sys
import time
import json
import math
import threading
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve

# Allow running from repo root (lerobot package lives in lerobot/src)
_lerobot_root = Path(__file__).resolve().parent / "lerobot" / "src"
if str(_lerobot_root) not in sys.path:
    sys.path.insert(0, str(_lerobot_root))

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe import Image, ImageFormat
from ultralytics import YOLO

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.configs import Cv2Backends
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig

import torch
from lerobot.policies.act import ACTPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.configs import PreTrainedConfig

# ---------------------------------------------------------------------------
# MediaPipe face landmarker model (auto-download on first run)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Emotion helpers (same landmark-index logic as emotion.py)
# ---------------------------------------------------------------------------

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _dist(p1, p2):
    return math.dist(p1, p2)


def _point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
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


def mouth_aspect_ratio(landmarks, w, h):
    top = _point(landmarks, 13, w, h)
    bot = _point(landmarks, 14, w, h)
    left = _point(landmarks, 61, w, h)
    right = _point(landmarks, 291, w, h)
    v = _dist(top, bot)
    hz = _dist(left, right)
    if hz == 0:
        return 0.0
    return v / hz


def face_center(landmarks, w, h):
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    return sum(xs) // len(xs), sum(ys) // len(ys)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDBULL_CLASS_ID = 39
PERSON_CLASS_ID = 0
YOLO_CONF = 0.5

EAR_CLOSED   = 0.20
MAR_YAWN     = 0.55
TIRED_EYES_S = 1.0         # seconds of closed eyes → TIRED
VERY_TIRED_S = 2.0
MICROSLEEP_S = 4.0

SWEEP_CALIB_FILE  = Path.home() / ".cache" / "lerobot_sweep_calib.json"
REDBULL_CALIB_FILE = Path.home() / ".cache" / "lerobot_redbull_calib.json"
PHONE_CALIB_FILE   = Path.home() / ".cache" / "lerobot_phone_calib.json"

CAMERA_INDEX = "/dev/video2"
ROBOT_PORT   = "/dev/ttyACM0"
LEADER_PORT  = "/dev/ttyACM1"
W, H, FPS    = 640, 480, 30
LOOK_UP_TILT = {"elbow_flex.pos": 0.0, "wrist_flex.pos": 0.0}  # degrees added to tilt camera up

STAND_UP_POSE = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -80.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": -30.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 60.0,
}

ACT_CHECKPOINT   = "outputs/train/act_see_trash_grab/checkpoints/last/pretrained_model"
ACT_CLEAN_STEPS  = 150       # ~5 seconds at 30 fps
ACT_CLEAN_TASK   = "grab trash on the table"
ACT_CAMERA_INDEX = "/dev/video0"   # separate camera for ACT/CLEAN state (e.g. overhead table view)

# ---------------------------------------------------------------------------
# Robot / camera setup
# ---------------------------------------------------------------------------

robot_cfg = SO100FollowerConfig(
    port=ROBOT_PORT,
    id="le_robot_follower",
    cameras={
        "front": OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX, width=W, height=H, fps=FPS,
            backend=Cv2Backends.V4L2,
        )
    },
    use_degrees=True,
)
leader_cfg = SO100LeaderConfig(port=LEADER_PORT, id="le_robot_leader", use_degrees=True)

robot = SO100Follower(robot_cfg)
leader = SO100Leader(leader_cfg)
yolo = YOLO("yolov8n.pt")

bus_lock = threading.Lock()
running = True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs():
    with bus_lock:
        return robot.get_observation()


def _act(joints):
    with bus_lock:
        robot.send_action(joints)


def _get_joints():
    obs = _obs()
    return {k: v for k, v in obs.items() if k.endswith(".pos")}


def _joints_copy(obs=None):
    if obs is None:
        obs = _obs()
    return {k: v for k, v in obs.items() if k.endswith(".pos")}


def _sleep(s, granularity=0.05):
    """Interruptible sleep."""
    t0 = time.time()
    while time.time() - t0 < s and running:
        time.sleep(granularity)


def _move_to(joints, duration=1.0):
    _act(joints)
    _sleep(duration)


def _centre_error(frame, cx, cy):
    """Pixel error from image centre – positive x → target is RIGHT of centre."""
    fh, fw = frame.shape[:2]
    return cx - fw / 2, cy - fh / 2


# ---------------------------------------------------------------------------
# Calibration persistence
# ---------------------------------------------------------------------------

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if isinstance(x, (int, float)) else x)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sweep calibration  (left / centre / right)
# ---------------------------------------------------------------------------

def calibrate_sweep():
    print("\n=== Sweep Calibration ===")
    print("Use the LEADER arm to position the robot for each of the 3 sweep poses.")
    print("Press Enter to start teleop, then press 's' when ready to save each pose.")

    robot.connect()
    leader.connect()

    teleop_running = True

    def _teleop():
        while teleop_running and running:
            try:
                action = leader.get_action()
            except Exception:
                break
            _act(action)
            time.sleep(1 / 30)

    t = threading.Thread(target=_teleop, daemon=True)
    t.start()

    poses = {}
    for name in ("LEFT", "CENTRE", "RIGHT"):
        input(f"\nMove leader to **{name}** sweep position. Press Enter when ready, then 's' to save...")
        while True:
            frame = _obs()["front"].copy()
            cv2.putText(frame, f"Position arm for {name} – press 's' to save", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Sweep Calibration", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                poses[name.lower()] = _get_joints()
                print(f"  Saved {name}: {list(poses[name.lower()].keys())[:3]}...")
                break
            if key == ord("q"):
                teleop_running = False
                t.join(timeout=1)
                cv2.destroyAllWindows()
                leader.disconnect()
                robot.disconnect()
                return

    teleop_running = False
    t.join(timeout=1)

    cv2.destroyAllWindows()
    leader.disconnect()

    save_json(SWEEP_CALIB_FILE, poses)
    print(f"Sweep calibration saved → {SWEEP_CALIB_FILE}")
    robot.disconnect()


# ---------------------------------------------------------------------------
# Person / face detection on the robot camera
# ---------------------------------------------------------------------------

def detect_person_yolo(frame):
    """Return list of (cx, cy, bbox) for detected persons (upper half of frame only)."""
    fh, fw = frame.shape[:2]
    results = yolo(frame, verbose=False)[0]
    persons = []
    for det in results.boxes:
        if int(det.cls[0]) == PERSON_CLASS_ID and float(det.conf[0]) >= YOLO_CONF:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if cy < fh * 0.55:
                persons.append((cx, cy, (x1, y1, x2, y2)))
    return persons


def detect_face_mediapipe(frame, face_landmarker):
    """Return (landmarks, cx, cy) or None.  landmarks is a list of NormalizedLandmark."""
    mp_image = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = face_landmarker.detect(mp_image)
    if result.face_landmarks:
        lm = result.face_landmarks[0]
        cx, cy = face_center(lm, W, H)
        return lm, cx, cy
    return None


# ---------------------------------------------------------------------------
# Emotion / fatigue analysis (ported from emotion.py)
# ---------------------------------------------------------------------------

def analyse_face(landmarks, w, h, now, state):
    """Update rolling state dict, return fatigue_status string."""
    left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    ear = (left_ear + right_ear) / 2.0
    mar = mouth_aspect_ratio(landmarks, w, h)

    eyes_closed = ear < EAR_CLOSED
    mouth_open  = mar > MAR_YAWN

    # eye closure tracking
    if eyes_closed:
        if state["eyes_closed_start"] is None:
            state["eyes_closed_start"] = now
        state["eyes_closed_duration"] = now - state["eyes_closed_start"]
    else:
        if state["was_eyes_closed"] and state["eyes_closed_start"] is not None:
            dur = now - state["eyes_closed_start"]
            if 0.05 <= dur <= 0.7:
                state["blink_times"].append(now)
        state["eyes_closed_start"] = None
        state["eyes_closed_duration"] = 0.0
    state["was_eyes_closed"] = eyes_closed

    # yawn tracking
    if mouth_open:
        if state["yawn_start"] is None:
            state["yawn_start"] = now
        state["yawn_duration"] = now - state["yawn_start"]
    else:
        if state["yawn_start"] is not None:
            if now - state["yawn_start"] >= 0.7:
                state["yawn_times"].append(now)
        state["yawn_start"] = None
        state["yawn_duration"] = 0.0

    # PERCLOS
    state["perclos_window"].append(1 if eyes_closed else 0)
    perclos = sum(state["perclos_window"]) / max(1, len(state["perclos_window"]))

    # clean old events
    while state["blink_times"] and now - state["blink_times"][0] > 60:
        state["blink_times"].popleft()
    while state["yawn_times"] and now - state["yawn_times"][0] > 60:
        state["yawn_times"].popleft()

    blink_rate = len(state["blink_times"]) * (60.0 / max(1.0, now - state["app_start"])) if now - state["app_start"] > 10 else 0
    yawns_last_min = len(state["yawn_times"])

    closed_dur = state["eyes_closed_duration"]

    # fatigue score
    score = 0
    if closed_dur >= MICROSLEEP_S:
        score += 85
    elif closed_dur >= VERY_TIRED_S:
        score += 70
    elif closed_dur >= TIRED_EYES_S:
        score += 45

    if perclos > 0.80:       score += 60
    elif perclos > 0.45:     score += 45
    elif perclos > 0.30:     score += 30
    elif perclos > 0.18:     score += 15

    if yawns_last_min >= 2:  score += 30
    elif yawns_last_min == 1: score += 18
    if ear < 0.23:           score += 10
    if blink_rate < 5:       score += 8
    score = min(score, 100)

    # classification
    if closed_dur >= MICROSLEEP_S:
        status = "MICROSLEEP_RISK"
    elif closed_dur >= VERY_TIRED_S:
        status = "VERY_TIRED"
    elif closed_dur >= TIRED_EYES_S:
        status = "TIRED"
    elif score >= 70:
        status = "VERY_TIRED"
    elif score >= 35:
        status = "TIRED"
    else:
        status = "NOT_TIRED"

    return status, ear, mar, perclos, score, blink_rate, closed_dur


# ---------------------------------------------------------------------------
# Red Bull fetch helpers
# ---------------------------------------------------------------------------

def _load_redbull_calib():
    try:
        return load_json(REDBULL_CALIB_FILE)
    except Exception:
        return None


def _interpolate_hover(pixel, hover_points):
    cx, cy = pixel
    best = min(hover_points, key=lambda p: (p["pixel"][0] - cx) ** 2 + (p["pixel"][1] - cy) ** 2)
    return best["joints"].copy()


def _grab_redbull():
    """Run the Red Bull grab sequence (must hold bus_lock externally where needed)."""
    calib = _load_redbull_calib()
    if calib is None:
        print("[REDBULL] No Red Bull calibration found – skipping grab.")
        return False

    hover_points = calib["hover_points"]
    delta = calib["grasp_delta"]
    lift_pos = calib["lift_position"]

    # 1) move to a neutral / safe pose first (the first hover point is a good starting spot)
    #    then look for can
    for _ in range(6):
        frame = _obs()["front"]
    results = yolo(frame, verbose=False)[0]
    detections = []
    for det in results.boxes:
        if int(det.cls[0]) == REDBULL_CLASS_ID and float(det.conf[0]) >= YOLO_CONF:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            detections.append(((x1 + x2) // 2, (y1 + y2) // 2, (x1, y1, x2, y2)))

    if not detections:
        print("[REDBULL] No Red Bull can in view.")
        return False

    cx, cy, _ = detections[0]
    hover = _interpolate_hover((cx, cy), hover_points)
    hover["gripper.pos"] = 40.0
    _act(hover)
    _sleep(1.0)
    if not running:
        return False

    grasp = _joints_copy()
    for key in delta:
        if key != "gripper.pos":
            grasp[key] = grasp.get(key, 0) + delta[key]
    grasp["gripper.pos"] = 40.0
    _act(grasp)
    _sleep(1.0)
    if not running:
        return False

    grasp["gripper.pos"] = 60.0   # close gripper
    _act(grasp)
    _sleep(0.6)

    _act(lift_pos)
    _sleep(1.0)
    print("[REDBULL] Can grabbed.")
    return True


# ---------------------------------------------------------------------------
# Deliver Red Bull toward the human's last-known direction
# ---------------------------------------------------------------------------

def _deliver_to_human(human_cx, human_cy):
    """Move arm toward the corner where the human was seen."""
    calib = _load_redbull_calib()
    if calib is None:
        print("[DELIVER] No calibration – holding position.")
        return

    hover_points = calib["hover_points"]
    target = _interpolate_hover((human_cx, human_cy), hover_points)
    target["gripper.pos"] = 60.0   # keep gripping
    _act(target)
    _sleep(2.0)

    # release
    target["gripper.pos"] = 40.0
    _act(target)
    _sleep(0.8)
    print("[DELIVER] Red Bull delivered!")


# ---------------------------------------------------------------------------
# Visual feedback overlay
# ---------------------------------------------------------------------------

def draw_overlay(frame, info):
    y = 30
    for label, value in info:
        cv2.putText(frame, f"{label}: {value}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y += 28


# ---------------------------------------------------------------------------
# Main state machine
# ---------------------------------------------------------------------------

def main():
    global running

    mode = input("1 = calibrate sweep, 2 = calibrate Red Bull, 3 = RUN\nChoice [1/2/3]: ").strip()

    if mode == "1":
        calibrate_sweep()
        return
    elif mode == "2":
        print("Run:  cd lerobot && python redbull_grabber.py  → choose 1 (calibrate grasp)")
        return

    # --- RUN MODE ---

    robot.connect()

    if not SWEEP_CALIB_FILE.exists():
        print("[CALIB] No sweep calibration found – auto-calibrating from current pose...")
        obs = robot.get_observation()
        centre_pose = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
        left_pose = centre_pose.copy()
        right_pose = centre_pose.copy()
        left_pose["shoulder_pan.pos"] -= 180.0
        right_pose["shoulder_pan.pos"] += 180.0
        sweep = {"left": left_pose, "centre": centre_pose, "right": right_pose}
        save_json(SWEEP_CALIB_FILE, sweep)
        print(f"[CALIB] Auto-calibrated. Centre: {centre_pose['shoulder_pan.pos']:.1f}°, "
              f"Left: {left_pose['shoulder_pan.pos']:.1f}°, Right: {right_pose['shoulder_pan.pos']:.1f}°")
    else:
        sweep = load_json(SWEEP_CALIB_FILE)

    left_pose   = sweep["left"]
    centre_pose = sweep["centre"]
    right_pose  = sweep["right"]

    for pose in (left_pose, centre_pose, right_pose):
        for joint, offset in LOOK_UP_TILT.items():
            if joint in pose:
                pose[joint] += offset

    # ── ACT policy for CLEAN state ────────────────────────────────────
    act_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act_policy = None
    act_preprocessor = None
    act_postprocessor = None
    act_dataset_features = None

    act_checkpoint_path = Path(__file__).resolve().parent / "lerobot" / ACT_CHECKPOINT
    if act_checkpoint_path.exists():
        print("[ACT] Loading policy for CLEAN state...")
        act_config = PreTrainedConfig.from_pretrained(act_checkpoint_path)
        act_policy = ACTPolicy.from_pretrained(act_checkpoint_path, config=act_config)
        act_policy.to(act_device)
        act_policy.eval()
        act_preprocessor, act_postprocessor = make_pre_post_processors(
            policy_cfg=act_config,
            pretrained_path=act_checkpoint_path,
            preprocessor_overrides={"device_processor": {"device": str(act_device)}},
        )
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        act_dataset_features = {**action_features, **obs_features}
        print("[ACT] Policy loaded.")
    else:
        print(f"[ACT] WARNING: checkpoint not found at {act_checkpoint_path} – CLEAN state disabled")

    # ── Separate camera for ACT/CLEAN state ───────────────────────────
    act_cap = None
    if ACT_CAMERA_INDEX != CAMERA_INDEX:
        act_cap = cv2.VideoCapture(ACT_CAMERA_INDEX)
        act_cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        act_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        act_cap.set(cv2.CAP_PROP_FPS, FPS)
        if not act_cap.isOpened():
            print(f"[ACT] WARNING: could not open ACT camera {ACT_CAMERA_INDEX}, falling back to robot camera")
            act_cap = None
        else:
            print(f"[ACT] Using separate camera {ACT_CAMERA_INDEX} for CLEAN state")

    face_landmarker = _create_face_landmarker()

    # rolling emotion state
    emo_state = {
        "app_start": time.time(),
        "eyes_closed_start": None,
        "eyes_closed_duration": 0.0,
        "was_eyes_closed": False,
        "yawn_start": None,
        "yawn_duration": 0.0,
        "perclos_window": deque(maxlen=180),
        "blink_times": deque(maxlen=100),
        "yawn_times": deque(maxlen=30),
    }

    STATE = "SWEEP"           # SWEEP | WATCH | FETCH | BRING | CLEAN
    sweep_dir = 1             # 1 = left→right, -1 = right→left
    current_pose = left_pose.copy()
    sweep_target = right_pose.copy()

    fatigue_status = "UNKNOWN"
    ear_val = mar_val = perclos_val = score_val = 0.0
    blinks = closed_dur = 0.0
    tired_confirmed_count = 0
    TIRED_CONFIRM_NEEDED = 3

    human_last_cx = W // 2
    human_last_cy = H // 2
    lost_face_time = None
    FACE_LOST_TIMEOUT = 2.0

    print("\n=== Le Robot ACTIVE ===")
    print("Sweeping for humans... Press 'q' to quit.\n")

    try:
        while running:
            obs = _obs()
            frame = obs["front"].copy()
            now = time.time()

            persons = detect_person_yolo(frame)
            face_data = detect_face_mediapipe(frame, face_landmarker)

            has_human = len(persons) > 0 or face_data is not None
            human_cx, human_cy = W // 2, H // 2

            if face_data is not None:
                _, human_cx, human_cy = face_data
                lost_face_time = None
            elif len(persons) > 0:
                human_cx, human_cy, _ = persons[0]
                lost_face_time = None
            else:
                if lost_face_time is None:
                    lost_face_time = now

            # ── STATE MACHINE ──────────────────────────────────────────
            if STATE == "SWEEP":
                for key in current_pose:
                    if key != "gripper.pos":
                        current_pose[key] += (sweep_target[key] - current_pose[key]) * 0.05
                current_pose["gripper.pos"] = 25.0
                _act(current_pose)

                dist_sum = sum(abs(current_pose[k] - sweep_target[k])
                               for k in current_pose if k != "gripper.pos")
                if dist_sum < 1.5:
                    sweep_dir *= -1
                    sweep_target = right_pose.copy() if sweep_dir == 1 else left_pose.copy()

                if has_human:
                    STATE = "WATCH"
                    emo_state["app_start"] = now
                    tired_confirmed_count = 0
                    print("[STATE] → WATCH – human detected")

            elif STATE == "WATCH":
                err_x, err_y = _centre_error(frame, human_cx, human_cy)

                watch_pose = centre_pose.copy()
                for key in watch_pose:
                    if "shoulder" in key.lower() or "rotation" in key.lower():
                        watch_pose[key] += err_x * 0.005
                        break
                for key in watch_pose:
                    if "shoulder" in key.lower() or "lift" in key.lower():
                        if key != list(watch_pose.keys())[0]:
                            watch_pose[key] += err_y * 0.003
                            break
                watch_pose["gripper.pos"] = 25.0
                _act(watch_pose)

                if face_data is not None:
                    landmarks, _, _ = face_data
                    fatigue_status, ear_val, mar_val, perclos_val, score_val, blinks, closed_dur = \
                        analyse_face(landmarks, W, H, now, emo_state)
                    human_last_cx, human_last_cy = human_cx, human_cy

                    if fatigue_status in ("TIRED", "VERY_TIRED", "MICROSLEEP_RISK",
                                          "SLEEPING_100_DEMO", "SLEEPING_LIKE"):
                        tired_confirmed_count += 1
                    else:
                        tired_confirmed_count = 0

                    if tired_confirmed_count >= TIRED_CONFIRM_NEEDED:
                        STATE = "FETCH"
                        print(f"[STATE] → FETCH – human is {fatigue_status}")
                elif lost_face_time and now - lost_face_time > FACE_LOST_TIMEOUT:
                    STATE = "SWEEP"
                    print("[STATE] → SWEEP – face lost")

            elif STATE == "FETCH":
                calib = _load_redbull_calib()
                if calib is None:
                    print("[FETCH] No Red Bull calib – returning to SWEEP")
                    STATE = "SWEEP"
                    continue

                rb_neutral = calib["hover_points"][0]["joints"].copy()
                rb_neutral["gripper.pos"] = 40.0
                _move_to(rb_neutral, duration=1.5)

                success = _grab_redbull()
                if success:
                    _move_to(STAND_UP_POSE, duration=1.5)
                    _sleep(1.0)
                    STATE = "BRING"
                else:
                    STATE = "SWEEP"
                print(f"[STATE] → {STATE}")

            elif STATE == "BRING":
                _deliver_to_human(human_last_cx, human_last_cy)
                if act_policy is not None:
                    STATE = "CLEAN"
                    clean_start_time = time.time()
                    clean_step_count = 0
                    print("[STATE] → CLEAN – running ACT to grab the empty can")
                else:
                    STATE = "SWEEP"
                    tired_confirmed_count = 0
                    print("[STATE] → SWEEP – delivery complete (no ACT model loaded)")

            elif STATE == "CLEAN":
                if clean_step_count >= ACT_CLEAN_STEPS:
                    STATE = "SWEEP"
                    tired_confirmed_count = 0
                    print("[STATE] → SWEEP – clean complete")
                    continue

                # Use separate ACT camera if available, otherwise fall back to robot camera
                if act_cap is not None:
                    ret, act_frame = act_cap.read()
                    if ret:
                        obs["front"] = act_frame

                act_policy.reset()
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=act_dataset_features,
                    device=act_device,
                    task=ACT_CLEAN_TASK,
                )
                with torch.inference_mode():
                    obs_frame = act_preprocessor(obs_frame)
                    action = act_policy.select_action(obs_frame)
                    action = act_postprocessor(action)
                action = make_robot_action(action, act_dataset_features)
                _act(action)
                clean_step_count += 1

            # ── Overlay ─────────────────────────────────────────────
            info = [
                ("State", STATE),
                ("Fatigue", fatigue_status),
                ("Score", f"{score_val:.0f}/100"),
                ("EAR", f"{ear_val:.3f}"),
            ]
            if face_data is not None:
                for lm in face_data[0]:
                    px, py = int(lm.x * W), int(lm.y * H)
                    cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

            if has_human:
                cv2.circle(frame, (human_cx, human_cy), 12, (0, 255, 255), 3)

            cv2.line(frame, (W // 2 - 20, H // 2), (W // 2 + 20, H // 2), (100, 100, 100), 1)
            cv2.line(frame, (W // 2, H // 2 - 20), (W // 2, H // 2 + 20), (100, 100, 100), 1)

            draw_overlay(frame, info)
            if STATE == "CLEAN":
                cam_label = f"cam: {ACT_CAMERA_INDEX}" if act_cap is not None else "cam: robot"
                cv2.putText(frame, cam_label, (10, H - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.putText(frame, "Q = quit", (10, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("Le Robot", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(1 / FPS)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        cv2.destroyAllWindows()
        face_landmarker.close()
        if act_cap is not None:
            act_cap.release()
        robot.disconnect()


def _signal_handler(sig, frame):
    global running
    running = False
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)

if __name__ == "__main__":
    main()
