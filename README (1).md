# Hidden Hand

> **Autonomous fatigue detection, Red Bull delivery, can recycling, and an escalating safety net — powered by LeRobot SO-101.**

[![Demo Video](https://img.shields.io/badge/Watch-Demo%20Video-red)](YOUR_VIDEO_URL_HERE)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/ehtravail/desk_hero_can_to_bin)
[![HuggingFace Policy](https://img.shields.io/badge/HuggingFace-Policy-yellow)](https://huggingface.co/ehtravail/act_can_to_bin_150k)
[![LeRobot Hackathon 2026](https://img.shields.io/badge/LeRobot-Hackathon%202026-blue)](https://huggingface.co/lerobot)

---

## What it does

Hidden Hand is a fully autonomous desk assistant that detects when you're falling asleep, delivers a Red Bull, collects the empty can, and — if you *still* won't wake up — escalates to a human. A webcam tracks your face; when it detects sustained eye closure, yawning, or a drooping head, it classifies your state on a 6-level risk ladder and the SO-101 arm hands you a can. A wake-up melody plays while the arm moves. If you keep falling asleep, the system warns you with a spoken message, then alerts an emergency contact by email.

Everything runs **locally** — no cloud, no GPU, no training data required. The only outbound network step is the optional emergency email.

**The loop:**

```
1. DETECT    — webcam reads your face (EAR, MAR, PERCLOS, head pose)
2. PREVENT   — early spoken nudge before it gets dangerous
3. DELIVER   — arm fetches a Red Bull and hands it to you
4. WAKE UP   — configurable melody plays to ease you back to alertness
5. WEIGH     — arm "feels" the can via servo current to confirm a real grasp
6. COLLECT   — press B; arm picks up the empty can and bins it
7. ESCALATE  — still asleep after 3 cans? spoken alert + emergency email
```

---

## Who it's for

- **Security workers** — monitoring feeds through the night, unable to leave their post
- **Students** — homework, assignments, hackathons; the deadline doesn't care if you're tired
- **Night medical staff, long-haul drivers, firefighters (alertness training)** — anyone whose vigilance has real stakes
- **Developers, hobbyists, remote workers** — anyone whose job demands hours of sustained focus

---

## How it works

```
Webcam ─► MediaPipe FaceMesh ─► Fatigue classifier ─► SO-101 arm ─► Red Bull delivered
            (478 landmarks)        (6-level ladder)      (scripted poses + weight check)
                                          │
                    ┌─────────────────────┼──────────────────────┐
                    ▼                     ▼                      ▼
             TIRED / VERY_TIRED    SLEEPING_LIKE         after 3 Red Bulls
             spoken prevention     music alarm           spoken emergency
             message               (music only)          + alert email
```

The detection writes a live `emotion_detection.json` state file every frame, which any external system (dashboard, Slack, etc.) can read. On escalation it also writes `emergency.json` and sends the alert email directly via SMTP.

---

## Files

| File | Purpose |
|------|---------|
| `emotion_detection_redbull.py` | **Main app.** Webcam, fatigue detection, robot trigger, music, TTS, escalation, emergency email. Launch this. |
| `scripted_redbull.py` | Robot handoff + bin sequence + **weight sensing**. |
| `read_pose.py` | Calibration helper. Prints live joint positions. |
| `very_tired_music.mp3` | Wake-up melody played on deep-sleep states. |
| `emotion_detection.json` | Live state file (auto-generated at runtime). |
| `emergency.json` | Escalation alert (auto-generated when escalation fires). |

---

## Requirements

- Python 3.10–3.12
- A webcam
- An SO-101 / SO-101t robotic arm (optional — detection works without it)
- LeRobot installed and the arm calibrated

### Install dependencies

```bash
pip install "mediapipe==0.10.14" pygame opencv-python numpy pyttsx3
```

> **MediaPipe version matters.** Versions 0.10.15+ removed the `mp.solutions` API. You must use **0.10.14** — the app refuses to start and tells you if the wrong version is installed.

> **pyttsx3** uses Windows SAPI natively (no extra install on Windows). On Linux also run `apt install espeak-ng`. If unavailable the app keeps running — TTS is silently disabled.

### Install LeRobot

```bash
pip install 'lerobot[all]'
```

Verify:
```bash
python -c "from lerobot.robots.so101_follower import SO101Follower; print('OK')"
```

---

## Setup

### 1. Same folder

`emotion_detection_redbull.py`, `scripted_redbull.py`, and `very_tired_music.mp3` must be in the **same directory**. If they're split, the robot import fails loudly and the arm never moves.

### 2. Find your robot port

```bash
# Linux
ls /dev/ttyACM*
lerobot-find-port

# Windows: Device Manager → Ports (COM3, COM4, etc.)
```

Port auto-detects (`COM3` on Windows, `/dev/ttyACM0` on Linux). Update `ROBOT_PORT` in `scripted_redbull.py` and `read_pose.py` only if yours differs.

### 3. Set your robot ID

```python
# scripted_redbull.py and read_pose.py
ROBOT_ID = "my_awesome_follower_arm"   # ← match your LeRobot calibration ID
```

### 4. Calibrate the delivery poses

Run the pose reader, physically move the arm to each position, copy the joint angles:

```bash
python read_pose.py
```

Tune in `POSES` inside `scripted_redbull.py`:

| Pose | What it is |
|------|-----------|
| `home` | Neutral/safe resting pose |
| `above_can` | Hovering above the full can |
| `grab_can` | Lowered to grip height |
| `close_gripper` | Gripper closed on can |
| `lift_can` | Can lifted |
| `weigh_pose` | Fixed pose for current/weight reading |
| `handoff_{left,center,right}` | Delivery zones |
| `release_{left,center,right}` | Gripper open at each zone |

### 5. Calibrate the bin poses

Same process — tune the empty-can collection sequence:

| Pose | What it is |
|------|-----------|
| `above_empty_can` | Hover above the return zone |
| `grab_empty_can` | Lower to grip height |
| `grip_empty_can` | Gripper closed on empty can |
| `lift_empty_can` | Can lifted |
| `above_bin` | Swung over the bin |
| `drop_in_bin` | Gripper open → can falls in |

### 6. Calibrate weight sensing (optional but recommended)

In the **WEIGHT-SENSING CONFIG** block of `scripted_redbull.py`, hold the arm at the weigh pose and note the servo current with no can and with a full can. Set `I_BASELINE`, `WEIGHT_BUCKETS`, and `GRIPPER_EMPTY_POS`. Takes ~15 minutes.

### 7. Enable the arm

```python
# scripted_redbull.py
DRY_RUN = False   # arm does nothing until you set this to False
```

> Keep `DRY_RUN = True` until you've verified the full arm path. In dry-run mode the app logs pose names but sends no servo commands.

### 8. Emergency email (optional)

```bash
export HIDDENHAND_EMAIL_USER="youraddress@gmail.com"
export HIDDENHAND_EMAIL_PASSWORD="your16charAppPassword"   # Gmail App Password
```

Get the App Password at https://myaccount.google.com/apppasswords. The recipient defaults to `ehtravail@gmail.com` — change `EMAIL_CONFIG["recipient"]` in the main app. If no password is set, the escalation still fires (writes `emergency.json`, speaks the alert) — it just skips the email.

---

## Running

```bash
python emotion_detection_redbull.py
```

On startup you'll see:
```
[ROBOT] scripted_redbull.py imported OK — arm is armed.
[TTS] pyttsx3 ready.
```
or diagnostic blocks telling you exactly what's missing.

---

## Controls

| Key | Action |
|-----|--------|
| `R` | **Manual Red Bull trigger** — bypasses detection and cooldown. Best for demos. |
| `B` | **Bin the empty can** — put the empty in the return zone, then press B. The arm picks it up and drops it in the bin. |
| `F` | Toggle can full/empty (mark as refilled after a handoff). |
| `M` | **Mute/unmute** — silences both the alarm music and the spoken voice. |
| `X` | **Reset episode** — clears the drink count and re-arms escalation ("restart after 3 times"). |
| `P` | **Toggle prevention** — enables/disables the early spoken fatigue nudges. |
| `Q` | Quit. |

### On-screen HUD

```
Drinks: 1/3 | Voice: ON | Prevention: ON
Can full: True | Handoff: center | Red Bull: NO
Q quit | F can | R test | B bin | M mute | X reset | P prev
```

- `Voice: ON` — TTS available and active
- `Voice: MUTED` — `M` was pressed
- `Voice: N/A` — pyttsx3 not installed
- `Drinks` turns red when the escalation limit is reached

---

## The fatigue ladder

Fatigue is classified primarily on continuous eye-closure duration. These are **behavioral webcam states, not medical diagnoses.**

| State | Eyes closed | Effect |
|-------|-------------|--------|
| `NOT_TIRED` | — | Idle |
| `TIRED` | 1 s | Prevention voice |
| `VERY_TIRED` | 2 s | Prevention voice |
| `MICROSLEEP_RISK` | 4 s | Triggers arm |
| `SLEEPING_LIKE` | 8 s | Arm + music (no voice) |
| `SLEEPING_100_DEMO` | 10 s | Arm + music (no voice) |

> **Demo vs. production:** 8 s / 10 s are tuned short for a live demo. For real deployment raise these to ≈20–30 s and ≈40–60 s. Commented in the code.

**Signals used:** EAR · MAR · PERCLOS · head pitch/yaw/roll · blink rate · yawn count.

---

## Voice messages (text-to-speech)

The danger tier (SLEEPING_LIKE) is **intentionally silent** — the music and the physical arm are enough. Voice is reserved for early prevention and the emergency, where it adds information without feeling aggressive.

### Prevention tier — calm (TIRED / VERY_TIRED)
18-second cooldown, rotates randomly:
- *"You're getting tired. Consider a short break."*
- *"Look away from the screen for 20 seconds."*
- *"Time to stretch — your eyes need a rest."*
- *"Early fatigue detected. Take a breath."*
- *"Try the 20-20-20 rule: look 20 feet away for 20 seconds."*

### Emergency tier — urgent (after 3 cans, still asleep)
Bypasses cooldown, fires immediately:
- *"Emergency. You have not woken up. An alert has been sent to your emergency contact."*
- *"Critical alert. You did not respond. Help has been notified."*
- *"Maximum escalation reached. Your emergency contact has been informed."*

`M` silences both tiers. `P` toggles only prevention (emergency voice is always active unless `M`).

---

## Configurable music timing

```python
MUSIC_CONFIG = {
    "start_rule": "on_danger",   # "on_danger" | "delay" | "after_redbull"
    "delay_seconds": 3.0,
}
```

- **`on_danger`** — music starts the moment deep sleep is detected (default)
- **`delay`** — waits `delay_seconds` before playing
- **`after_redbull`** — music only starts once a can has been delivered

---

## Weight sensing — the robot feels the can

**The physics:** Feetech STS3215 servos report present current over the LeRobot bus. Motor torque is roughly proportional to current (**τ ≈ Kₜ·I**), so at a fixed holding pose the extra current the arm draws is proportional to payload mass. The app reads current at the canonical `weigh_pose`, subtracts `I_BASELINE`, and maps the delta to a bucket.

**Two signals used together:**
1. **Gripper closure** — if the gripper closes almost fully it grabbed air; if it stops wider, something's between the fingers.
2. **Holding current** — classifies: `EMPTY` / `CAN_FULL` / `TOO_HEAVY`.

**Sense → classify → act:**
- Missed grasp → retry once
- Too heavy → abort for safety
- Good can → proceed with the handoff

**Both paths share the same weigh routine:**
- **Scripted** — built into `give_redbull()`
- **ACT / imitation learning** — call `weigh_after_grasp(robot)` after the policy grabs. Clean seam: **ACT = dexterity, scripted weigh = measurement.**

> **Honest limit:** hobby-servo current is noisy. This is **bucket-accurate, not gram-accurate.** Pitch it as "the robot feels how heavy it is," never "±1 gram." Always weigh at the same pose.

---

## Escalation

```python
ESCALATION_DRINK_LIMIT = 3
ESCALATION_STATES = ["SLEEPING_LIKE", "SLEEPING_100_DEMO"]
```

After 3 delivered cans with the person still in a deep-sleep state:
1. Writes `emergency.json`
2. Speaks the emergency message out loud
3. Sends an alert email via SMTP

Drink count and emergency latch auto-reset when the person wakes up. Press `X` to reset manually.

---

## Vision pipeline — why not YOLO? (recycling layer)

Stock YOLO's `bottle` class scores ~0 confidence on a Red Bull can viewed from directly above — COCO training images are all side-view. The recycling pipeline instead targets the **blue pull-tab**: flat on top, always visible from overhead, stable color signature.

**Pipeline:** HSV mask → saturation filter (excludes gripper plastic) → morphological closing (rejoins fragmented blobs) → Sobel edge check (metal rim vs. wood shadow) → temporal stabilization (N consecutive frames). Multiple cans get UUIDs, worked one by one.

**Known false positive:** paper fold shadows can occasionally pass all checks. Low priority — paper is never a sort target.

---

## Training

| Parameter | Value |
|---|---|
| Policy | ACT (Action Chunking Transformer) |
| Dataset | `ehtravail/desk_hero_can_to_bin` |
| Episodes | 100+ teleoperated demos |
| Camera | 1 × 640×480 @ 30fps |
| Steps | 100k → 150k |
| Training hardware | RunPod RTX 4090 / A100 |
| Runtime hardware | Local CPU — no cloud at inference |

---

## Architecture notes

- **Thread-safe robot calls** — arm runs in a daemon thread with lock + cooldown; camera loop never blocks.
- **Non-blocking TTS** — spoken messages run in a background thread with per-tier lock.
- **Non-blocking email** — SMTP send runs in a background thread with cooldown.
- **State debouncing** — fatigue states must hold 0.4 s before confirmed; kills single-frame false positives.
- **Graceful degradation** — if LeRobot, pygame, pyttsx3, SMTP, or weight telemetry are missing, detection keeps running and the JSON state file keeps writing. Only the affected feature is disabled.

---

## Troubleshooting

**Arm doesn't move:** press `R` and read the terminal — it prints `ROBOT_ACTION_AVAILABLE`, `can_is_full`, and the trigger result. Confirm `DRY_RUN = False` and that `python scripted_redbull.py` moves the arm standalone.

**Bin sequence doesn't work:** confirm the bin poses are calibrated (`above_empty_can`, `above_bin`, etc.) and `DRY_RUN = False`.

**Music doesn't play:** confirm `pygame` is installed and `very_tired_music.mp3` is in the same folder.

**Voice not working:** check the `[TTS]` line at startup. On Linux: `apt install espeak-ng` first. On Windows: `pip install pyttsx3` is all you need.

**Email doesn't arrive:** confirm `HIDDENHAND_EMAIL_USER` and `HIDDENHAND_EMAIL_PASSWORD` are set (Gmail App Password required). Check the terminal for `[EMAIL]` lines.

**Weight reads "unknown":** your LeRobot build may not expose servo current. The app falls back to gripper-closure automatically. Check if `read_pose.py` observations include `.current` or `.load` keys.

**Camera not found:** change `cv2.VideoCapture(0)` index to `1`.

**`mp.solutions` error on startup:** `pip install "mediapipe==0.10.14"`.

---

## HuggingFace

| Resource | Link |
|---|---|
| Training dataset | [ehtravail/desk_hero_can_to_bin](https://huggingface.co/datasets/ehtravail/desk_hero_can_to_bin) |
| ACT policy (100k) | [ehtravail/act_can_to_bin_100k](https://huggingface.co/ehtravail/act_can_to_bin_100k) |
| ACT policy (150k) | [ehtravail/act_can_to_bin_150k](https://huggingface.co/ehtravail/act_can_to_bin_150k) |

---

## Disclaimer

This is a hackathon demo. The fatigue classifier uses heuristic facial cues and is **not a medical device.** It does not diagnose sleep disorders or any health condition. The emergency-alert feature is a best-effort notification, not a guaranteed safety system. Don't rely on it for anything safety-critical.

---

*Built at the LeRobot / Berlin Robotics × AI Hackathon 2026 · Hidden Hand*
