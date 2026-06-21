"""
Scripted Red Bull handoff for LeRobot SO-101 / SO-101t.

This file is intentionally pose-based: no recording is required.
You must tune POSES for your real desk setup.

Workflow:
1. Keep DRY_RUN = True first.
2. Run: python scripted_redbull.py
3. Tune POSES.
4. Set DRY_RUN = False only when the arm path is safe.
"""

import time
import platform
import threading

# -------------------------------------------------------------------
# LeRobot import
# -------------------------------------------------------------------
# Depending on your LeRobot version, the import path can differ.
# The first one is the expected SO-101 path. The others are fallbacks.
try:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
except Exception:
    try:
        from lerobot.common.robot_devices.robots.configs import So101RobotConfig as SO101FollowerConfig
        from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot as SO101Follower
    except Exception:
        SO101Follower = None
        SO101FollowerConfig = None


ROBOT_PORT = "COM3" if platform.system() == "Windows" else "/dev/ttyACM0"
# On Windows: check Device Manager → Ports to confirm COM number.
# On Linux: confirm with: ls /dev/ttyACM* after plugging in the arm.
ROBOT_ID = "my_awesome_follower_arm" # Must match the calibrated robot id.

# Keep True until the poses are calibrated and safe.
DRY_RUN = True

# Prevent two movements from running at the same time.
_action_lock = threading.Lock()


def make_action(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper):
    return {
        "shoulder_pan.pos": float(shoulder_pan),
        "shoulder_lift.pos": float(shoulder_lift),
        "elbow_flex.pos": float(elbow_flex),
        "wrist_flex.pos": float(wrist_flex),
        "wrist_roll.pos": float(wrist_roll),
        "gripper.pos": float(gripper),
    }


# -------------------------------------------------------------------
# IMPORTANT: these are only starter values.
# Tune them with your real robot, desk, can position and handoff zone.
# -------------------------------------------------------------------
POSES = {
    # Neutral/safe pose.
    "home": make_action(0, -40, 60, 0, 0, 75),

    # Stand up – arm fully upright, gripper very high.
    # wrist_flex (motor 4) is stretched (negative) so gripper points up.
    "stand_up": make_action(0, -80, 0, -30, 0, 85),

    # Red Bull fixed pickup zone.
    "above_can": make_action(-25, -30, 55, 10, 0, 85),
    "grab_can": make_action(-25, -12, 65, 10, 0, 85),
    "close_gripper": make_action(-25, -12, 65, 10, 0, 25),
    "lift_can": make_action(-25, -38, 55, 0, 0, 25),

    # Fixed handoff zones. The camera chooses left/center/right.
    "handoff_left": make_action(35, -30, 45, 10, 0, 25),
    "handoff_center": make_action(20, -30, 45, 10, 0, 25),
    "handoff_right": make_action(5, -30, 45, 10, 0, 25),

    # Release at the selected handoff zone.
    "release_left": make_action(35, -30, 45, 10, 0, 85),
    "release_center": make_action(20, -30, 45, 10, 0, 85),
    "release_right": make_action(5, -30, 45, 10, 0, 85),
}


def _connect_robot():
    if SO101Follower is None or SO101FollowerConfig is None:
        raise ImportError(
            "Could not import LeRobot SO-101 classes. Check your LeRobot installation and import path."
        )

    # Newer LeRobot style.
    try:
        config = SO101FollowerConfig(
            port=ROBOT_PORT,
            id=ROBOT_ID,
            max_relative_target=20,
        )
        robot = SO101Follower(config)
        robot.connect()
        return robot
    except TypeError:
        # Fallback for older API shapes.
        config = SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID)
        robot = SO101Follower(config)
        robot.connect()
        return robot


def move(robot, pose_name, sleep_s=1.0):
    action = POSES[pose_name]
    print(f"[ROBOT] {pose_name}: {action}")

    if not DRY_RUN:
        robot.send_action(action)

    time.sleep(sleep_s)


def give_redbull(person_position="center"):
    """
    Pick the Red Bull from a fixed position and offer it to a fixed handoff zone.

    person_position: "left", "center", or "right".
    This does not dynamically track the hand. It selects one calibrated handoff pose.
    """
    if person_position not in {"left", "center", "right"}:
        person_position = "center"

    if not _action_lock.acquire(blocking=False):
        print("[ROBOT] Movement already running. Ignored.")
        return

    robot = None

    try:
        print(f"[ROBOT] Starting Red Bull handoff: {person_position}")

        if not DRY_RUN:
            robot = _connect_robot()

        move(robot, "home", 1.0)
        move(robot, "above_can", 1.0)
        move(robot, "grab_can", 1.0)
        move(robot, "close_gripper", 0.8)
        move(robot, "lift_can", 1.0)
        move(robot, f"handoff_{person_position}", 1.5)

        print("[ROBOT] Waiting for the person to take the Red Bull...")
        time.sleep(2.0)

        move(robot, f"release_{person_position}", 0.8)
        move(robot, "home", 1.0)

        print("[ROBOT] Red Bull handoff finished.")

    finally:
        if not DRY_RUN and robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass

        _action_lock.release()


if __name__ == "__main__":
    give_redbull("center")
