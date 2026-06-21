"""
Small helper to print current SO-101 joint positions.
Use it to tune scripted_redbull.py POSES.
"""

import time
import platform

try:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
except Exception:
    SO101Follower = None
    SO101FollowerConfig = None

ROBOT_PORT = "COM3" if platform.system() == "Windows" else "/dev/ttyACM0"
ROBOT_ID = "my_awesome_follower_arm"


def main():
    if SO101Follower is None or SO101FollowerConfig is None:
        raise ImportError("Could not import SO101Follower. Check your LeRobot installation/import path.")

    config = SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID)
    robot = SO101Follower(config)
    robot.connect()

    try:
        while True:
            obs = robot.get_observation()
            pose = {
                key: round(float(value), 2)
                for key, value in obs.items()
                if key.endswith(".pos")
            }
            print(pose)
            time.sleep(0.5)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
