#
# Reset SBX UVC camera USB interface
#
# use this script to reset USB interface of SBX UVC camera
# when no data is fed or device is unresponsive.
#
# Usage: change TARGET_VID, TARGET_PID (specify usb device's VID and PID) and
#        run this script with root privileges.
#
# @author An Jung-In <jian@fssolution.co.kr>
#

import os
import time
from typing import Union


def try_read(path: str) -> Union[str, None]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return f.read().strip()


def find_uvc(target_vid: str = "04b4", target_pid: str = "00f9") -> str:
    device_tree_root = "/sys/bus/usb/devices"
    for device_folder in os.listdir(device_tree_root):
        device_folder = os.path.join(device_tree_root, device_folder)

        device_pid = try_read(os.path.join(device_folder, "idProduct"))
        device_vid = try_read(os.path.join(device_folder, "idVendor"))

        if device_pid == target_pid and device_vid == target_vid:
            return device_folder
    return None


def reset_usb(device_tree_path: str, sleep_time: int = 3) -> bool:
    device_authorized_path = os.path.join(device_tree_path, "authorized")
    if not os.path.exists(device_authorized_path):
        raise RuntimeError(f"Device autorized file does not exist: {device_tree_path}")

    try:
        with open(device_authorized_path, "w") as f:
            f.write("0")
        print(f"Waiting for {sleep_time} seconds for kernel to apply ...")
        time.sleep(sleep_time)

        with open(device_authorized_path, "w") as f:
            f.write("1")
        print(f"Waiting for another {sleep_time} seconds for kernel to apply ...")
        time.sleep(sleep_time)

    except PermissionError:
        print(
            "ERROR: run this command with sudo. (No additional pip library required!)"
        )
        return False

    return True


if __name__ == "__main__":
    TARGET_VID = "04b4"
    TARGET_PID = "00f9"

    device_tree_path = find_uvc(target_vid=TARGET_VID, target_pid=TARGET_PID)
    if not device_tree_path:
        print(
            f"ERROR: Device with target VID=0x{TARGET_VID}, PID=0x{TARGET_PID} not found!"
        )

    print(f"Trying to reset device {device_tree_path} ...")
    if not reset_usb(device_tree_path):
        print("Failed resetting device.")
    else:
        print("Done.")
