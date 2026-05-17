"""
功能：
  1. 连接 Piper 机械臂（CAN）+ D435i 相机
  2. 使能
  3. 画正方形，全程实时显示相机画面
  4. 回零位
  5. 相机继续显示，按 q 退出

使用方式：
  conda activate mujoco && python demos/real_arm.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pyrealsense2 as rs
from piper_sdk import C_PiperInterface_V2

PIPELINE = None
CAM_STOP = False


def start_camera():
    global PIPELINE
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    PIPELINE = pipeline
    print("D435i 相机已启动")


def show_camera_frame():
    global CAM_STOP
    if PIPELINE is None:
        return
    frames = PIPELINE.wait_for_frames()
    color_frame = frames.get_color_frame()
    if color_frame:
        image = np.asanyarray(color_frame.get_data())
        cv2.imshow("D435i", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        CAM_STOP = True


def stop_camera():
    global PIPELINE
    if PIPELINE is not None:
        PIPELINE.stop()
        PIPELINE = None
    cv2.destroyAllWindows()


def sleep_with_camera(seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        show_camera_frame()
        if CAM_STOP:
            return
        time.sleep(0.03)


def connect_and_enable(can_name="can0"):
    piper = C_PiperInterface_V2(can_name)
    piper.ConnectPort()
    time.sleep(0.2)
    while not piper.EnablePiper():
        time.sleep(0.01)
    print("机械臂使能成功")
    return piper


def draw_square(piper, speed=100):
    print("画正方形 - 移动到起点 (MOVEP)")
    piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
    piper.EndPoseCtrl(150000, -50000, 200000, -179900, 0, -179900)
    sleep_with_camera(2)
    if CAM_STOP:
        return

    print("画正方形 - 边1 (MOVEL)")
    piper.MotionCtrl_2(0x01, 0x02, speed, 0x00)
    piper.EndPoseCtrl(150000, 50000, 200000, -179900, 0, -179900)
    sleep_with_camera(2)
    if CAM_STOP:
        return

    print("画正方形 - 边2 (MOVEL)")
    piper.MotionCtrl_2(0x01, 0x02, speed, 0x00)
    piper.EndPoseCtrl(250000, 50000, 200000, -179900, 0, -179900)
    sleep_with_camera(2)
    if CAM_STOP:
        return

    print("画正方形 - 边3 (MOVEL)")
    piper.MotionCtrl_2(0x01, 0x02, speed, 0x00)
    piper.EndPoseCtrl(250000, -50000, 200000, -179900, 0, -179900)
    sleep_with_camera(2)
    if CAM_STOP:
        return

    print("画正方形 - 边4 回起点 (MOVEL)")
    piper.MotionCtrl_2(0x01, 0x02, speed, 0x00)
    piper.EndPoseCtrl(150000, -50000, 200000, -179900, 0, -179900)
    sleep_with_camera(2)
    print("正方形完成")


def go_zero(piper, speed=30):
    print("先抬高到安全位置 (MOVEP)")
    piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
    piper.EndPoseCtrl(150000, 0, 240000, -179900, 0, -179900)
    sleep_with_camera(3)
    if CAM_STOP:
        return

    print("回零位 (MOVEJ)")
    factor = 57295.7795
    joints = [0, 0, 0, 0, 0, 0]
    sdk_joints = [round(j * factor) for j in joints]
    piper.ModeCtrl(0x01, 0x01, speed, 0x00)
    piper.JointCtrl(*sdk_joints)
    piper.GripperCtrl(0, 1000, 0x01, 0)
    sleep_with_camera(3)
    print("回零完成")


def main():
    start_camera()

    piper = connect_and_enable("can0")

    if not CAM_STOP:
        draw_square(piper)
        sleep_with_camera(1)

    if not CAM_STOP:
        go_zero(piper)
        sleep_with_camera(1)

    print("运动完成，按 q 关闭相机")

    while not CAM_STOP:
        show_camera_frame()
        time.sleep(0.03)

    stop_camera()
    print("已退出")


if __name__ == "__main__":
    main()
