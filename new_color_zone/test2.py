#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

# ========================================
# ======= LiDAR + Vision 통합 노드 =======
# ========================================

class Phase1Navigator(Node):
    def __init__(self) -> None:
        super().__init__('phase1_navigator')

        # --- ROS2 구독자 설정 ---
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.subscription_camera = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )

        # --- 초기 변수 ---
        self.bridge = CvBridge()
        self.lidar_command: str = 'F'
        self.vision_command: str = 'F'
        self.yellow_detected: bool = False
        self.hold_mode: bool = False
        self.last_hold_time: float = 0.0

        # --- 파라미터 ---
        self.safe_distance: float = 1.2   # 라이다 임계 거리 (m)
        self.turn_time: float = 1.5       # 우회전 시간 (s)
        self.forward_time: float = 2.0    # 직진 시간 (s)

        self.get_logger().info('Phase1 Navigator Node Initialized.')

    # ------------------------------
    # 1️⃣ LiDAR 콜백
    # ------------------------------
    def lidar_callback(self, msg: LaserScan) -> None:
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isnan(ranges), np.inf, ranges)

        left = np.min(ranges[len(ranges)//2 + 40: len(ranges)//2 + 200])
        right = np.min(ranges[len(ranges)//2 - 200: len(ranges)//2 - 40])
        front = np.min(ranges[len(ranges)//2 - 40: len(ranges)//2 + 40])

        if front < self.safe_distance:
            if left > right:
                self.lidar_command = 'L'
            else:
                self.lidar_command = 'R'
        else:
            self.lidar_command = 'F'

    # ------------------------------
    # 2️⃣ 카메라 콜백 (OpenCV 기반)
    # ------------------------------
    def camera_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- HSV 범위 설정 ---
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        lower_red1 = np.array([0, 120, 100])
        upper_red1 = np.array([5, 255, 255])
        lower_red2 = np.array([170, 120, 100])
        upper_red2 = np.array([180, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # --- 노란 부표 감지 ---
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_yellow:
            max_yellow = max(contours_yellow, key=cv2.contourArea)
            if cv2.contourArea(max_yellow) > 2000 and not self.hold_mode:
                self.hold_mode = True
                self.last_hold_time = time.time()
                self.get_logger().info('🟡 Yellow Buoy Detected! Holding position...')
                self.set_motor_state('S')
                return

        # --- 위치 유지 중이면 ---
        if self.hold_mode:
            if time.time() - self.last_hold_time < 5:
                self.set_motor_state('S')
                return
            else:
                self.get_logger().info('✅ Hold complete. Turning right...')
                self.set_motor_state('R')
                time.sleep(self.turn_time)
                self.get_logger().info('⬆️ Going forward...')
                self.set_motor_state('F')
                time.sleep(self.forward_time)
                self.hold_mode = False
                return

        # --- 초록 & 빨강 게이트 인식 ---
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_green and contours_red:
            cx_green = np.mean([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2 for c in contours_green])
            cx_red = np.mean([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2 for c in contours_red])
            gate_center = (cx_green + cx_red) / 2
            frame_center = frame.shape[1] / 2
            diff = gate_center - frame_center

            if abs(diff) < 50:
                self.vision_command = 'F'
            elif diff > 0:
                self.vision_command = 'R'
            else:
                self.vision_command = 'L'
        else:
            self.vision_command = 'F'

        # --- 최종 명령: LiDAR 우선 ---
        if self.lidar_command in ['L', 'R'] and not self.hold_mode:
            self.set_motor_state(self.lidar_command)
        else:
            self.set_motor_state(self.vision_command)

        # --- 디버그 화면 ---
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

    # ------------------------------
    # 3️⃣ 모터 상태 전송 (테스트용 print)
    # ------------------------------
    def set_motor_state(self, command: str) -> None:
        if command == 'F':
            action = "Forward"
        elif command == 'L':
            action = "Turn Left"
        elif command == 'R':
            action = "Turn Right"
        elif command == 'S':
            action = "Stop"
        else:
            action = "Unknown"
        self.get_logger().info(f"🚗 Command: {action}")

# ------------------------------
# 메인 실행
# ------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = Phase1Navigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
