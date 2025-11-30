#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import serial
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

# ======================
# 설정
# ======================
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
MODEL_PATH = 'docking.pt'
CONFIDENCE_THRESHOLD = 0.5

COLOR_W, COLOR_H = 640, 480
DEADZONE = 80
FORWARD_THRESHOLD = 220
FORWARD_TIME = 0.25
TURN_TIME = 0.12

TARGET_NAME = 'red_triangle'

SEARCH_TIMEOUT = 3.0   # 최근 본 타깃 기준 유지 시간
SCAN_INTERVAL = 1.5    # 스캔 주기
SCAN_TIME = 0.6        # 회전 유지 시간

# ======================
# 모터 클래스
# ======================
class Motor:
    def __init__(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)
            self.ser.write(b'7')
            time.sleep(0.05)
            self.ser.write(b'x')
            print("✅ 모터 연결 완료")
        except:
            self.ser = None
            print("❌ 모터 연결 실패")

    def cmd(self, c: bytes):
        if self.ser and self.ser.is_open:
            self.ser.write(c)
            time.sleep(0.01)

    def forward(self): self.cmd(b'w')
    def left(self): self.cmd(b'a')
    def right(self): self.cmd(b'd')
    def stop(self): self.cmd(b'x')

    def close(self):
        self.stop()
        if self.ser: self.ser.close()

# ======================
# Docking Controller
# ======================
class DockingController(Node):
    def __init__(self):
        super().__init__('docking_controller')
        self.bridge = CvBridge()
        self.motor = Motor()

        print("📦 YOLO 로딩 중...")
        self.model = YOLO(MODEL_PATH)
        self.conf = CONFIDENCE_THRESHOLD
        print("✅ YOLO 로드 완료")

        self.sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.callback, 10
        )

        # 상태 변수
        self.last_seen_time = 0
        self.last_target_cx = None
        self.search_direction = 'right'
        self.last_scan_time = 0
        self.target_locked = False

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.process_frame(frame)

    def process_frame(self, frame):
        results = self.model.predict(frame, conf=self.conf, verbose=False, device='cpu')
        frame_cx = COLOR_W // 2
        target = None

        # --- YOLO 탐지 ---
        for r in results:
            for box in r.boxes:
                cls = r.names[int(box.cls[0])]
                if cls == TARGET_NAME:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    cx, cy = x1 + w // 2, y1 + h // 2
                    target = {'cx': cx, 'cy': cy, 'h': h}
                    self.last_seen_time = time.time()
                    self.last_target_cx = cx
                    break

        # --- 타깃이 안 보이는 경우 ---
        if not target:
            self.handle_lost_target(frame, frame_cx)
            cv2.imshow("Docking", frame)
            cv2.waitKey(1)
            return

        # --- 타깃이 보이는 경우 ---
        error = target['cx'] - frame_cx
        cv2.line(frame, (frame_cx, 0), (frame_cx, COLOR_H), (0, 255, 255), 2)
        cv2.circle(frame, (target['cx'], target['cy']), 10, (0, 255, 0), -1)
        cv2.putText(frame, f"Error: {error}px", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if target['h'] > FORWARD_THRESHOLD:
            cv2.putText(frame, "PARKED!", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            self.motor.stop()
            self.target_locked = True

        elif abs(error) <= DEADZONE:
            cv2.putText(frame, "Aligned - Forward", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.motor.forward()
            time.sleep(FORWARD_TIME)
            self.motor.stop()

        else:
            if error > 0:
                cv2.putText(frame, "Turning Right", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                self.motor.right()
            else:
                cv2.putText(frame, "Turning Left", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                self.motor.left()
            time.sleep(TURN_TIME * min(abs(error) / 100, 1.5))
            self.motor.stop()

        cv2.imshow("Docking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.motor.close()
            rclpy.shutdown()

    def handle_lost_target(self, frame, frame_cx):
        elapsed = time.time() - self.last_seen_time
        now = time.time()

        if elapsed < SEARCH_TIMEOUT and self.last_target_cx:
            # 최근에 봤던 방향으로 회전
            cv2.putText(frame, "Re-acquiring target...", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if self.last_target_cx < frame_cx:
                self.motor.left()
            else:
                self.motor.right()
            time.sleep(0.15)
            self.motor.stop()

        elif now - self.last_scan_time > SCAN_INTERVAL:
            # 일정 시간 이상 타깃 없음 → 좌우 스캔
            cv2.putText(frame, f"Searching ({self.search_direction})", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if self.search_direction == 'right':
                self.motor.right()
                time.sleep(SCAN_TIME)
                self.search_direction = 'left'
            else:
                self.motor.left()
                time.sleep(SCAN_TIME)
                self.search_direction = 'right'
            self.motor.stop()
            self.last_scan_time = now

def main(args=None):
    rclpy.init(args=args)
    node = DockingController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.motor.close()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
