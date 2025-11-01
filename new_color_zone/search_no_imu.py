# 111
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase2: Search Mission (No IMU) - 시간 기반 선회
- 목표 색상 부표 탐지 및 접근
- 일정 거리 도달 시 선회 시작
- 빨강/초록: 시계방향 / 파랑: 반시계방향
- 시간 기반 360도 회전
"""

import time
import serial
from typing import Optional, Tuple
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ----------------------------
# ---- 설정 파라미터 ----
# ----------------------------
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
DEFAULT_SPEED = '5'

COLOR_W, COLOR_H = 640, 480

# ⭐ 미션 설정
print("=== 탐색(Search) 미션 시작 ===")
TARGET_COLOR = input("목표 색상을 입력하세요 (red/green/blue): ").strip().lower()
while TARGET_COLOR not in ['red', 'green', 'blue']:
    print("❌ 잘못된 입력! red, green, blue 중 하나를 입력하세요.")
    TARGET_COLOR = input("목표 색상을 입력하세요 (red/green/blue): ").strip().lower()

# 선회 방향 결정
CLOCKWISE = TARGET_COLOR in ['red', 'green']
DIRECTION_TEXT = "시계방향(CW)" if CLOCKWISE else "반시계방향(CCW)"
print(f"✅ 목표: {TARGET_COLOR.upper()} 부표")
print(f"✅ 선회 방향: {DIRECTION_TEXT}")

# HSV 범위
HSV_RANGES = {
    'red': [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([165, 100, 100]), np.array([180, 255, 255]))
    ],
    'green': [
        (np.array([72, 120, 90]), np.array([92, 255, 255]))
    ],
    'blue': [
        (np.array([100, 150, 100]), np.array([130, 255, 255]))
    ]
}

MIN_AREA = 500

# 선회 설정
APPROACH_DISTANCE = 3.0  # 선회 시작 거리 (미터)
CIRCLE_SPEED = '4'  # 선회 속도

# ⭐ 시간 기반 선회 설정
# 이 값은 실제 테스트를 통해 보정 필요!
CIRCLE_TIME_PER_90DEG = 2.0  # 90도 회전에 걸리는 시간 (초)
FULL_CIRCLE_TIME = CIRCLE_TIME_PER_90DEG * 4  # 360도 = 8초

print(f"⚙️  예상 360도 선회 시간: {FULL_CIRCLE_TIME}초")
calibrate = input("선회 시간을 수동 설정하시겠습니까? (y/N): ").strip().lower()
if calibrate == 'y':
    FULL_CIRCLE_TIME = float(input("360도 선회에 걸리는 시간(초)를 입력하세요: "))
    print(f"✅ 선회 시간: {FULL_CIRCLE_TIME}초로 설정")

# 타이밍
FORWARD_TIME = 0.3
TURN_TIME = 0.4
SCAN_TURN_TIME = 1.0

# ----------------------------
# ---- 아두이노 모터 제어 ----
# ----------------------------
class ArduinoMotorController:
    def __init__(self, port: str = SERIAL_PORT, baudrate: int = BAUD_RATE):
        self.ser = None
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            self.set_speed(DEFAULT_SPEED)
            self.stop()
            print(f"✅ 아두이노 연결: {port}")
        except serial.SerialException as e:
            print(f"❌ 아두이노 연결 실패: {e}")
            
    def send_command(self, command: bytes):
        if self.ser and self.ser.is_open:
            self.ser.write(command)
            time.sleep(0.01)
    
    def set_speed(self, speed: str):
        if speed.isdigit() and '0' <= speed <= '9':
            self.send_command(speed.encode())
    
    def forward(self):
        self.send_command(b'w')
    
    def backward(self):
        self.send_command(b's')
    
    def left(self):
        self.send_command(b'a')
    
    def right(self):
        self.send_command(b'd')
    
    def stop(self):
        self.send_command(b'x')
    
    def close(self):
        if self.ser and self.ser.is_open:
            self.stop()
            self.ser.close()
            print("✅ 아두이노 종료")

motor_controller = None

def init_motor():
    global motor_controller
    motor_controller = ArduinoMotorController()
    return motor_controller

def motor_action(action: str):
    global motor_controller
    if not motor_controller:
        return
    
    action = action.lower()
    if action == 'forward':
        motor_controller.forward()
    elif action == 'backward':
        motor_controller.backward()
    elif action == 'left':
        motor_controller.left()
    elif action == 'right':
        motor_controller.right()
    elif action == 'stop':
        motor_controller.stop()

# ----------------------------
# ---- 비전 처리 ----
# ----------------------------
def detect_color(hsv: np.ndarray, color: str) -> Optional[Tuple[int, int, int, int]]:
    """특정 색상 검출 (가장 큰 것만)"""
    if color not in HSV_RANGES:
        return None
    
    masks = []
    for lower, upper in HSV_RANGES[color]:
        mask = cv2.inRange(hsv, lower, upper)
        masks.append(mask)
    
    combined_mask = masks[0]
    for m in masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, m)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    if not valid:
        return None
    
    largest = max(valid, key=cv2.contourArea)
    return cv2.boundingRect(largest)

# ----------------------------
# ---- 시간 기반 회전 추적 ----
# ----------------------------
class TimeBasedRotation:
    """시간 기반 회전 추적"""
    
    def __init__(self, full_circle_time: float):
        self.full_circle_time = full_circle_time
        self.start_time = None
        self.is_active = False
    
    def start(self):
        """선회 시작"""
        self.start_time = time.time()
        self.is_active = True
        print(f"🔄 선회 시작! 목표: {self.full_circle_time}초")
    
    def get_elapsed(self) -> float:
        """경과 시간"""
        if not self.is_active or self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_progress_percent(self) -> float:
        """진행률 (%)"""
        elapsed = self.get_elapsed()
        return (elapsed / self.full_circle_time) * 100
    
    def get_estimated_degrees(self) -> float:
        """예상 회전 각도"""
        elapsed = self.get_elapsed()
        return (elapsed / self.full_circle_time) * 360
    
    def is_complete(self) -> bool:
        """360도 완료 여부"""
        return self.is_active and self.get_elapsed() >= self.full_circle_time
    
    def stop(self):
        """선회 종료"""
        self.is_active = False

# ----------------------------
# ---- ROS2 Search Navigator ----
# ----------------------------
class SearchNavigator(Node):
    def __init__(self):
        super().__init__('search_navigator')
        
        self.motor = init_motor()
        
        # ROS2 구독자
        self.bridge = CvBridge()
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.color_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw',
            self.depth_callback, 10
        )
        
        self.color_img = None
        self.depth_img = None
        
        # 미션 상태
        self.mission_stage = 'SEARCHING'  # SEARCHING -> APPROACHING -> CIRCLING -> COMPLETE
        self.rotation_tracker = TimeBasedRotation(FULL_CIRCLE_TIME)
        self.last_buoy_seen = time.time()
        self.scan_direction = 'right'
        
        self.get_logger().info(f"=== 탐색 미션 시작: {TARGET_COLOR.upper()} 부표 ===")
        self.get_logger().info(f"=== 선회 방향: {DIRECTION_TEXT} ===")

    def color_callback(self, msg: Image):
        self.color_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.process_frame()

    def depth_callback(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def get_depth(self, x: int, y: int) -> float:
        """특정 픽셀의 깊이 값"""
        if self.depth_img is None:
            return 0.0
        try:
            val = self.depth_img[y, x]
            if np.issubdtype(self.depth_img.dtype, np.integer):
                return float(val) / 1000.0
            return float(val)
        except:
            return 0.0

    def process_frame(self):
        """메인 프레임 처리"""
        if self.color_img is None:
            return
        
        frame = self.color_img.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 목표 색상 검출
        target_bb = detect_color(hsv, TARGET_COLOR)
        
        # 디버그 표시
        if target_bb:
            x, y, w, h = target_bb
            color_bgr = (0, 0, 255) if TARGET_COLOR == 'red' else \
                        (0, 255, 0) if TARGET_COLOR == 'green' else \
                        (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 3)
            cv2.putText(frame, f"{TARGET_COLOR.upper()} TARGET", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        
        # 미션 상태 표시
        cv2.putText(frame, f"Stage: {self.mission_stage}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if self.mission_stage == 'CIRCLING':
            elapsed = self.rotation_tracker.get_elapsed()
            progress = self.rotation_tracker.get_progress_percent()
            estimated_deg = self.rotation_tracker.get_estimated_degrees()
            
            cv2.putText(frame, f"Time: {elapsed:.1f}s / {FULL_CIRCLE_TIME:.1f}s", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Progress: {progress:.1f}% (~{estimated_deg:.0f} deg)", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 미션 단계별 처리
        if self.mission_stage == 'SEARCHING':
            self.searching_stage(target_bb, frame)
        elif self.mission_stage == 'APPROACHING':
            self.approaching_stage(target_bb, frame)
        elif self.mission_stage == 'CIRCLING':
            self.circling_stage(target_bb, frame)
        elif self.mission_stage == 'COMPLETE':
            self.complete_stage(frame)
        
        cv2.imshow("Search Navigator", frame)
        cv2.waitKey(1)

    def searching_stage(self, target_bb, frame):
        """목표 부표 탐색"""
        if target_bb:
            self.last_buoy_seen = time.time()
            self.get_logger().info(f"✅ {TARGET_COLOR.upper()} 부표 발견!")
            self.mission_stage = 'APPROACHING'
        else:
            cv2.putText(frame, f"Searching {TARGET_COLOR.upper()} buoy...", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.scan_for_buoy()

    def approaching_stage(self, target_bb, frame):
        """목표 부표 접근"""
        if not target_bb:
            # 부표 놓쳤을 때
            if time.time() - self.last_buoy_seen < 2.0:
                motor_action('forward')
                time.sleep(FORWARD_TIME)
                motor_action('stop')
            else:
                self.get_logger().info("❌ 부표 놓침 → 재탐색")
                self.mission_stage = 'SEARCHING'
            return
        
        self.last_buoy_seen = time.time()
        
        x, y, w, h = target_bb
        cx = x + w // 2
        cy = y + h // 2
        
        depth = self.get_depth(cx, cy)
        
        if 0.1 < depth < APPROACH_DISTANCE:
            self.get_logger().info(f"🎯 목표 거리 {depth:.2f}m 도달 → 선회 시작!")
            self.start_circling()
            return
        
        cv2.putText(frame, f"Approaching: {depth:.2f}m / {APPROACH_DISTANCE:.2f}m", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 부표 중앙으로 정렬하며 접근
        frame_cx = frame.shape[1] // 2
        error = cx - frame_cx
        
        if abs(error) > 40:
            if error > 0:
                motor_action('right')
            else:
                motor_action('left')
            time.sleep(TURN_TIME * 0.3)
        else:
            motor_action('forward')
            time.sleep(FORWARD_TIME)
        
        motor_action('stop')

    def start_circling(self):
        """선회 시작"""
        self.mission_stage = 'CIRCLING'
        self.rotation_tracker.start()
        
        # 선회 속도 설정
        self.motor.set_speed(CIRCLE_SPEED)
        
        self.get_logger().info(f"🔄 {DIRECTION_TEXT} 선회 시작!")

    def circling_stage(self, target_bb, frame):
        """선회 동작"""
        
        # ⭐ 시간 기반 360도 완료 체크
        if self.rotation_tracker.is_complete():
            elapsed = self.rotation_tracker.get_elapsed()
            self.get_logger().info(f"✅ 선회 완료! 소요 시간: {elapsed:.1f}초")
            self.rotation_tracker.stop()
            self.mission_stage = 'COMPLETE'
            motor_action('stop')
            return
        
        elapsed = self.rotation_tracker.get_elapsed()
        progress = self.rotation_tracker.get_progress_percent()
        
        # 진행률 표시
        cv2.putText(frame, f"Circling {DIRECTION_TEXT}...", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ⭐ 선회 방향에 따라 회전 + 약간의 전진 (원 그리기)
        if CLOCKWISE:
            motor_action('right')
        else:
            motor_action('left')
        
        # 약간의 전진을 추가하여 제자리 회전이 아닌 원형 선회
        # (이 부분은 로봇 특성에 따라 조정 필요)
        time.sleep(0.05)
        motor_action('forward')
        time.sleep(0.02)

    def scan_for_buoy(self):
        """부표 스캔"""
        if self.scan_direction == 'left':
            motor_action('left')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'right'
        else:
            motor_action('right')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'left'
        
        motor_action('stop')

    def complete_stage(self, frame):
        """미션 완료"""
        cv2.putText(frame, "MISSION COMPLETE!", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        motor_action('stop')
        time.sleep(3)
        self.get_logger().info("🎉 탐색 미션 완료!")
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SearchNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if motor_controller:
            motor_controller.close()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()