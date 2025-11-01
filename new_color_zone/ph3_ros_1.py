#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase2: 탐색 선회 (ROS2 버전)
- RealSense D435i 사용
- 시간 기반 360도 선회
- 모폴로지 연산으로 노이즈 제거
- FPS 실시간 표시
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

# ⭐ HSV 범위 (개선된 버전)
HSV_RANGES = {
    'red': [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ],
    'green': [
        (np.array([35, 70, 70]), np.array([85, 255, 255]))
    ],
    'blue': [
        (np.array([90, 80, 50]), np.array([130, 255, 255]))
    ]
}

MIN_AREA = 500

# ⭐ 모폴로지 연산용 커널
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 선회 설정
CIRCLE_SPEED = '4'

# 시간 기반 선회 설정
FULL_CIRCLE_TIME = 8.0  # ROS2 파라미터로 설정 가능

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
        self.use_serial = True
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            self.set_speed(DEFAULT_SPEED)
            self.stop()
            print(f"✅ 아두이노 연결: {port}")
        except Exception as e:
            print(f"⚠️  아두이노 연결 실패: {e}")
            print("⚠️  시뮬레이션 모드로 실행합니다.")
            self.use_serial = False
            
    def send_command(self, command: bytes):
        if self.use_serial and self.ser and self.ser.is_open:
            self.ser.write(command)
            time.sleep(0.01)
        else:
            cmd = command.decode('utf-8', errors='ignore')
            print(f"[MOTOR] {cmd}")
    
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
        if self.use_serial and self.ser and self.ser.is_open:
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
    """특정 색상 검출 (가장 큰 것만) - 모폴로지 연산 적용"""
    if color not in HSV_RANGES:
        return None
    
    masks = []
    for lower, upper in HSV_RANGES[color]:
        mask = cv2.inRange(hsv, lower, upper)
        # ⭐ 모폴로지 연산으로 노이즈 제거
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)
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
        
        # ROS2 파라미터
        self.declare_parameter('target_color', 'green')
        self.declare_parameter('circle_time', FULL_CIRCLE_TIME)
        
        target_color = self.get_parameter('target_color').get_parameter_value().string_value
        self.target_color = target_color.lower()
        circle_time = self.get_parameter('circle_time').get_parameter_value().double_value
        
        # 선회 방향 결정
        self.clockwise = self.target_color in ['red', 'green']
        direction_text = "시계방향(CW)" if self.clockwise else "반시계방향(CCW)"
        
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
        self.mission_stage = 'SEARCHING'
        self.rotation_tracker = TimeBasedRotation(circle_time)
        self.last_buoy_seen = time.time()
        self.scan_direction = 'right'
        self.last_scan_time = 0
        
        # FPS 추적
        self._t_prev = time.time()
        self._fps_smooth = None
        
        self.get_logger().info(f"=== 탐색 미션 시작: {self.target_color.upper()} 부표 ===")
        self.get_logger().info(f"=== 선회 방향: {direction_text} ===")

    def color_callback(self, msg: Image):
        self.color_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.process_frame()

    def depth_callback(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def get_depth(self, x: int, y: int) -> float:
        if self.depth_img is None:
            return 0.0
        try:
            val = self.depth_img[y, x]
            if np.issubdtype(self.depth_img.dtype, np.integer):
                return float(val) / 1000.0
            return float(val)
        except:
            return 0.0

    def _update_fps(self) -> float:
        t = time.time()
        dt = t - self._t_prev
        self._t_prev = t
        fps = 1.0 / dt if dt > 1e-6 else 0.0
        
        if self._fps_smooth is None:
            self._fps_smooth = fps
        else:
            self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * fps
        
        return self._fps_smooth

    def process_frame(self):
        if self.color_img is None:
            return
        
        frame = self.color_img.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        target_bb = detect_color(hsv, self.target_color)
        
        # 디버그 표시
        if target_bb:
            x, y, w, h = target_bb
            color_bgr = (0, 0, 255) if self.target_color == 'red' else \
                        (0, 255, 0) if self.target_color == 'green' else \
                        (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 3)
            cv2.putText(frame, f"{self.target_color.upper()} TARGET", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        
        # 미션 상태 표시
        cv2.putText(frame, f"Stage: {self.mission_stage}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if self.mission_stage == 'CIRCLING':
            elapsed = self.rotation_tracker.get_elapsed()
            progress = self.rotation_tracker.get_progress_percent()
            estimated_deg = self.rotation_tracker.get_estimated_degrees()
            
            cv2.putText(frame, f"Time: {elapsed:.1f}s / {self.rotation_tracker.full_circle_time:.1f}s", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Progress: {progress:.1f}% (~{estimated_deg:.0f} deg)", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # FPS 표시
        fps = self._update_fps()
        cv2.putText(frame, f"{fps:5.1f} FPS", (20, COLOR_H - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        
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
        if target_bb:
            self.last_buoy_seen = time.time()
            self.get_logger().info(f"✅ {self.target_color.upper()} 부표 발견!")
            self.mission_stage = 'APPROACHING'
        else:
            cv2.putText(frame, f"Searching {self.target_color.upper()} buoy...", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            self.scan_for_buoy()
            
            if time.time() - self.last_scan_time > 0.5:
                motor_action('stop')

    def approaching_stage(self, target_bb, frame):
        if not target_bb:
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
        
        area = w * h
        
        if area > 15000:
            self.get_logger().info(f"🎯 목표 거리 도달 → 선회 시작!")
            self.start_circling()
            return
        
        cv2.putText(frame, f"Approaching... (area: {area})", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
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
        self.mission_stage = 'CIRCLING'
        self.rotation_tracker.start()
        self.motor.set_speed(CIRCLE_SPEED)
        
        direction = "시계방향" if self.clockwise else "반시계방향"
        self.get_logger().info(f"🔄 {direction} 선회 시작!")

    def circling_stage(self, target_bb, frame):
        if self.rotation_tracker.is_complete():
            elapsed = self.rotation_tracker.get_elapsed()
            self.get_logger().info(f"✅ 선회 완료! 소요 시간: {elapsed:.1f}초")
            self.rotation_tracker.stop()
            self.mission_stage = 'COMPLETE'
            motor_action('stop')
            return
        
        cv2.putText(frame, f"Circling {'CW' if self.clockwise else 'CCW'}...", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ⭐ 순수 회전만 수행
        if self.clockwise:
            motor_action('right')
        else:
            motor_action('left')

    def scan_for_buoy(self):
        current_time = time.time()
        
        if current_time - self.last_scan_time < 2.0:
            return
        
        self.last_scan_time = current_time
        self.get_logger().info(f"🔍 [{self.scan_direction}] 스캔 시작...")
        
        if self.scan_direction == 'left':
            motor_action('left')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'right'
        else:
            motor_action('right')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'left'
        
        motor_action('stop')
        self.get_logger().info("✅ 스캔 완료")

    def complete_stage(self, frame):
        cv2.putText(frame, "MISSION COMPLETE!", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        motor_action('stop')
        self.get_logger().info("🎉 탐색 미션 완료!")

    def destroy_node(self):
        if motor_controller:
            motor_controller.close()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SearchNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()