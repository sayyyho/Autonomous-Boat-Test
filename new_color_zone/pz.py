#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase1: Improved Navigation with Simple HSV Detection + Horizontal Gate Logic
- 단순하고 강건한 HSV 색상 감지
- 수평 정렬된 게이트만 유효하게 인식
- 좌=초록, 우=빨강 배치 시 직진 신호
"""

import time
import serial
from typing import List, Tuple, Optional, Dict
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

TOTAL_GATES = int(input("통과해야 할 게이트 수를 입력하세요 (기본 5): ") or "5")
print(f"✅ 총 {TOTAL_GATES}개의 게이트를 통과합니다.")

COLOR_W, COLOR_H = 640, 480

# ⭐ 단순화된 HSV 범위 (두 번째 코드 기반)
HSV_GREEN_LOWER = np.array([72, 120, 90])
HSV_GREEN_UPPER = np.array([92, 255, 255])

HSV_RED_LOWER1 = np.array([0, 100, 100])
HSV_RED_UPPER1 = np.array([10, 255, 255])
HSV_RED_LOWER2 = np.array([165, 100, 100])
HSV_RED_UPPER2 = np.array([180, 255, 255])

HSV_YELLOW_LOWER = np.array([22, 120, 120])
HSV_YELLOW_UPPER = np.array([32, 255, 255])

# 최소 면적 필터
MIN_AREA_GREEN = 500
MIN_AREA_RED = 500
MIN_AREA_YELLOW = 1000

# 수평 정렬 허용 오차 (픽셀)
Y_ALIGNMENT_THRESHOLD = 75

# 게이트 중심 데드존 (픽셀)
GATE_CENTER_DEADZONE = 40

# 타이밍 설정
FORWARD_TIME = 0.3
TURN_SMALL_TIME = 0.4
SCAN_TURN_TIME = 1.0
APPROACH_TIME = 0.5

YELLOW_STOP_DISTANCE = 5.0
YELLOW_WAIT_TIME = 5.0
AFTER_YELLOW_FORWARD = 3.0

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
    """모터 액션 실행"""
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
# ---- 비전 처리 함수 ----
# ----------------------------
def detect_green(hsv: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """초록색 객체 검출"""
    mask = cv2.inRange(hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_AREA_GREEN:
            results.append(cv2.boundingRect(cnt))
    
    return results

def detect_red(hsv: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """빨간색 객체 검출 (2개 범위 병합)"""
    mask1 = cv2.inRange(hsv, HSV_RED_LOWER1, HSV_RED_UPPER1)
    mask2 = cv2.inRange(hsv, HSV_RED_LOWER2, HSV_RED_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_AREA_RED:
            results.append(cv2.boundingRect(cnt))
    
    return results

def detect_yellow(hsv: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """노란색 객체 검출 (가장 큰 것만)"""
    mask = cv2.inRange(hsv, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA_YELLOW]
    if not valid:
        return None
    
    largest = max(valid, key=cv2.contourArea)
    return cv2.boundingRect(largest)

def find_horizontal_gate_pair(red_bbs: List[Tuple[int, int, int, int]], 
                               green_bbs: List[Tuple[int, int, int, int]],
                               frame_width: int) -> Optional[Tuple[Tuple, Tuple]]:
    """
    수평 정렬된 게이트 쌍 찾기
    조건: 좌=초록, 우=빨강 + Y좌표 정렬
    """
    if not red_bbs or not green_bbs:
        return None
    
    frame_center = frame_width // 2
    best_pair = None
    min_distance = float('inf')
    
    for green_bb in green_bbs:
        gx, gy, gw, gh = green_bb
        green_cx = gx + gw // 2
        green_cy = gy + gh // 2
        
        for red_bb in red_bbs:
            rx, ry, rw, rh = red_bb
            red_cx = rx + rw // 2
            red_cy = ry + rh // 2
            
            # ⭐ 조건 1: 좌=초록, 우=빨강
            if green_cx >= red_cx:
                continue
            
            # ⭐ 조건 2: Y좌표 수평 정렬
            if abs(green_cy - red_cy) > Y_ALIGNMENT_THRESHOLD:
                continue
            
            # 게이트 중심 계산
            gate_center_x = (green_cx + red_cx) // 2
            distance = abs(gate_center_x - frame_center)
            
            if distance < min_distance:
                min_distance = distance
                best_pair = (red_bb, green_bb)
    
    return best_pair

# ----------------------------
# ---- ROS2 Navigator Node ----
# ----------------------------
class Phase1Navigator(Node):
    def __init__(self):
        super().__init__('phase1_navigator')
        
        self.motor = init_motor()
        
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
        
        self.mission_stage = 'NAVIGATION'  # NAVIGATION -> STATION_KEEPING -> DOCKING
        self.gates_passed = 0
        self.gate_passing_state = 'SEARCHING'  # SEARCHING -> APPROACHING -> PASSING
        
        self.last_gate_seen = time.time()
        self.scan_direction = 'right'
        
        self.get_logger().info("=== Phase1 Navigator 시작 ===")

    def color_callback(self, msg: Image):
        self.color_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.process_frame()

    def depth_callback(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def get_depth(self, x: int, y: int) -> float:
        """특정 픽셀의 깊이 값 (미터)"""
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
        
        # ⭐ 단순 HSV 기반 색상 검출
        green_bbs = detect_green(hsv)
        red_bbs = detect_red(hsv)
        yellow_bb = detect_yellow(hsv)
        
        # 디버그 시각화
        for bb in green_bbs:
            x, y, w, h = bb
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "GREEN", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for bb in red_bbs:
            x, y, w, h = bb
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "RED", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if yellow_bb:
            x, y, w, h = yellow_bb
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, "YELLOW", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 미션 단계별 처리
        cv2.putText(frame, f"Stage: {self.mission_stage} | Gates: {self.gates_passed}/{TOTAL_GATES}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if self.mission_stage == 'NAVIGATION':
            self.navigation_stage(red_bbs, green_bbs, yellow_bb, frame)
        elif self.mission_stage == 'STATION_KEEPING':
            self.station_keeping_stage(yellow_bb, frame)
        elif self.mission_stage == 'DOCKING':
            self.docking_stage(frame)
        
        cv2.imshow("Phase1 Navigator", frame)
        cv2.waitKey(1)

    def navigation_stage(self, red_bbs, green_bbs, yellow_bb, frame):
        """항법 단계: 게이트 통과"""
        
        # 모든 게이트 통과 완료 시
        if self.gates_passed >= TOTAL_GATES:
            if yellow_bb:
                self.get_logger().info(f"✅ {TOTAL_GATES}개 게이트 통과 완료 → 노란부표 발견!")
                self.mission_stage = 'STATION_KEEPING'
                return
            else:
                cv2.putText(frame, "All gates passed! Searching YELLOW...", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                self.scan_for_buoy()
                return
        
        # ⭐ 수평 정렬 게이트 찾기
        gate_pair = find_horizontal_gate_pair(red_bbs, green_bbs, frame.shape[1])
        
        if gate_pair:
            self.last_gate_seen = time.time()
            red_bb, green_bb = gate_pair
            
            rx, ry, rw, rh = red_bb
            gx, gy, gw, gh = green_bb
            
            red_cx = rx + rw // 2
            green_cx = gx + gw // 2
            gate_cx = (red_cx + green_cx) // 2
            gate_cy = (ry + rh//2 + gy + gh//2) // 2
            
            # 게이트 강조 표시
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 0, 255), 3)
            cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (255, 0, 255), 3)
            cv2.line(frame, (gate_cx, 0), (gate_cx, COLOR_H), (0, 255, 255), 2)
            cv2.circle(frame, (gate_cx, gate_cy), 10, (0, 255, 255), -1)
            
            cv2.putText(frame, f"GATE #{self.gates_passed+1}", (gate_cx-50, gate_cy-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # ⭐ 게이트 통과 로직
            self.pass_through_gate(gate_cx, gate_cy, frame)
        
        else:
            # 게이트 없을 때
            cv2.putText(frame, f"Searching Gate #{self.gates_passed+1}...", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if time.time() - self.last_gate_seen < 2.0:
                # 최근 봤으면 직진 유지
                motor_action('forward')
                time.sleep(FORWARD_TIME)
            else:
                # 안 보이면 스캔
                self.scan_for_buoy()
            
            motor_action('stop')

    def pass_through_gate(self, gate_cx: int, gate_cy: int, frame: np.ndarray):
        """게이트 중앙으로 정렬 후 통과"""
        frame_cx = frame.shape[1] // 2
        
        # ⭐ 게이트가 화면 하단에 가까워지면 통과 중
        if gate_cy > COLOR_H * 0.65:
            if self.gate_passing_state != 'PASSING':
                self.gate_passing_state = 'PASSING'
                self.get_logger().info(f"🚪 게이트 #{self.gates_passed+1} 통과 시작")
            
            # 중앙 정렬하며 전진
            error = gate_cx - frame_cx
            if abs(error) > GATE_CENTER_DEADZONE // 2:
                if error > 0:
                    motor_action('right')
                else:
                    motor_action('left')
                time.sleep(TURN_SMALL_TIME * 0.3)
            
            motor_action('forward')
            time.sleep(APPROACH_TIME * 1.5)
            motor_action('stop')
            
            # 통과 카운트
            self.gates_passed += 1
            self.get_logger().info(f"✅ 게이트 #{self.gates_passed}/{TOTAL_GATES} 통과!")
            self.gate_passing_state = 'SEARCHING'
        
        else:
            # 접근 중 - 중앙 정렬
            self.gate_passing_state = 'APPROACHING'
            error = gate_cx - frame_cx
            
            if abs(error) <= GATE_CENTER_DEADZONE:
                self.get_logger().info("✅ 게이트 중앙 정렬 → 직진")
                motor_action('forward')
                time.sleep(APPROACH_TIME)
            elif error > 0:
                self.get_logger().info(f"우측 {error}px → 우회전")
                motor_action('right')
                time.sleep(TURN_SMALL_TIME * min(abs(error)/100, 1.0))
            else:
                self.get_logger().info(f"좌측 {abs(error)}px → 좌회전")
                motor_action('left')
                time.sleep(TURN_SMALL_TIME * min(abs(error)/100, 1.0))
            
            motor_action('stop')

    def scan_for_buoy(self):
        """부표 찾기 위한 좌우 스캔"""
        self.get_logger().info(f"🔍 [{self.scan_direction}] 스캔 중...")
        
        if self.scan_direction == 'left':
            motor_action('left')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'right'
        else:
            motor_action('right')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'left'
        
        motor_action('stop')

    def station_keeping_stage(self, yellow_bb, frame):
        """위치유지 단계"""
        if yellow_bb:
            x, y, w, h = yellow_bb
            cx = x + w // 2
            cy = y + h // 2
            
            depth = self.get_depth(cx, cy)
            
            if 0.1 < depth < YELLOW_STOP_DISTANCE:
                self.get_logger().info(f"🟡 노란부표 {depth:.2f}m 도달 → 5초 대기")
                motor_action('stop')
                time.sleep(YELLOW_WAIT_TIME)
                self.mission_stage = 'DOCKING'
                return
            
            # 접근
            frame_cx = frame.shape[1] // 2
            if cx < frame_cx - GATE_CENTER_DEADZONE:
                motor_action('left')
            elif cx > frame_cx + GATE_CENTER_DEADZONE:
                motor_action('right')
            else:
                motor_action('forward')
            time.sleep(APPROACH_TIME)
            motor_action('stop')
        else:
            self.scan_for_buoy()

    def docking_stage(self, frame):
        """도킹 구역으로 전진"""
        self.get_logger().info(f"🚢 도킹 구역으로 {AFTER_YELLOW_FORWARD}초 전진")
        motor_action('forward')
        time.sleep(AFTER_YELLOW_FORWARD)
        motor_action('stop')
        self.get_logger().info("✅ Phase1 완료!")
        cv2.waitKey(3000)
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = Phase1Navigator()
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