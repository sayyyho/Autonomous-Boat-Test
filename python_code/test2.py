#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase1: Navigation through Gates + Yellow Buoy Stop + Forward to Dock Area (ROS2 Version)
- Arduino-based motor control via serial communication
- Green-priority overlap: Overlapping red/green detections are treated as GREEN
- Aligned-gate logic: Only passes through horizontally-aligned (same Y-level) pairs
- Depth fallback for navigation
- Yellow buoy detection: approach to within 5m, wait 5s, then move forward toward dock
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
# ---- 설정 파라미터 영역 ----
# ----------------------------
# 아두이노 시리얼 설정
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
DEFAULT_SPEED = '5'  # 기본 속도 (0-9)

# ⭐ 미션 설정
TOTAL_GATES = int(input("통과해야 할 게이트 수를 입력하세요: ") or "5")  # 기본값 5개
print(f"총 {TOTAL_GATES}개의 게이트를 통과합니다.")

COLOR_W, COLOR_H = 640, 480

FORWARD_SPEED_TIME = 0.2
TURN_90_TIME = 1.1
TURN_SMALL_TIME = 0.4
SCAN_TURN_TIME = 1.0
APPROACH_FORWARD_TIME = 0.5
DEPTH_SAFE_DISTANCE = 1.0

# 크기 필터링 파라미터
MIN_CONTOUR_AREA_RED = 1000
MAX_CONTOUR_AREA_RED = 50000
MIN_CONTOUR_AREA_GREEN = 500
MAX_CONTOUR_AREA_GREEN = 30000
MIN_CONTOUR_AREA_YELLOW = 1000
MAX_CONTOUR_AREA_YELLOW = 40000

# 종횡비 제한
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 3.0
MIN_SIZE_PIXELS = 20  # 최소 가로/세로 픽셀

# 수평 정렬 게이트 허용 오차 (픽셀)
Y_ALIGNMENT_THRESHOLD_PX = 75

GATE_CENTER_DEADZONE = 40
DEPTH_SECTOR_WIDTH = 60
DEPTH_SAMPLE_Y = int(COLOR_H * 0.5)
YELLOW_STOP_DISTANCE = 5.0
YELLOW_WAIT_TIME = 5.0
AFTER_YELLOW_FORWARD_TIME = 3.0

# C자형 트랙 대응: 지속적인 좌우 스캔
CONTINUOUS_SCAN_INTERVAL = 2.0
GATE_LOST_THRESHOLD = 3.0

# HSV 범위 설정
HSV_RANGES: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
    'RED': [
        (np.array([0, 150, 120]), np.array([8, 255, 255])),
        (np.array([172, 150, 120]), np.array([180, 255, 255]))
    ],
    'GREEN': [
        (np.array([72, 120, 90]), np.array([92, 255, 255])),
    ],
    'YELLOW': [
        (np.array([22, 120, 120]), np.array([32, 255, 255]))
    ]
}

# ----------------------------
# ---- 아두이노 모터 제어 클래스 ----
# ----------------------------
class ArduinoMotorController:
    """아두이노와 시리얼 통신으로 모터 제어"""
    
    def __init__(self, port: str = SERIAL_PORT, baudrate: int = BAUD_RATE):
        self.ser = None
        self.current_command = b'x'  # 현재 명령 저장
        self.current_speed = DEFAULT_SPEED.encode()
        
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # 아두이노 부팅 대기
            self.set_speed(DEFAULT_SPEED)
            self.stop()
            print(f"✅ 아두이노 연결 성공: {port}")
        except serial.SerialException as e:
            print(f"❌ 아두이노 연결 실패: {e}")
            print("포트를 확인하거나 권한을 확인하세요: sudo usermod -a -G dialout $USER")
            
    def send_command(self, command: bytes):
        """아두이노에 명령 전송"""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(command)
                self.current_command = command
                time.sleep(0.01)  # 명령 전송 후 짧은 대기
            except Exception as e:
                print(f"명령 전송 실패: {e}")
    
    def set_speed(self, speed: str):
        """속도 설정 (0-9)"""
        if speed.isdigit() and '0' <= speed <= '9':
            self.current_speed = speed.encode()
            self.send_command(self.current_speed)
    
    def forward(self):
        """전진"""
        self.send_command(b'w')
    
    def backward(self):
        """후진"""
        self.send_command(b's')
    
    def left(self):
        """좌회전"""
        self.send_command(b'a')
    
    def right(self):
        """우회전"""
        self.send_command(b'd')
    
    def stop(self):
        """정지"""
        self.send_command(b'x')
    
    def close(self):
        """시리얼 포트 닫기"""
        if self.ser and self.ser.is_open:
            self.stop()  # 종료 전 모터 정지
            self.ser.close()
            print("✅ 아두이노 연결 종료")

# 전역 모터 컨트롤러 인스턴스
motor_controller = None

def init_motor_controller():
    """모터 컨트롤러 초기화"""
    global motor_controller
    motor_controller = ArduinoMotorController()
    return motor_controller

def set_motor_state_named(state: str) -> None:
    """명명된 상태로 모터 제어"""
    global motor_controller
    if not motor_controller:
        return
        
    state = state.lower()
    if state == 'forward':
        motor_controller.forward()
    elif state == 'backward':
        motor_controller.backward()
    elif state == 'left':
        motor_controller.left()
    elif state == 'right':
        motor_controller.right()
    elif state == 'stop':
        motor_controller.stop()

# ----------------------------
# ---- Vision 유틸 ---
# ----------------------------
def mask_for_color(hsv: np.ndarray, color: str) -> np.ndarray:
    color = color.upper()
    if color not in HSV_RANGES:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    masks = [cv2.inRange(hsv, lower, upper) for (lower, upper) in HSV_RANGES[color]]
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def find_all_contours_with_size_filter(mask: np.ndarray, min_area: int, max_area: int) -> List[Tuple[int, int, int, int]]:
    """크기와 비율로 필터링된 컨투어 찾기"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    
    valid_bbs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 면적 필터
        if area < min_area or area > max_area:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 종횡비 필터
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            continue
        
        # 최소 크기 필터
        if w < MIN_SIZE_PIXELS or h < MIN_SIZE_PIXELS:
            continue
        
        valid_bbs.append((x, y, w, h))
    
    return valid_bbs

def find_largest_contour_with_size_filter(mask: np.ndarray, min_area: int, max_area: int) -> Optional[Tuple[int, int, int, int]]:
    """크기 필터링된 가장 큰 컨투어 찾기"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            if MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO and w >= MIN_SIZE_PIXELS and h >= MIN_SIZE_PIXELS:
                valid_contours.append(cnt)
    
    if not valid_contours:
        return None
    
    largest = max(valid_contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)

def has_overlap(bb1: Tuple[int, int, int, int], bb2: Tuple[int, int, int, int]) -> bool:
    """두 바운딩 박스(x, y, w, h)가 겹치는지 2D로 확인"""
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    
    # AABB (Axis-Aligned Bounding Box) 충돌 검사
    if (x1 < x2 + w2 and x1 + w1 > x2 and
        y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False

def filter_overlapping_colors(red_bbs: List[Tuple[int, int, int, int]], 
                               green_bbs: List[Tuple[int, int, int, int]]) -> Tuple[List, List]:
    """
    빨강과 초록 바운딩 박스가 2D로 겹치면, 초록으로 우선 인지 (빨강 제거)
    """
    filtered_red = []
    
    for red_bb in red_bbs:
        is_actually_green = False
        for green_bb in green_bbs:
            # 2D 겹침 검사
            if has_overlap(red_bb, green_bb):
                is_actually_green = True
                break
        
        # 어떤 초록과도 겹치지 않는 빨강만 유지
        if not is_actually_green:
            filtered_red.append(red_bb)
    
    return filtered_red, green_bbs

def find_closest_gate_pair(red_bbs: List[Tuple[int, int, int, int]], 
                           green_bbs: List[Tuple[int, int, int, int]],
                           frame_width: int) -> Optional[Tuple[Tuple[int, int, int, int], 
                                                                Tuple[int, int, int, int]]]:
    """
    규칙(좌=초록, 우=빨강)과 수평 정렬(Y좌표)을 만족하는
    가장 중앙에 가까운 게이트 쌍을 찾음
    """
    if not red_bbs or not green_bbs:
        return None
    
    frame_center = frame_width // 2
    min_distance = float('inf')
    best_pair = None
    
    for green_bb in green_bbs:
        gx, gy, gw, gh = green_bb
        green_cx = gx + gw // 2
        green_cy = gy + gh // 2
        
        for red_bb in red_bbs:
            rx, ry, rw, rh = red_bb
            red_cx = rx + rw // 2
            red_cy = ry + rh // 2
            
            # 규칙 1: 좌측=초록, 우측=빨강 확인
            if green_cx >= red_cx:
                continue
            
            # 규칙 2: 수평 정렬(Y좌표) 확인
            if abs(green_cy - red_cy) > Y_ALIGNMENT_THRESHOLD_PX:
                continue
            
            # 게이트 중심 계산
            gate_center = (red_cx + green_cx) // 2
            
            # 프레임 중앙과의 거리
            distance = abs(gate_center - frame_center)
            
            if distance < min_distance:
                min_distance = distance
                best_pair = (red_bb, green_bb)
    
    return best_pair

# ----------------------------
# ---- Phase1 Navigator (ROS2) ----
# ----------------------------
class Phase1Navigator(Node):
    def __init__(self):
        super().__init__('phase1_navigator')
        
        # 모터 컨트롤러 초기화
        self.motor = init_motor_controller()
        
        # ROS2 구독자 설정
        self.bridge = CvBridge()
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        
        # 상태 변수
        self.color_img = None
        self.depth_img = None
        self.scan_direction = 'right'
        self.last_scan_time = 0
        self.last_gate_seen_time = time.time()
        self.last_auto_scan_time = time.time()
        self.mission_complete = False
        
        # ⭐ 미션 단계 관리 변수 추가
        self.mission_stage = 'NAVIGATION'  # 'NAVIGATION' -> 'STATION_KEEPING' -> 'DOCKING' 
        self.gates_passed = 0  # 통과한 게이트 수
        self.last_gate_position = None  # 마지막 게이트 위치 추적
        self.gate_passing_state = 'SEARCHING'  # 'SEARCHING' -> 'APPROACHING' -> 'PASSING'
        self.continuous_forward_time = 0  # 연속 전진 시간 누적
        
        self.get_logger().info("=== Phase1 Navigator 시작 (ROS2 + Arduino) ===")

    def color_callback(self, msg: Image):
        """컬러 이미지 수신"""
        self.color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_frame()

    def depth_callback(self, msg: Image):
        """깊이 이미지 수신"""
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_depth_at_point(self, x: int, y: int) -> float:
        """특정 픽셀의 깊이 값 반환 (미터 단위)"""
        if self.depth_img is None:
            return 0.0
        try:
            depth_val = self.depth_img[y, x]
            if np.issubdtype(self.depth_img.dtype, np.integer):
                return float(depth_val) / 1000.0  # mm to meters
            elif np.issubdtype(self.depth_img.dtype, np.floating):
                return float(depth_val)  # already in meters
            return 0.0
        except Exception as e:
            self.get_logger().warn(f"Get depth error: {e}")
            return 0.0

    def depth_sector_distances(self) -> Tuple[float, float, float]:
        """좌/중앙/우 섹터의 최소 거리 계산"""
        if self.depth_img is None:
            return (float('inf'), float('inf'), float('inf'))
        
        cx = COLOR_W // 2
        y = DEPTH_SAMPLE_Y
        
        def sector(px_start, px_end):
            vals = []
            for px in range(px_start, px_end):
                dist = self.get_depth_at_point(px, y)
                if dist > 0.1 and dist < 20.0:
                    vals.append(dist)
            return min(vals) if vals else float('inf')
        
        left = sector(max(0, cx - 3*DEPTH_SECTOR_WIDTH), max(0, cx - DEPTH_SECTOR_WIDTH))
        front = sector(max(0, cx - DEPTH_SECTOR_WIDTH), min(COLOR_W, cx + DEPTH_SECTOR_WIDTH))
        right = sector(min(COLOR_W-1, cx + DEPTH_SECTOR_WIDTH), min(COLOR_W, cx + 3*DEPTH_SECTOR_WIDTH))
        
        return left, front, right

    def auto_scan_for_gate(self):
        """C자형 트랙 대응: 자동 좌우 스캔으로 게이트 찾기"""
        current_time = time.time()
        
        if (current_time - self.last_gate_seen_time > GATE_LOST_THRESHOLD or 
            current_time - self.last_auto_scan_time > CONTINUOUS_SCAN_INTERVAL):
            
            self.last_auto_scan_time = current_time
            self.get_logger().info(f"🔍 [AUTO SCAN] 게이트 #{self.gates_passed+1} 찾기 - {self.scan_direction} 스캔")
            
            # 좌우 번갈아가며 스캔
            if self.scan_direction == 'left':
                set_motor_state_named('left')
                time.sleep(SCAN_TURN_TIME)  # 충분히 돌려서 부표 찾기
                self.scan_direction = 'right'
            else:
                set_motor_state_named('right')
                time.sleep(SCAN_TURN_TIME)
                self.scan_direction = 'left'
            
            set_motor_state_named('stop')
            time.sleep(0.2)  # 안정화 대기

    def process_frame(self):
        """메인 프로세싱 로직"""
        if self.color_img is None or self.mission_complete:
            return
        
        color_img = self.color_img.copy()
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        # 색상 감지 (크기 필터링 적용)
        mask_red = mask_for_color(hsv, 'RED')
        mask_green = mask_for_color(hsv, 'GREEN')
        mask_yellow = mask_for_color(hsv, 'YELLOW')

        red_bbs_raw = find_all_contours_with_size_filter(mask_red, MIN_CONTOUR_AREA_RED, MAX_CONTOUR_AREA_RED)
        green_bbs = find_all_contours_with_size_filter(mask_green, MIN_CONTOUR_AREA_GREEN, MAX_CONTOUR_AREA_GREEN)
        yellow_bb = find_largest_contour_with_size_filter(mask_yellow, MIN_CONTOUR_AREA_YELLOW, MAX_CONTOUR_AREA_YELLOW)

        # '초록 우선' 2D 겹침 필터 적용
        red_bbs, green_bbs = filter_overlapping_colors(red_bbs_raw, green_bbs)

        # 디버그: 필터링된 부표 표시
        for bb in red_bbs:
            x, y, w, h = bb
            area = w * h
            cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(color_img, f"RED({area})", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        for bb in green_bbs:
            x, y, w, h = bb
            area = w * h
            cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(color_img, f"GREEN({area})", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ⭐ 미션 단계 표시
        stage_text = f"Stage: {self.mission_stage} | Gates: {self.gates_passed}"
        cv2.putText(color_img, stage_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ⭐ 미션 단계별 처리
        if self.mission_stage == 'NAVIGATION':
            self.navigation_stage_process(red_bbs, green_bbs, yellow_bb, color_img)
        elif self.mission_stage == 'STATION_KEEPING':
            self.station_keeping_stage_process(yellow_bb, color_img)
        elif self.mission_stage == 'DOCKING':
            self.docking_stage_process(color_img)
        
        cv2.imshow("Phase1 View", color_img)
        cv2.waitKey(1)
    
    def navigation_stage_process(self, red_bbs, green_bbs, yellow_bb, color_img):
        """항로 추종 단계 처리"""
        
        # ⭐ 모든 게이트 통과 완료 시 노란 부표 탐색 모드로 전환
        if self.gates_passed >= TOTAL_GATES:
            if yellow_bb:
                self.get_logger().info(f"✅ 모든 {TOTAL_GATES}개 게이트 통과 완료!")
                self.get_logger().info("🟡 위치유지 구역 진입 - 노란부표 감지")
                self.mission_stage = 'STATION_KEEPING'
                self.last_gate_seen_time = time.time()
                return
            else:
                cv2.putText(color_img, f"All {TOTAL_GATES} gates passed! Searching YELLOW...", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # 노란 부표 찾기 위해 스캔
                self.auto_scan_for_gate()
                self.depth_follow(color_img)
                return
        
        # 게이트 처리: '수평 정렬'된 쌍 찾기
        if red_bbs and green_bbs:
            self.last_gate_seen_time = time.time()
            
            # ⭐ 빨강 1개 + 초록 1개 = 게이트로 인식
            self.get_logger().info(f"🔴 빨강 {len(red_bbs)}개, 🟢 초록 {len(green_bbs)}개 감지")
            
            gate_pair = find_closest_gate_pair(red_bbs, green_bbs, color_img.shape[1])
            
            if gate_pair:
                red_bb, green_bb = gate_pair
                
                rx, ry, rw, rh = red_bb
                gx, gy, gw, gh = green_bb
                
                red_cx, green_cx = rx + rw//2, gx + gw//2
                red_cy, green_cy = ry + rh//2, gy + gh//2
                gate_center_x = (red_cx + green_cx)//2
                gate_center_y = (red_cy + green_cy)//2
                
                # 선택된 게이트 쌍 강조 표시
                cv2.rectangle(color_img, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 3)
                cv2.rectangle(color_img, (gx, gy), (gx+gw, gy+gh), (0, 255, 0), 3)
                cv2.line(color_img, (gate_center_x, 0), (gate_center_x, COLOR_H), (255, 255, 0), 2)
                
                # 게이트 중점 표시
                cv2.circle(color_img, (gate_center_x, gate_center_y), 10, (255, 255, 0), -1)
                cv2.putText(color_img, f"GATE #{self.gates_passed+1}/{TOTAL_GATES}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(color_img, "TARGET: CENTER", (gate_center_x-50, gate_center_y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # ⭐ 게이트 통과 상태 관리 - 반드시 중앙을 통과
                self.manage_gate_passing((gate_center_x, gate_center_y), color_img)
            else:
                self.get_logger().info("부표는 보이나 유효한 수평 게이트가 없음 -> 계속 탐색")
                cv2.putText(color_img, "Searching valid gate pair...", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # 유효한 게이트 쌍이 안 만들어지면 좌우 스캔으로 찾기
                self.auto_scan_for_gate()
        
        elif red_bbs or green_bbs:
            # 한쪽 부표만 보일 때 - 짝을 찾기 위해 스캔
            self.last_gate_seen_time = time.time()
            visible_color = 'RED' if red_bbs else 'GREEN'
            missing_color = 'GREEN' if red_bbs else 'RED'
            bb = red_bbs[0] if red_bbs else green_bbs[0]
            
            cx = bb[0] + bb[2]//2
            cv2.putText(color_img, f"Found {visible_color}, scanning for {missing_color}...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # ⭐ 게이트 통과 직후일 수 있으므로 전진 유지
            if self.gate_passing_state == 'PASSING':
                self.continuous_forward(color_img)
            else:
                # 반대쪽 부표 찾기 위해 스캔
                self.scan_for_pair(visible_color, cx, color_img)
        
        else:
            # 부표가 아예 없을 시 
            cv2.putText(color_img, f"Searching gate #{self.gates_passed+1}/{TOTAL_GATES}...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # ⭐ 최근 게이트 통과 직후라면 전진 유지
            if time.time() - self.last_gate_seen_time < 2.0:
                self.continuous_forward(color_img)
            else:
                # 좌우 스캔하며 게이트 찾기
                self.auto_scan_for_gate()
                self.depth_follow(color_img)
    
    def station_keeping_stage_process(self, yellow_bb, color_img):
        """위치유지 단계 처리"""
        if yellow_bb:
            self.get_logger().info("🟡 노란부표 위치유지 중")
            x, y, w, h = yellow_bb
            cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(color_img, "STATION KEEPING", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            if self.approach_yellow_and_wait(yellow_bb, color_img):
                self.get_logger().info("✅ 위치유지 완료 → 도킹 단계로 전환")
                self.mission_stage = 'DOCKING'
        else:
            cv2.putText(color_img, "Searching for yellow buoy...", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.depth_follow(color_img)
    
    def docking_stage_process(self, color_img):
        """도킹 구역 이동 단계"""
        self.get_logger().info(f"🚢 도킹 구역으로 {AFTER_YELLOW_FORWARD_TIME}초 전진")
        cv2.putText(color_img, "Moving to DOCK", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        set_motor_state_named('forward')
        time.sleep(AFTER_YELLOW_FORWARD_TIME)
        set_motor_state_named('stop')
        
        self.get_logger().info("✅ Phase1 완료")
        self.mission_complete = True
    
    def manage_gate_passing(self, gate_center: Tuple[int, int], frame: np.ndarray):
        """게이트 통과 상태 관리 및 카운팅 - 반드시 중앙 통과"""
        gate_x, gate_y = gate_center
        frame_cx = frame.shape[1]//2
        frame_cy = frame.shape[1]//2
        
        # ⭐ 게이트가 화면 하단에 가까워지면 통과 중으로 판단
        if gate_y > COLOR_H * 0.65:  # 화면 하단 65% 이상
            if self.gate_passing_state != 'PASSING':
                self.gate_passing_state = 'PASSING'
                self.get_logger().info(f"🚪 게이트 #{self.gates_passed+1} 통과 시작")
            
            # 게이트 중앙으로 정렬하며 통과
            error = gate_x - frame_cx
            if abs(error) > GATE_CENTER_DEADZONE//2:  # 더 정밀한 중앙 정렬
                if error > 0:
                    set_motor_state_named('right')
                    time.sleep(TURN_SMALL_TIME * 0.3)
                else:
                    set_motor_state_named('left')
                    time.sleep(TURN_SMALL_TIME * 0.3)
            
            # 전진하여 게이트 통과
            set_motor_state_named('forward')
            time.sleep(APPROACH_FORWARD_TIME * 1.5)  # 확실한 통과를 위해 길게
            set_motor_state_named('stop')
            
            # 통과 직후 카운트 증가
            if self.last_gate_position and gate_y > self.last_gate_position[1] + 50:
                self.gates_passed += 1
                self.get_logger().info(f"✅ 게이트 #{self.gates_passed}/{TOTAL_GATES} 통과 완료!")
                self.gate_passing_state = 'SEARCHING'
                self.last_gate_position = None
            
        elif gate_y < COLOR_H * 0.35 and self.gate_passing_state == 'PASSING':
            # 새로운 게이트가 화면 상단에 나타남 = 이전 게이트 통과 완료
            self.gates_passed += 1
            self.gate_passing_state = 'SEARCHING' 
            self.get_logger().info(f"✅ 게이트 #{self.gates_passed}/{TOTAL_GATES} 통과 완료!")
            
        else:
            # 게이트 접근 중 - 중앙 정렬
            self.gate_passing_state = 'APPROACHING'
            error = gate_x - frame_cx
            
            # ⭐ 게이트 중앙으로 정확히 정렬
            if abs(error) <= GATE_CENTER_DEADZONE:
                self.get_logger().info("✅ 게이트 중앙 정렬 완료 → 전진")
                set_motor_state_named('forward')
                time.sleep(APPROACH_FORWARD_TIME)
            elif error > 0:
                self.get_logger().info(f"게이트가 우측에 {error}px → 우회전")
                set_motor_state_named('right')
                time.sleep(TURN_SMALL_TIME * (abs(error) / 100))  # 오차에 비례한 회전
            else:
                self.get_logger().info(f"게이트가 좌측에 {abs(error)}px → 좌회전")
                set_motor_state_named('left')
                time.sleep(TURN_SMALL_TIME * (abs(error) / 100))  # 오차에 비례한 회전
            set_motor_state_named('stop')
        
        self.last_gate_position = gate_center
    
    def continuous_forward(self, frame: np.ndarray):
        """연속 전진 로직 - 게이트 통과 직후 다음 게이트 탐색"""
        self.get_logger().info("🚀 연속 전진 모드 - 다음 게이트 탐색")
        
        # 전방 안전 확인 후 전진
        left, front, right = self.depth_sector_distances()
        
        if front > DEPTH_SAFE_DISTANCE and front != float('inf'):
            set_motor_state_named('forward')
            time.sleep(FORWARD_SPEED_TIME * 2)  # 일반 전진보다 길게
            self.continuous_forward_time += FORWARD_SPEED_TIME * 2
        else:
            # 장애물 있으면 회피
            if left > right:
                set_motor_state_named('left')
                time.sleep(TURN_SMALL_TIME * 0.5)
                set_motor_state_named('forward')
                time.sleep(FORWARD_SPEED_TIME)
            else:
                set_motor_state_named('right')
                time.sleep(TURN_SMALL_TIME * 0.5)
                set_motor_state_named('forward')
                time.sleep(FORWARD_SPEED_TIME)
        
        set_motor_state_named('stop')
        
        # 너무 오래 전진했으면 리셋
        if self.continuous_forward_time > 5.0:
            self.continuous_forward_time = 0
            self.gate_passing_state = 'SEARCHING'

    def scan_for_pair(self, visible_color: str, cx: int, frame: np.ndarray):
        """한쪽 부표만 보일 때 짝 찾기 위한 스캔"""
        current_time = time.time()
        
        if current_time - self.last_scan_time < 1.0:
            return
        
        self.last_scan_time = current_time
        
        # 보이는 부표의 반대편을 스캔
        if visible_color == 'GREEN':
            # 초록이 보이면 우측(빨강) 스캔
            self.get_logger().info("🟢 초록 감지 → 우측에서 빨강 찾기")
            set_motor_state_named('right')
            time.sleep(SCAN_TURN_TIME * 0.7)
        else:
            # 빨강이 보이면 좌측(초록) 스캔
            self.get_logger().info("🔴 빨강 감지 → 좌측에서 초록 찾기")
            set_motor_state_named('left')
            time.sleep(SCAN_TURN_TIME * 0.7)
        
        set_motor_state_named('stop')
    
    def single_color_scan(self, color: str, cx: int, frame: np.ndarray):
        current_time = time.time()
        
        if current_time - self.last_scan_time < 1.0:
            return
        
        self.last_scan_time = current_time
        
        self.get_logger().info(f"[SCAN] {color} 단독 감지 → {self.scan_direction} 방향으로 1초 스캔")
        
        if self.scan_direction == 'left':
            set_motor_state_named('left')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'right'
        else:
            set_motor_state_named('right')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'left'
        
        set_motor_state_named('stop')

    def depth_follow(self, frame: np.ndarray):
        """게이트나 부표가 없을 때, 깊이 기반 회피/전진 동작"""
        left, front, right = self.depth_sector_distances()
        
        self.get_logger().info(f"[DEPTH] L:{left:.2f} / F:{front:.2f} / R:{right:.2f}")
        if front > DEPTH_SAFE_DISTANCE and front != float('inf'):
            self.get_logger().info("전방 안전 → 전진")
            set_motor_state_named('forward')
            time.sleep(FORWARD_SPEED_TIME)
        elif left > right:
            self.get_logger().info("좌측 공간 여유 → 좌회전")
            set_motor_state_named('left')
            time.sleep(TURN_SMALL_TIME)
        else:
            self.get_logger().info("우측 공간 여유 → 우회전")
            set_motor_state_named('right')
            time.sleep(TURN_SMALL_TIME)
        
        set_motor_state_named('stop')
    
    def approach_yellow_and_wait(self, yellow_bb: Tuple[int, int, int, int], frame: np.ndarray) -> bool:
        """노란부표 접근 및 일정 거리 내 정지 대기"""
        x, y, w, h = yellow_bb
        cx = x + w // 2
        cy = y + h // 2
        
        depth = self.get_depth_at_point(cx, cy)
        
        if depth == 0 or np.isnan(depth) or depth > 20.0:
            self.get_logger().info(f"[YELLOW] 깊이 정보 없음/유효하지 않음 ({depth:.2f}m) → 정지")
            set_motor_state_named('stop')
            return False
        
        self.get_logger().info(f"[YELLOW] 노란부표 거리: {depth:.2f}m")
        
        if depth > YELLOW_STOP_DISTANCE:
            # 거리가 멀면, 중심으로 정렬하며 전진
            frame_cx = frame.shape[1] // 2
            if cx < frame_cx - GATE_CENTER_DEADZONE:
                self.get_logger().info("노란부표 좌측 → 좌회전")
                set_motor_state_named('left')
                time.sleep(TURN_SMALL_TIME)
            elif cx > frame_cx + GATE_CENTER_DEADZONE:
                self.get_logger().info("노란부표 우측 → 우회전")
                set_motor_state_named('right')
                time.sleep(TURN_SMALL_TIME)
            else:
                self.get_logger().info("5m 이상 → 접근 계속 (전진)")
                set_motor_state_named('forward')
                time.sleep(APPROACH_FORWARD_TIME)
            set_motor_state_named('stop')
            return False
        else:
            # 목표 거리 이내 도달
            self.get_logger().info("🟡 5m 이내 도달 → 정지 및 5초 대기")
            set_motor_state_named('stop')
            
            for i in range(int(YELLOW_WAIT_TIME), 0, -1):
                self.get_logger().info(f"⏱️  {i}초...")
                time.sleep(1)
            
            self.get_logger().info("✅ 5초 대기 완료!")
            return True

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