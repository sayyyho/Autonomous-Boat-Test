#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase1: Navigation through Gates + Yellow Buoy Stop + Forward to Dock Area (ROS2 Version)
- (New) Green-priority overlap: Overlapping red/green detections are treated as GREEN.
- (New) Aligned-gate logic: Only passes through horizontally-aligned (same Y-level) pairs.
- Depth fallback for navigation
- Yellow buoy detection: approach to within 5m, wait 5s, then move forward toward dock
"""

import time
import subprocess
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
GPIOSET_PATH = '/usr/bin/gpioset'
CHIP = 'gpiochip4'
MOTOR_A_FRONT = 19
MOTOR_A_BACK = 26
MOTOR_B_FRONT = 21
MOTOR_B_BACK = 20

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

# --- ⭐️ [로직 수정 1] 수평 정렬 게이트 허용 오차 (픽셀) ---
# 두 부표의 중심 Y좌표가 이 값 이내여야 게이트로 인정
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

# --- ⭐️ [로직 수정 2] HSV 범위 수정 ---
# "물색"을 감지하던 [40, 60, 60] 대신 [40, 100, 100]을 사용
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
# ---- 유틸리티 / 모터 제어 ---
# ----------------------------
def set_motor_state(a_f: int, a_b: int, b_f: int, b_b: int) -> None:
    cmd = [GPIOSET_PATH, CHIP,
           f"{MOTOR_A_FRONT}={a_f}", f"{MOTOR_A_BACK}={a_b}",
           f"{MOTOR_B_FRONT}={b_f}", f"{MOTOR_B_BACK}={b_b}"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def set_motor_state_named(state: str) -> None:
    state = state.lower()
    mapping = {
        'forward': (1, 0, 1, 0),
        'backward': (0, 1, 0, 1),
        'left': (0, 1, 1, 0),
        'right': (1, 0, 0, 1),
        'stop': (0, 0, 0, 0)
    }
    set_motor_state(*mapping.get(state, (0, 0, 0, 0)))

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

# --- ⭐️ [로직 수정 3] 2D 겹침(Overlap) 확인 함수 추가 ---
def has_overlap(bb1: Tuple[int, int, int, int], bb2: Tuple[int, int, int, int]) -> bool:
    """두 바운딩 박스(x, y, w, h)가 겹치는지 2D로 확인"""
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    
    # AABB (Axis-Aligned Bounding Box) 충돌 검사
    if (x1 < x2 + w2 and x1 + w1 > x2 and
        y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False

# --- ⭐️ [로직 수정 4] '초록 우선' 겹침 필터로 변경 ---
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

# --- ⭐️ [로직 수정 5] '수평 정렬' 게이트 찾기 로직으로 변경 ---
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
        green_cy = gy + gh // 2 # Y좌표 중심
        
        for red_bb in red_bbs:
            rx, ry, rw, rh = red_bb
            red_cx = rx + rw // 2
            red_cy = ry + rh // 2 # Y좌표 중심
            
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
        
        self.get_logger().info("=== Phase1 Navigator 시작 (ROS2) ===")

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
            # 뎁스 이미지 포맷에 따라 'z16' (uint16) 또는 '32FC1' (float32) 일 수 있음
            # Realsense ROS 래퍼는 보통 'z16' (mm 단위)
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
                if dist > 0.1 and dist < 20.0: # 유효 거리 (0.1m ~ 20m)
                    vals.append(dist)
            return min(vals) if vals else float('inf')
        
        left = sector(max(0, cx - 3*DEPTH_SECTOR_WIDTH), max(0, cx - DEPTH_SECTOR_WIDTH))
        front = sector(max(0, cx - DEPTH_SECTOR_WIDTH), min(COLOR_W, cx + DEPTH_SECTOR_WIDTH))
        right = sector(min(COLOR_W-1, cx + DEPTH_SECTOR_WIDTH), min(COLOR_W, cx + 3*DEPTH_SECTOR_WIDTH))
        
        return left, front, right

    def auto_scan_for_gate(self):
        """C자형 트랙 대응: 자동 좌우 스캔"""
        current_time = time.time()
        
        # 게이트를 오래 못 봤거나 주기적 스캔 시간이 되면
        if (current_time - self.last_gate_seen_time > GATE_LOST_THRESHOLD or 
            current_time - self.last_auto_scan_time > CONTINUOUS_SCAN_INTERVAL):
            
            self.last_auto_scan_time = current_time
            self.get_logger().info(f"[AUTO SCAN] {self.scan_direction} 방향으로 게이트 탐색")
            
            if self.scan_direction == 'left':
                set_motor_state_named('left')
                time.sleep(SCAN_TURN_TIME * 0.5)
                self.scan_direction = 'right'
            else:
                set_motor_state_named('right')
                time.sleep(SCAN_TURN_TIME * 0.5)
                self.scan_direction = 'left'
            
            set_motor_state_named('stop')

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

        # 🔴🟢 (수정) '초록 우선' 2D 겹침 필터 적용
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

        # 🟡 노란부표 처리 - 게이트가 없을 때만 감지
        if yellow_bb and not (red_bbs or green_bbs):
            self.last_gate_seen_time = time.time() # 노란 부표도 '표식'으로 간주하여 스캔 타이머 리셋
            self.get_logger().info("🟡 노란부표 감지")
            x, y, w, h = yellow_bb
            cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(color_img, "YELLOW", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            if self.approach_yellow_and_wait(yellow_bb, color_img):
                self.get_logger().info(f"🚢 도킹 구역으로 {AFTER_YELLOW_FORWARD_TIME}초 전진")
                set_motor_state_named('forward')
                time.sleep(AFTER_YELLOW_FORWARD_TIME)
                set_motor_state_named('stop')
                self.get_logger().info("✅ Phase1 완료")
                self.mission_complete = True

        # 🟥🟩 게이트 처리: (수정) '수평 정렬'된 쌍 찾기
        elif red_bbs and green_bbs:
            self.last_gate_seen_time = time.time()
            
            gate_pair = find_closest_gate_pair(red_bbs, green_bbs, color_img.shape[1])
            
            if gate_pair:
                red_bb, green_bb = gate_pair
                
                rx, ry, rw, rh = red_bb
                gx, gy, gw, gh = green_bb
                
                red_cx, green_cx = rx + rw//2, gx + gw//2
                gate_center = (red_cx + green_cx)//2
                
                # 선택된 게이트 쌍 강조 표시
                cv2.rectangle(color_img, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 3)
                cv2.rectangle(color_img, (gx, gy), (gx+gw, gy+gh), (0, 255, 0), 3)
                cv2.line(color_img, (gate_center, 0), (gate_center, COLOR_H), (255, 255, 0), 2)
                cv2.putText(color_img, "GATE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                self.approach_gate(gate_center, color_img)
            else:
                # --- ⭐️ [로직 수정 6] 유효 게이트 없을 시 Depth Follow ---
                self.get_logger().info("부표는 보이나 유효한 수평 게이트가 없음 -> Depth Follow")
                cv2.putText(color_img, "No Aligned Gate -> Depth", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                self.depth_follow(color_img)
            
        elif red_bbs or green_bbs:
            # 한쪽 부표만 보일 때
            self.last_gate_seen_time = time.time() # 한쪽이라도 보이면 타이머 리셋
            visible_color = 'RED' if red_bbs else 'GREEN'
            bb = red_bbs[0] if red_bbs else green_bbs[0]
            
            cx = bb[0] + bb[2]//2
            cv2.putText(color_img, f"SCANNING for {visible_color}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            self.single_color_scan(visible_color, cx, color_img)
        
        else:
            # --- ⭐️ [로직 수정 7] 부표가 아예 없을 시 Depth Follow ---
            # (auto_scan_for_gate 대신 depth_follow로 변경하여 안전성 확보)
            cv2.putText(color_img, "No Buoys -> Depth Follow", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.depth_follow(color_img)
            
            # C-트랙 등을 위한 주기적 스캔은 별도로 계속 확인
            self.auto_scan_for_gate()
        
        cv2.imshow("Phase1 View", color_img)
        cv2.waitKey(1)

    def approach_gate(self, gate_center: int, frame: np.ndarray):
        frame_cx = frame.shape[1]//2
        error = gate_center - frame_cx
        if abs(error) <= GATE_CENTER_DEADZONE:
            self.get_logger().info("게이트 중앙 정렬 완료 → 전진")
            set_motor_state_named('forward'); time.sleep(APPROACH_FORWARD_TIME)
        elif error > 0:
            self.get_logger().info("게이트 우측 → 우회전")
            set_motor_state_named('right'); time.sleep(TURN_SMALL_TIME)
        else:
            self.get_logger().info("게이트 좌측 → 좌회전")
            set_motor_state_named('left'); time.sleep(TURN_SMALL_TIME)
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
        
        if depth == 0 or np.isnan(depth) or depth > 20.0: # 유효 거리 20m 초과 시 무시
            self.get_logger().info(f"[YELLOW] 깊이 정보 없음/유효하지 않음 ({depth:.2f}m) → 정지")
            set_motor_state_named('stop')
            return False
        
        self.get_logger().info(f"[YELLOW] 노란부표 거리: {depth:.2f}m")
        
        if depth > YELLOW_STOP_DISTANCE:
            # 거리가 멀면, 중심으로 정렬하며 전진
            frame_cx = frame.shape[1] // 2
            if cx < frame_cx - GATE_CENTER_DEADZONE:
                self.get_logger().info("노란부표 좌측 → 좌회전")
                set_motor_state_named('left'); time.sleep(TURN_SMALL_TIME)
            elif cx > frame_cx + GATE_CENTER_DEADZONE:
                self.get_logger().info("노란부표 우측 → 우회전")
                set_motor_state_named('right'); time.sleep(TURN_SMALL_TIME)
            else:
                self.get_logger().info("5m 이상 → 접근 계속 (전진)")
                set_motor_state_named('forward'); time.sleep(APPROACH_FORWARD_TIME)
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
        set_motor_state_named('stop')
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()