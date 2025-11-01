#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase1: Navigation through Gates + Yellow Buoy Stop + Forward to Dock Area
- Red/Green Gate detection and passage
- Depth fallback for navigation
- Yellow buoy detection: approach to within 5m, wait 5s, then move forward toward dock
"""

import time
import subprocess
import sys
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
import pyrealsense2 as rs
from collections import deque

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
MIN_CONTOUR_AREA = 300
GATE_CENTER_DEADZONE = 40
DEPTH_SECTOR_WIDTH = 60
DEPTH_SAMPLE_Y = int(COLOR_H * 0.5)
YELLOW_STOP_DISTANCE = 5.0  # m
YELLOW_WAIT_TIME = 5.0
AFTER_YELLOW_FORWARD_TIME = 3.0  # 노란 부표 후 도킹 방향으로 전진 시간

HSV_RANGES: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
    'RED': [
        (np.array([0, 120, 100]), np.array([10, 255, 255])),
        (np.array([160, 120, 100]), np.array([180, 255, 255]))
    ],
    'GREEN': [
        # 스카이블루/청록색 범위 (물 제외)
        # H: 85-100 (청록~하늘색), S: 80-255 (높은 채도로 물 제외), V: 100-255 (밝기)
        (np.array([85, 80, 100]), np.array([100, 255, 255]))
    ],
    'YELLOW': [
        (np.array([20, 120, 120]), np.array([35, 255, 255]))
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
# ---- Vision & Depth 유틸 ---
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

def find_all_contours(mask: np.ndarray, min_area: int = MIN_CONTOUR_AREA) -> List[Tuple[int, int, int, int]]:
    """여러 개의 컨투어를 모두 찾아 반환"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    
    valid_bbs = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            valid_bbs.append(cv2.boundingRect(cnt))
    
    return valid_bbs

def find_largest_contour_center(mask: np.ndarray, min_area: int = MIN_CONTOUR_AREA) -> Optional[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    return cv2.boundingRect(largest)

def depth_sector_distances(depth_frame, color_width: int) -> Tuple[float, float, float]:
    cx = color_width // 2
    y = DEPTH_SAMPLE_Y
    def sector(px_start, px_end):
        vals = [depth_frame.get_distance(px, y) for px in range(px_start, px_end)]
        vals = [v for v in vals if v > 0]
        return min(vals) if vals else float('inf')
    left = sector(max(0, cx - 3*DEPTH_SECTOR_WIDTH), max(0, cx - DEPTH_SECTOR_WIDTH))
    front = sector(max(0, cx - DEPTH_SECTOR_WIDTH), min(color_width, cx + DEPTH_SECTOR_WIDTH))
    right = sector(min(color_width-1, cx + DEPTH_SECTOR_WIDTH), min(color_width, cx + 3*DEPTH_SECTOR_WIDTH))
    return left, front, right

# ----------------------------
# ---- Phase1 Navigator ------
# ----------------------------
class Phase1Navigator:
    def __init__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, COLOR_W, COLOR_H, rs.format.z16, 30)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.state = 'SEARCH_GATE'
        self.gate_last_center = None
        self.loop_delay = 0.05
        self.scan_direction = 'right'
        self.last_scan_time = 0


    def run(self):
        print("=== Phase1 시작 (Gate + Yellow Detection + Move to Dock) ===")
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                
                color_img = np.asanyarray(color_frame.get_data())
                hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

                # --- 색상 감지 (여러 개 찾기) ---
                mask_red = mask_for_color(hsv, 'RED')
                mask_green = mask_for_color(hsv, 'GREEN')
                mask_yellow = mask_for_color(hsv, 'YELLOW')

                red_bbs = find_all_contours(mask_red)
                green_bbs = find_all_contours(mask_green)
                yellow_bb = find_largest_contour_center(mask_yellow)

                # 디버그용: 모든 감지된 부표 표시
                for bb in red_bbs:
                    x, y, w, h = bb
                    cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(color_img, "RED", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                for bb in green_bbs:
                    x, y, w, h = bb
                    cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(color_img, "SKYBLUE", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # 노란부표 감지 시 -> 5초 대기 후 전진
                if yellow_bb:
                    print("🟡 노란부표 감지: 접근 및 정지 단계 진입")
                    x, y, w, h = yellow_bb
                    cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(color_img, "YELLOW", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # 노란 부표 접근 → 5초 대기 → 전진
                    if self._approach_yellow_and_wait(yellow_bb, depth_frame, color_img):
                        # 5초 대기 완료 후 도킹 방향으로 전진
                        print(f"🚢 도킹 구역으로 {AFTER_YELLOW_FORWARD_TIME}초 전진 시작")
                        set_motor_state_named('forward')
                        time.sleep(AFTER_YELLOW_FORWARD_TIME)
                        set_motor_state_named('stop')
                        print("✅ Phase1 완료 - 도킹 구역 도착")
                        break

                # 🟥🔵 게이트 추종 (양쪽 다 보일 때)
                if red_bbs and green_bbs:
                    red_bb = max(red_bbs, key=lambda bb: bb[2]*bb[3])
                    green_bb = max(green_bbs, key=lambda bb: bb[2]*bb[3])
                    
                    rx, ry, rw, rh = red_bb
                    gx, gy, gw, gh = green_bb
                    
                    red_cx, green_cx = rx + rw//2, gx + gw//2
                    gate_center = (red_cx + green_cx)//2
                    
                    cv2.line(color_img, (gate_center, 0), (gate_center, COLOR_H), (255, 255, 0), 2)
                    cv2.putText(color_img, "GATE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    self._approach_gate(gate_center, color_img)
                    
                elif red_bbs or green_bbs:
                    visible_color = 'RED' if red_bbs else 'SKYBLUE'
                    bb = red_bbs[0] if red_bbs else green_bbs[0]
                    
                    cx = bb[0] + bb[2]//2
                    cv2.putText(color_img, f"SCANNING for {visible_color}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    self._single_color_scan(visible_color, cx, color_img, depth_frame)
                
                else:
                    cv2.putText(color_img, "Depth Following", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    self._depth_follow(depth_frame, color_img)
                
                cv2.imshow("Phase1 View", color_img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(self.loop_delay)

        except KeyboardInterrupt:
            print("종료 명령 수신")
        finally:
            set_motor_state_named('stop')
            self.pipeline.stop()
            cv2.destroyAllWindows()


    # ---- 게이트 접근 ----
    def _approach_gate(self, gate_center: int, frame: np.ndarray):
        frame_cx = frame.shape[1]//2
        error = gate_center - frame_cx
        if abs(error) <= GATE_CENTER_DEADZONE:
            print("게이트 중앙 정렬 완료 → 전진")
            set_motor_state_named('forward'); time.sleep(APPROACH_FORWARD_TIME)
        elif error > 0:
            print("게이트 우측 → 우회전")
            set_motor_state_named('right'); time.sleep(TURN_SMALL_TIME)
        else:
            print("게이트 좌측 → 좌회전")
            set_motor_state_named('left'); time.sleep(TURN_SMALL_TIME)
        set_motor_state_named('stop')

    # ---- 한쪽색만 있을 때 스캔 (1초 텀) ----
    def _single_color_scan(self, color: str, cx: int, frame: np.ndarray, depth_frame):
        current_time = time.time()
        
        if current_time - self.last_scan_time < 1.0:
            return
        
        self.last_scan_time = current_time
        
        print(f"[SCAN] {color} 단독 감지 → {self.scan_direction} 방향으로 1초 스캔")
        
        if self.scan_direction == 'left':
            set_motor_state_named('left')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'right'
        else:
            set_motor_state_named('right')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'left'
        
        set_motor_state_named('stop')

    # ---- Depth fallback ----
    def _depth_follow(self, depth_frame, frame: np.ndarray):
        left, front, right = depth_sector_distances(depth_frame, frame.shape[1])
        
        if front > DEPTH_SAFE_DISTANCE:
            set_motor_state_named('forward'); time.sleep(FORWARD_SPEED_TIME)
        elif left > right:
            set_motor_state_named('left'); time.sleep(TURN_SMALL_TIME)
        else:
            set_motor_state_named('right'); time.sleep(TURN_SMALL_TIME)
        set_motor_state_named('stop')

    # ---- 노란부표 접근 및 5초 대기 (True 반환 시 대기 완료) ----
    def _approach_yellow_and_wait(self, yellow_bb, depth_frame, frame: np.ndarray) -> bool:
        x, y, w, h = yellow_bb
        cx = x + w // 2
        frame_cx = frame.shape[1] // 2

        # 중심 정렬
        if cx < frame_cx - GATE_CENTER_DEADZONE:
            print("노란부표 좌측 → 좌회전")
            set_motor_state_named('left'); time.sleep(TURN_SMALL_TIME)
            set_motor_state_named('stop')
            return False
        elif cx > frame_cx + GATE_CENTER_DEADZONE:
            print("노란부표 우측 → 우회전")
            set_motor_state_named('right'); time.sleep(TURN_SMALL_TIME)
            set_motor_state_named('stop')
            return False
        else:
            # 거리 계산
            dist = depth_frame.get_distance(cx, y + h//2)
            print(f"노란부표 거리: {dist:.2f}m")
            
            if dist > YELLOW_STOP_DISTANCE:
                print("5m 이상 → 접근 계속")
                set_motor_state_named('forward'); time.sleep(FORWARD_SPEED_TIME)
                set_motor_state_named('stop')
                return False
            else:
                print("🟡 5m 이내 도달 → 정지 및 5초 대기")
                set_motor_state_named('stop')
                
                # 5초 카운트다운
                for i in range(5, 0, -1):
                    print(f"⏱️  {i}초...")
                    time.sleep(1)
                
                print("✅ 5초 대기 완료!")
                return True  # 대기 완료, 전진 신호

# ----------------------------
# ---- 실행 -----------------
# ----------------------------
if __name__ == '__main__':
    nav = Phase1Navigator()
    nav.run()