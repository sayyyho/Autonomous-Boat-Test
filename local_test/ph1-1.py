#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase1: 게이트 통과 (완전 개선 버전)
- 웹캠 사용
- ROS2 없이 실행 가능
- 초록 우선 검출 + 수평 정렬 게이트
- 모폴로지 연산으로 노이즈 제거
- FPS 실시간 표시
- 스캔 쿨다운 적용
"""

import time
import serial
from typing import List, Tuple, Optional
import cv2
import numpy as np

# ----------------------------
# ---- 설정 파라미터 ----
# ----------------------------
SERIAL_PORT = '/dev/ttyACM0'  # Windows: 'COM3', Linux: '/dev/ttyACM0'
BAUD_RATE = 9600
DEFAULT_SPEED = '5'

TOTAL_GATES = int(input("통과해야 할 게이트 수를 입력하세요 (기본 5): ") or "5")
print(f"✅ 총 {TOTAL_GATES}개의 게이트를 통과합니다.")

COLOR_W, COLOR_H = 640, 480

# ⭐ HSV 범위 (개선된 버전)
HSV_GREEN_LOWER = np.array([35, 70, 70])
HSV_GREEN_UPPER = np.array([85, 255, 255])

HSV_RED_LOWER1 = np.array([0, 120, 70])
HSV_RED_UPPER1 = np.array([10, 255, 255])
HSV_RED_LOWER2 = np.array([170, 120, 70])
HSV_RED_UPPER2 = np.array([180, 255, 255])

HSV_YELLOW_LOWER = np.array([22, 120, 120])
HSV_YELLOW_UPPER = np.array([32, 255, 255])

# 최소 면적
MIN_AREA_GREEN = 500
MIN_AREA_RED = 500
MIN_AREA_YELLOW = 1000

# ⭐ 모폴로지 연산용 커널
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 초록 우선 판단
OVERLAP_EXPANSION = 1.5

# 수평 정렬 허용 오차
Y_ALIGNMENT_THRESHOLD = 75

# 게이트 중심 데드존
GATE_CENTER_DEADZONE = 40

# 타이밍
FORWARD_TIME = 0.3
TURN_SMALL_TIME = 0.4
SCAN_TURN_TIME = 1.0
APPROACH_TIME = 0.5

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
            # 시뮬레이션 모드
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
# ---- 카메라 초기화 ----
# ----------------------------
def find_camera(max_index=10):
    """사용 가능한 카메라 찾기"""
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"✅ 카메라 찾음! 인덱스 {i}번을 사용합니다.")
                return cap
            else:
                cap.release()
    return None

# ----------------------------
# ---- 비전 처리 함수 ----
# ----------------------------
def expand_bbox(bbox: Tuple[int, int, int, int], factor: float) -> Tuple[int, int, int, int]:
    """바운딩 박스를 중심 기준으로 확장"""
    x, y, w, h = bbox
    cx, cy = x + w//2, y + h//2
    
    new_w = int(w * factor)
    new_h = int(h * factor)
    
    new_x = max(0, cx - new_w//2)
    new_y = max(0, cy - new_h//2)
    
    return (new_x, new_y, new_w, new_h)

def is_overlapping(bb1: Tuple[int, int, int, int], bb2: Tuple[int, int, int, int]) -> bool:
    """두 바운딩 박스가 겹치는지 확인"""
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    
    return (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2 and y1 + h1 > y2)

def detect_buoys_with_green_priority(hsv: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], 
                                                                 List[Tuple[int, int, int, int]]]:
    """
    초록 우선 부표 검출 (모폴로지 연산 적용)
    1. 초록색 먼저 검출
    2. 빨간색 검출
    3. 초록과 겹치는 빨강은 제거 (같은 꼬깔의 빨간 부분)
    """
    # 1. 초록색 검출
    green_mask = cv2.inRange(hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER)
    # ⭐ 모폴로지 연산으로 노이즈 제거
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)
    
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    green_bbs = []
    for cnt in green_contours:
        if cv2.contourArea(cnt) > MIN_AREA_GREEN:
            green_bbs.append(cv2.boundingRect(cnt))
    
    # 2. 빨간색 검출
    red_mask1 = cv2.inRange(hsv, HSV_RED_LOWER1, HSV_RED_UPPER1)
    red_mask2 = cv2.inRange(hsv, HSV_RED_LOWER2, HSV_RED_UPPER2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    # ⭐ 모폴로지 연산으로 노이즈 제거
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)
    
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    red_bbs_raw = []
    for cnt in red_contours:
        if cv2.contourArea(cnt) > MIN_AREA_RED:
            red_bbs_raw.append(cv2.boundingRect(cnt))
    
    # ⭐ 3. 초록과 겹치는 빨강 제거 (초록 우선 원칙)
    red_bbs_filtered = []
    
    for red_bb in red_bbs_raw:
        is_green_buoy = False
        
        for green_bb in green_bbs:
            # 초록 영역을 확장하여 겹침 판단 (같은 꼬깔인지 확인)
            expanded_green = expand_bbox(green_bb, OVERLAP_EXPANSION)
            
            if is_overlapping(red_bb, expanded_green):
                # 이 빨강은 초록 꼬깔의 일부임 → 제거
                is_green_buoy = True
                break
        
        if not is_green_buoy:
            # 순수 빨강 부표만 추가
            red_bbs_filtered.append(red_bb)
    
    return red_bbs_filtered, green_bbs

def detect_yellow(hsv: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """노란색 객체 검출 (모폴로지 연산 적용)"""
    mask = cv2.inRange(hsv, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)
    # ⭐ 모폴로지 연산으로 노이즈 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA_YELLOW]
    if not valid:
        return None
    
    largest = max(valid, key=cv2.contourArea)
    return cv2.boundingRect(largest)

def find_horizontal_gate_pair(red_bbs: List[Tuple[int, int, int, int]], 
                               green_bbs: List[Tuple[int, int, int, int]],
                               frame_width: int) -> Optional[Tuple[Tuple, Tuple]]:
    """수평 정렬된 게이트 쌍 찾기"""
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
            
            # 조건 1: 좌=초록, 우=빨강
            if green_cx >= red_cx:
                continue
            
            # 조건 2: Y좌표 수평 정렬
            if abs(green_cy - red_cy) > Y_ALIGNMENT_THRESHOLD:
                continue
            
            gate_center_x = (green_cx + red_cx) // 2
            distance = abs(gate_center_x - frame_center)
            
            if distance < min_distance:
                min_distance = distance
                best_pair = (red_bb, green_bb)
    
    return best_pair

# ----------------------------
# ---- 메인 네비게이터 ----
# ----------------------------
class Phase1Navigator:
    def __init__(self, cap):
        self.cap = cap
        self.motor = init_motor()
        
        # 미션 상태
        self.mission_stage = 'NAVIGATION'
        self.gates_passed = 0
        self.gate_passing_state = 'SEARCHING'
        
        self.last_gate_seen = time.time()
        self.scan_direction = 'right'
        self.last_scan_time = 0  # ⭐ 마지막 스캔 시간
        
        # ⭐ FPS 추적
        self._t_prev = time.time()
        self._fps_smooth = None
        
        print("=== Phase1 Navigator 시작 (완전 개선 버전) ===")
    
    def _update_fps(self) -> float:
        """FPS 계산 및 지수평활"""
        t = time.time()
        dt = t - self._t_prev
        self._t_prev = t
        fps = 1.0 / dt if dt > 1e-6 else 0.0
        
        if self._fps_smooth is None:
            self._fps_smooth = fps
        else:
            self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * fps
        
        return self._fps_smooth

    def process_frame(self, frame):
        """프레임 처리"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 초록 우선 검출
        red_bbs, green_bbs = detect_buoys_with_green_priority(hsv)
        yellow_bb = detect_yellow(hsv)
        
        # 디버그 시각화
        for bb in green_bbs:
            x, y, w, h = bb
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, "GREEN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for bb in red_bbs:
            x, y, w, h = bb
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(frame, "RED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if yellow_bb:
            x, y, w, h = yellow_bb
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
            cv2.putText(frame, "YELLOW", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 미션 정보 표시
        cv2.putText(frame, f"Stage: {self.mission_stage} | Gates: {self.gates_passed}/{TOTAL_GATES}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Green: {len(green_bbs)} | Red: {len(red_bbs)}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ⭐ FPS 표시
        fps = self._update_fps()
        cv2.putText(frame, f"{fps:5.1f} FPS", (20, COLOR_H - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        
        if self.mission_stage == 'NAVIGATION':
            self.navigation_stage(red_bbs, green_bbs, yellow_bb, frame)
        elif self.mission_stage == 'COMPLETE':
            self.complete_stage(frame)
        
        return frame

    def navigation_stage(self, red_bbs, green_bbs, yellow_bb, frame):
        """항법 단계"""
        
        if self.gates_passed >= TOTAL_GATES:
            cv2.putText(frame, f"All {TOTAL_GATES} gates passed!", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.mission_stage = 'COMPLETE'
            return
        
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
            
            # 게이트 강조
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 0, 255), 3)
            cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (255, 0, 255), 3)
            cv2.line(frame, (gate_cx, 0), (gate_cx, COLOR_H), (0, 255, 255), 2)
            cv2.circle(frame, (gate_cx, gate_cy), 10, (0, 255, 255), -1)
            
            cv2.putText(frame, f"GATE #{self.gates_passed+1}", (gate_cx-50, gate_cy-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            self.pass_through_gate(gate_cx, gate_cy, frame)
        
        else:
            cv2.putText(frame, f"Searching Gate #{self.gates_passed+1}...", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if time.time() - self.last_gate_seen < 2.0:
                motor_action('forward')
                time.sleep(FORWARD_TIME)
            else:
                # ⭐ 스캔 (쿨다운 적용됨)
                self.scan_for_buoy()
                
                # 스캔 중이 아니면 정지 상태 유지
                if time.time() - self.last_scan_time > 0.5:
                    motor_action('stop')
            
            motor_action('stop')

    def pass_through_gate(self, gate_cx: int, gate_cy: int, frame: np.ndarray):
        """게이트 통과"""
        frame_cx = frame.shape[1] // 2
        
        if gate_cy > COLOR_H * 0.65:
            if self.gate_passing_state != 'PASSING':
                self.gate_passing_state = 'PASSING'
                print(f"🚪 게이트 #{self.gates_passed+1} 통과 시작")
            
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
            
            self.gates_passed += 1
            print(f"✅ 게이트 #{self.gates_passed}/{TOTAL_GATES} 통과!")
            self.gate_passing_state = 'SEARCHING'
        
        else:
            self.gate_passing_state = 'APPROACHING'
            error = gate_cx - frame_cx
            
            if abs(error) <= GATE_CENTER_DEADZONE:
                print("✅ 게이트 중앙 정렬 → 직진")
                motor_action('forward')
                time.sleep(APPROACH_TIME)
            elif error > 0:
                print(f"우측 {error}px → 우회전")
                motor_action('right')
                time.sleep(TURN_SMALL_TIME * min(abs(error)/100, 1.0))
            else:
                print(f"좌측 {abs(error)}px → 좌회전")
                motor_action('left')
                time.sleep(TURN_SMALL_TIME * min(abs(error)/100, 1.0))
            
            motor_action('stop')

    def scan_for_buoy(self):
        """부표 스캔 (최소 2초 간격)"""
        current_time = time.time()
        
        # ⭐ 마지막 스캔으로부터 2초 미만이면 스킵
        if current_time - self.last_scan_time < 2.0:
            return
        
        self.last_scan_time = current_time
        print(f"🔍 [{self.scan_direction}] 스캔 시작...")
        
        if self.scan_direction == 'left':
            motor_action('left')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'right'
        else:
            motor_action('right')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'left'
        
        motor_action('stop')
        print(f"✅ 스캔 완료")

    def complete_stage(self, frame):
        """미션 완료"""
        cv2.putText(frame, "MISSION COMPLETE!", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        motor_action('stop')
        print("✅ Phase1 완료!")

    def run(self):
        """메인 루프"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 프레임 읽기 실패")
                    break
                
                # 프레임 처리
                processed = self.process_frame(frame)
                
                # 화면 표시
                cv2.imshow("Phase1 Navigator", processed)
                
                # 종료 조건
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("사용자 종료 요청")
                    break
                elif self.mission_stage == 'COMPLETE':
                    cv2.waitKey(3000)
                    break
        
        except KeyboardInterrupt:
            print("\n키보드 인터럽트")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """정리"""
        if self.motor:
            self.motor.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ 프로그램 종료")

# ----------------------------
# ---- 메인 실행 ----
# ----------------------------
def main():
    # 카메라 초기화
    cap = find_camera(10)
    if cap is None:
        print("❌ 카메라를 찾을 수 없습니다.")
        return
    
    # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, COLOR_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, COLOR_H)
    
    # 네비게이터 실행
    navigator = Phase1Navigator(cap)
    navigator.run()

if __name__ == '__main__':
    main()