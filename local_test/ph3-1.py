#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase2: 탐색 선회 (로컬 실행 버전)
- 웹캠 사용
- ROS2 없이 실행 가능
- 시간 기반 360도 선회
"""

import time
import serial
from typing import Optional, Tuple
import cv2
import numpy as np

# ----------------------------
# ---- 설정 파라미터 ----
# ----------------------------
SERIAL_PORT = '/dev/ttyACM0'  # Windows: 'COM3', Linux: '/dev/ttyACM0'
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

# HSV 범위 (개선된 버전)
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

# ⭐ 시간 기반 선회 설정
CIRCLE_TIME_PER_90DEG = 2.0
FULL_CIRCLE_TIME = CIRCLE_TIME_PER_90DEG * 4

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
# ---- 검색 네비게이터 ----
# ----------------------------
class SearchNavigator:
    def __init__(self, cap):
        self.cap = cap
        self.motor = init_motor()
        
        # 미션 상태
        self.mission_stage = 'SEARCHING'  # SEARCHING -> APPROACHING -> CIRCLING -> COMPLETE
        self.rotation_tracker = TimeBasedRotation(FULL_CIRCLE_TIME)
        self.last_buoy_seen = time.time()
        self.scan_direction = 'right'
        self.last_scan_time = 0  # 마지막 스캔 시간
        
        # ⭐ FPS 추적
        self._t_prev = time.time()
        self._fps_smooth = None
        
        print(f"=== 탐색 미션 시작: {TARGET_COLOR.upper()} 부표 ===")
        print(f"=== 선회 방향: {DIRECTION_TEXT} ===")
    
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
        
        # ⭐ FPS 표시
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
        
        return frame

    def searching_stage(self, target_bb, frame):
        """목표 부표 탐색"""
        if target_bb:
            self.last_buoy_seen = time.time()
            print(f"✅ {TARGET_COLOR.upper()} 부표 발견!")
            self.mission_stage = 'APPROACHING'
        else:
            cv2.putText(frame, f"Searching {TARGET_COLOR.upper()} buoy...", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 스캔 (쿨다운 적용됨)
            self.scan_for_buoy()
            
            # 스캔 중이 아니면 정지 상태 유지
            if time.time() - self.last_scan_time > 0.5:
                motor_action('stop')

    def approaching_stage(self, target_bb, frame):
        """목표 부표 접근"""
        if not target_bb:
            # 부표 놓쳤을 때
            if time.time() - self.last_buoy_seen < 2.0:
                motor_action('forward')
                time.sleep(FORWARD_TIME)
                motor_action('stop')
            else:
                print("❌ 부표 놓침 → 재탐색")
                self.mission_stage = 'SEARCHING'
            return
        
        self.last_buoy_seen = time.time()
        
        x, y, w, h = target_bb
        cx = x + w // 2
        
        # 부표 크기로 거리 추정 (간단한 방법)
        area = w * h
        
        # 크기가 충분히 크면 선회 시작
        if area > 15000:  # 임계값 조정 필요
            print(f"🎯 목표 거리 도달 → 선회 시작!")
            self.start_circling()
            return
        
        cv2.putText(frame, f"Approaching... (area: {area})", 
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
        
        print(f"🔄 {DIRECTION_TEXT} 선회 시작!")

    def circling_stage(self, target_bb, frame):
        """선회 동작"""
        
        # ⭐ 시간 기반 360도 완료 체크
        if self.rotation_tracker.is_complete():
            elapsed = self.rotation_tracker.get_elapsed()
            print(f"✅ 선회 완료! 소요 시간: {elapsed:.1f}초")
            self.rotation_tracker.stop()
            self.mission_stage = 'COMPLETE'
            motor_action('stop')
            return
        
        # 진행률 표시
        cv2.putText(frame, f"Circling {DIRECTION_TEXT}...", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ⭐ 선회 방향에 따라 순수 회전만 수행
        if CLOCKWISE:
            motor_action('right')  # 'd'만 전송
        else:
            motor_action('left')   # 'a'만 전송

    def scan_for_buoy(self):
        """부표 스캔 (최소 2초 간격)"""
        current_time = time.time()
        
        # 마지막 스캔으로부터 2초 미만이면 스킵
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
        print("🎉 탐색 미션 완료!")

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
                cv2.imshow("Search Navigator", processed)
                
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
    navigator = SearchNavigator(cap)
    navigator.run()

if __name__ == '__main__':
    main()