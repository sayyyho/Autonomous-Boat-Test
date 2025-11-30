#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KABOAT Phase1: Depth-based Gate Selection (깊이 기반 게이트 선택)
- 가장 가까운 게이트 쌍 우선 선택
- 면적 + Y좌표 기반 거리 추정
"""

import time
import serial
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ===========================
# 설정 파라미터
# ===========================

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
DEFAULT_SPEED = '5'

TOTAL_GATES = int(input("통과할 게이트 수 (기본 5): ") or "5")
print(f"✅ 총 {TOTAL_GATES}개의 게이트를 통과합니다.")

MODEL_PATH = './cone.pt'
CONFIDENCE_THRESHOLD = 0.5

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ⭐ 게이트 선택 기준
Y_ALIGNMENT_THRESHOLD = 100  # 수평 정렬 허용 오차 (넉넉하게)
MIN_CONE_AREA = 400  # 최소 콘 면적 (작게 → 원거리도 감지)
GATE_CENTER_DEADZONE = 50

# ⭐ 깊이 가중치
AREA_WEIGHT = 0.6  # 면적 비중
Y_WEIGHT = 0.4     # Y좌표 비중

# 타이밍
FORWARD_TIME = 0.3
TURN_TIME = 0.4
SCAN_TURN_TIME = 1.2
APPROACH_TIME = 0.6
GATE_PASS_TIME = 2.0

# ===========================
# 아두이노 모터 제어
# ===========================

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


# ===========================
# 카메라 초기화
# ===========================

def find_camera(max_index=10) -> Optional[cv2.VideoCapture]:
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"✅ 카메라 인덱스 {i}번 사용")
                return cap
            cap.release()
    return None


# ===========================
# YOLO 콘 검출기
# ===========================

class YOLOConeDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ 모델 파일 없음: {model_path}")
        
        print(f"📦 YOLO 모델 로딩: {model_path}")
        self.model = YOLO(str(model_path))
        print(f"✅ 모델 로드 완료 (신뢰도: {conf_threshold})")
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        red_cones = []
        green_cones = []
        
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                cls_name = r.names[cls_idx]
                confidence = float(box.conf[0])
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                
                area = w * h
                if area < MIN_CONE_AREA:
                    continue
                
                cx, cy = x1 + w // 2, y1 + h // 2
                
                # ⭐ 하단 Y좌표 (깊이 추정용)
                bottom_y = y2
                
                cone_data = {
                    'bbox': (x1, y1, w, h),
                    'conf': confidence,
                    'center': (cx, cy),
                    'area': area,
                    'bottom_y': bottom_y  # 추가
                }
                
                if cls_name == 'red_cone':
                    red_cones.append(cone_data)
                elif cls_name == 'green_cone':
                    green_cones.append(cone_data)
        
        return red_cones, green_cones


# ===========================
# ⭐ 깊이 기반 게이트 검출
# ===========================

def calculate_depth_score(cone: Dict, max_area: float, max_y: float) -> float:
    """
    깊이 점수 계산 (높을수록 가까움)
    
    Args:
        cone: 콘 정보
        max_area: 전체 콘 중 최대 면적
        max_y: 전체 콘 중 최대 Y좌표
    
    Returns:
        0~1 사이의 깊이 점수
    """
    # 면적 정규화 (0~1)
    area_score = cone['area'] / max_area if max_area > 0 else 0
    
    # Y좌표 정규화 (0~1)
    y_score = cone['bottom_y'] / max_y if max_y > 0 else 0
    
    # 가중 합산
    depth_score = AREA_WEIGHT * area_score + Y_WEIGHT * y_score
    
    return depth_score


def find_nearest_gate_pair(red_cones: List[Dict], 
                           green_cones: List[Dict],
                           frame_width: int,
                           frame_height: int) -> Optional[Tuple[Dict, Dict, float]]:
    """
    가장 가까운(depth 점수 높은) 게이트 쌍 찾기
    
    Returns:
        (red_cone, green_cone, depth_score) or None
    """
    if not red_cones or not green_cones:
        return None
    
    # 전체 콘에서 최대값 구하기 (정규화용)
    all_cones = red_cones + green_cones
    max_area = max(c['area'] for c in all_cones)
    max_y = max(c['bottom_y'] for c in all_cones)
    
    best_gate = None
    best_depth = -1
    
    for green in green_cones:
        green_cx, green_cy = green['center']
        
        for red in red_cones:
            red_cx, red_cy = red['center']
            
            # 조건 1: 초록(왼쪽) - 빨강(오른쪽) 배치
            if green_cx >= red_cx:
                continue
            
            # 조건 2: Y좌표 수평 정렬
            y_diff = abs(green_cy - red_cy)
            if y_diff > Y_ALIGNMENT_THRESHOLD:
                continue
            
            # ⭐ 조건 3: 게이트 쌍의 평균 깊이 점수 계산
            green_depth = calculate_depth_score(green, max_area, max_y)
            red_depth = calculate_depth_score(red, max_area, max_y)
            
            # 두 콘의 평균 깊이
            avg_depth = (green_depth + red_depth) / 2.0
            
            # 추가 보너스: 화면 중앙에 가까우면 가산점
            gate_cx = (green_cx + red_cx) // 2
            center_distance = abs(gate_cx - frame_width // 2)
            center_bonus = 1.0 - (center_distance / frame_width) * 0.2  # 최대 20% 가산
            
            final_score = avg_depth * center_bonus
            
            if final_score > best_depth:
                best_depth = final_score
                best_gate = (red, green, final_score)
    
    return best_gate


# ===========================
# 메인 네비게이터
# ===========================

class DepthBasedNavigator:
    def __init__(self, cap: cv2.VideoCapture, model_path: str):
        self.cap = cap
        self.motor = ArduinoMotorController()
        self.detector = YOLOConeDetector(model_path, CONFIDENCE_THRESHOLD)
        
        self.mission_stage = 'NAVIGATION'
        self.gates_passed = 0
        self.gate_state = 'SEARCHING'
        
        self.last_gate_seen = time.time()
        self.scan_direction = 'right'
        self.last_scan_time = 0
        
        self._t_prev = time.time()
        self._fps_smooth = None
        
        print("=" * 60)
        print("🚢 Depth-based Navigator 시작")
        print("=" * 60)
    
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
    
    def visualize_detections(self, frame: np.ndarray, 
                            red_cones: List[Dict], 
                            green_cones: List[Dict],
                            gate_info: Optional[Tuple[Dict, Dict, float]] = None):
        """검출 결과 시각화"""
        
        # 선택된 게이트 쌍
        selected_red = gate_info[0] if gate_info else None
        selected_green = gate_info[1] if gate_info else None
        
        # 초록 콘
        for cone in green_cones:
            x, y, w, h = cone['bbox']
            conf = cone['conf']
            cx, cy = cone['center']
            area = cone['area']
            
            is_selected = (selected_green and cone == selected_green)
            color = (0, 255, 255) if is_selected else (0, 255, 0)  # 선택되면 노란색
            thickness = 4 if is_selected else 2
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            
            label = f'G {conf:.2f} A:{area}'
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 빨강 콘
        for cone in red_cones:
            x, y, w, h = cone['bbox']
            conf = cone['conf']
            cx, cy = cone['center']
            area = cone['area']
            
            is_selected = (selected_red and cone == selected_red)
            color = (0, 255, 255) if is_selected else (0, 0, 255)
            thickness = 4 if is_selected else 2
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            
            label = f'R {conf:.2f} A:{area}'
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # ⭐ 선택된 게이트 강조
        if gate_info:
            red, green, depth_score = gate_info
            red_cx, red_cy = red['center']
            green_cx, green_cy = green['center']
            
            gate_cx = (red_cx + green_cx) // 2
            gate_cy = (red_cy + green_cy) // 2
            
            # 게이트 중심선
            cv2.line(frame, (gate_cx, 0), (gate_cx, CAMERA_HEIGHT), 
                    (0, 255, 255), 3)
            
            # 게이트 연결선
            cv2.line(frame, (green_cx, green_cy), (red_cx, red_cy), 
                    (255, 0, 255), 3)
            
            # 게이트 중심
            cv2.circle(frame, (gate_cx, gate_cy), 12, (0, 255, 255), -1)
            
            # ⭐ 깊이 점수 표시
            label = f"GATE #{self.gates_passed+1} | Depth: {depth_score:.2f}"
            cv2.putText(frame, label, (gate_cx-80, gate_cy-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 화면 중심선
        cv2.line(frame, (CAMERA_WIDTH//2, 0), 
                (CAMERA_WIDTH//2, CAMERA_HEIGHT), (255, 255, 255), 1)
        
        return frame
    
    def draw_info(self, frame: np.ndarray, 
                  red_count: int, green_count: int):
        cv2.putText(frame, 
                   f"Stage: {self.mission_stage} | Gates: {self.gates_passed}/{TOTAL_GATES}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Green: {green_count} | Red: {red_count}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"State: {self.gate_state}", 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        fps = self._update_fps()
        cv2.putText(frame, f"{fps:.1f} FPS", 
                   (20, CAMERA_HEIGHT - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # YOLO 검출
        red_cones, green_cones = self.detector.detect(frame)
        
        # ⭐ 깊이 기반 게이트 찾기
        gate_info = find_nearest_gate_pair(red_cones, green_cones, 
                                           CAMERA_WIDTH, CAMERA_HEIGHT)
        
        # 시각화
        frame = self.visualize_detections(frame, red_cones, green_cones, gate_info)
        frame = self.draw_info(frame, len(red_cones), len(green_cones))
        
        # 항법
        if self.mission_stage == 'NAVIGATION':
            self.navigation_logic(gate_info, frame)
        elif self.mission_stage == 'COMPLETE':
            self.complete_logic(frame)
        
        return frame
    
    def navigation_logic(self, gate_info: Optional[Tuple[Dict, Dict, float]], 
                        frame: np.ndarray):
        if self.gates_passed >= TOTAL_GATES:
            self.mission_stage = 'COMPLETE'
            return
        
        if gate_info:
            self.last_gate_seen = time.time()
            self.approach_and_pass_gate(gate_info, frame)
        else:
            self.search_gate()
    
    def approach_and_pass_gate(self, gate_info: Tuple[Dict, Dict, float], 
                               frame: np.ndarray):
        red, green, depth_score = gate_info
        red_cx, red_cy = red['center']
        green_cx, green_cy = green['center']
        
        gate_cx = (red_cx + green_cx) // 2
        gate_cy = (red_cy + green_cy) // 2
        
        frame_cx = CAMERA_WIDTH // 2
        
        # ⭐ 깊이 점수 기반 통과 판단 (점수 높으면 → 가까움)
        if depth_score > 0.6 or gate_cy > CAMERA_HEIGHT * 0.65:
            if self.gate_state != 'PASSING':
                self.gate_state = 'PASSING'
                print(f"🚪 게이트 #{self.gates_passed+1} 통과 (깊이: {depth_score:.2f})")
            
            error = gate_cx - frame_cx
            if abs(error) > GATE_CENTER_DEADZONE // 2:
                if error > 0:
                    self.motor.right()
                else:
                    self.motor.left()
                time.sleep(TURN_TIME * 0.3)
            
            self.motor.forward()
            time.sleep(GATE_PASS_TIME)
            self.motor.stop()
            
            self.gates_passed += 1
            print(f"✅ 게이트 #{self.gates_passed}/{TOTAL_GATES} 통과!")
            
            self.gate_state = 'SEARCHING'
            time.sleep(0.5)
        
        else:
            self.gate_state = 'APPROACHING'
            error = gate_cx - frame_cx
            
            if abs(error) <= GATE_CENTER_DEADZONE:
                print(f"→ 게이트 중앙 정렬 (깊이: {depth_score:.2f}) → 직진")
                self.motor.forward()
                time.sleep(APPROACH_TIME)
            elif error > 0:
                print(f"→ 우측 {error}px (깊이: {depth_score:.2f}) → 우회전")
                self.motor.right()
                time.sleep(TURN_TIME * min(abs(error)/100, 1.0))
                self.motor.forward()
                time.sleep(APPROACH_TIME * 0.5)
            else:
                print(f"→ 좌측 {abs(error)}px (깊이: {depth_score:.2f}) → 좌회전")
                self.motor.left()
                time.sleep(TURN_TIME * min(abs(error)/100, 1.0))
                self.motor.forward()
                time.sleep(APPROACH_TIME * 0.5)
            
            self.motor.stop()
    
    def search_gate(self):
        self.gate_state = 'SEARCHING'
        
        if time.time() - self.last_gate_seen < 2.0:
            self.motor.forward()
            time.sleep(FORWARD_TIME)
            self.motor.stop()
            return
        
        if time.time() - self.last_scan_time >= 2.0:
            self.scan_for_gate()
        else:
            self.motor.stop()
    
    def scan_for_gate(self):
        self.last_scan_time = time.time()
        print(f"🔍 [{self.scan_direction}] 스캔...")
        
        if self.scan_direction == 'left':
            self.motor.left()
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'right'
        else:
            self.motor.right()
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'left'
        
        self.motor.stop()
    
    def complete_logic(self, frame: np.ndarray):
        cv2.putText(frame, "MISSION COMPLETE!", 
                   (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (0, 255, 0), 3)
        self.motor.stop()
        print("🎉 Phase1 완료!")
    
    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 프레임 읽기 실패")
                    break
                
                processed = self.process_frame(frame)
                cv2.imshow("Depth-based Gate Navigator", processed)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n사용자 종료")
                    break
                elif key == ord('s'):
                    filename = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, processed)
                    print(f"📸 {filename}")
                
                if self.mission_stage == 'COMPLETE':
                    cv2.waitKey(3000)
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️  중단")
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.motor.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ 종료")


# ===========================
# 메인 실행
# ===========================

def main():
    print("\n" + "=" * 60)
    print("🚢 KABOAT Depth-based Gate Navigator")
    print("=" * 60)
    
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return
    
    cap = find_camera(10)
    if cap is None:
        print("❌ 카메라 없음")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    navigator = DepthBasedNavigator(cap, MODEL_PATH)
    navigator.run()


if __name__ == '__main__':
    main()