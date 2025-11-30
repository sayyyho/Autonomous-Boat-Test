#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KABOAT Phase1: YOLO-based Gate Navigation with Collision Avoidance
- YOLO로 빨강/초록 콘 검출
- 아두이노 시리얼 프로토콜: <L:±PWM,R:±PWM> 형식
- 좌측 초록 충돌 위험 → 우회전 보정
- 우측 빨강 충돌 위험 → 좌회전 보정
- 게이트 중앙 통과 목표
- 5쌍 게이트 통과
"""

import time
import serial
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from pathlib import Path

# ===========================
# 설정
# ===========================

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200  # 아두이노 코드와 일치
DEFAULT_SPEED = '7'

TOTAL_GATES = 5  # 총 5쌍의 게이트
MODEL_PATH = 'cone.pt'  # YOLO 모델 (red_cone, green_cone 클래스)
CONFIDENCE_THRESHOLD = 0.4

COLOR_W, COLOR_H = 640, 480

# ===========================
# 모터 제어 파라미터
# ===========================

# PWM 범위 (-255 ~ +255)
PWM_MAX = 255
PWM_TURN = 180      # 회전 시 PWM
PWM_FORWARD = 250   # 직진 시 PWM
PWM_SLOW = 200     # 저속 접근 시 PWM

# 타이밍
TURN_TIME = 0.2         # 회전 시간
FORWARD_TIME = 0.25     # 직진 시간
APPROACH_TIME = 0.3     # 접근 시간
PASS_TIME = 2.0         # 게이트 통과 시간
SCAN_TIME = 1.0         # 스캔 회전 시간

# ===========================
# 비전 파라미터
# ===========================

MIN_AREA = 300                      # 최소 콘 면적
DEADZONE = 80                       # 중앙 정렬 데드존 (픽셀)
Y_ALIGNMENT_THRESHOLD = 150         # 수평 정렬 허용 오차 (픽셀)

# ⭐ 충돌 회피 파라미터
COLLISION_DANGER_X = 80             # 위험: 화면 좌우 끝에서 80px 이내
COLLISION_WARNING_X = 150           # 경고: 화면 좌우 끝에서 150px 이내
COLLISION_THRESHOLD_Y = COLOR_H * 0.6  # 화면 하단 60% 이상에서만 충돌 감지
COLLISION_AREA_THRESHOLD = 15000    # 면적이 너무 크면 충돌 위험

# 게이트 통과 판단
GATE_PASS_Y_THRESHOLD = COLOR_H * 0.7  # 화면 하단 70% 이상
GATE_PASS_AREA_THRESHOLD = 12000       # 평균 면적이 이 이상이면 통과

# 스캔 설정
SCAN_INTERVAL = 2.0         # 스캔 주기
GATE_LOST_TIMEOUT = 2.0     # 게이트 미발견 시간

# ===========================
# 아두이노 모터 제어
# ===========================

class ArduinoMotorController:
    """
    아두이노 시리얼 프로토콜 통신
    형식: <L:±PWM,R:±PWM>
    예: <L:+200,R:+200> (직진)
        <L:+150,R:-150> (우회전)
    """
    def __init__(self, port: str = SERIAL_PORT, baudrate: int = BAUD_RATE):
        self.ser = None
        self.use_serial = True
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            self.stop()
            print(f"✅ 아두이노 연결: {port} @ {baudrate}bps")
        except Exception as e:
            print(f"⚠️  아두이노 연결 실패: {e}")
            print("⚠️  시뮬레이션 모드로 실행합니다.")
            self.use_serial = False
    
    def send_command(self, left_pwm: int, right_pwm: int):
        """
        좌우 모터 PWM 전송
        left_pwm, right_pwm: -255 ~ +255
        """
        left_pwm = int(np.clip(left_pwm, -PWM_MAX, PWM_MAX))
        right_pwm = int(np.clip(right_pwm, -PWM_MAX, PWM_MAX))
        
        # 부호 처리
        left_sign = '+' if left_pwm >= 0 else ''
        right_sign = '+' if right_pwm >= 0 else ''
        
        # 프로토콜 생성: <L:+200,R:+200>
        cmd = f"<L:{left_sign}{left_pwm},R:{right_sign}{right_pwm}>\n"
        
        if self.use_serial and self.ser and self.ser.is_open:
            self.ser.write(cmd.encode('utf-8'))
            time.sleep(0.01)
        else:
            print(f"[MOTOR] {cmd.strip()}")
    
    def forward(self, speed: int = PWM_FORWARD):
        """직진"""
        self.send_command(speed, speed)
    
    def backward(self, speed: int = PWM_FORWARD):
        """후진"""
        self.send_command(-speed, -speed)
    
    def turn_left(self, speed: int = PWM_TURN):
        """좌회전 (제자리)"""
        self.send_command(-speed, speed)
    
    def turn_right(self, speed: int = PWM_TURN):
        """우회전 (제자리)"""
        self.send_command(speed, -speed)
    
    def pivot_left(self, speed: int = PWM_TURN):
        """피벗 좌회전 (좌측 정지, 우측만 회전)"""
        self.send_command(0, speed)
    
    def pivot_right(self, speed: int = PWM_TURN):
        """피벗 우회전 (우측 정지, 좌측만 회전)"""
        self.send_command(speed, 0)
    
    def stop(self):
        """정지"""
        self.send_command(0, 0)
    
    def close(self):
        """종료"""
        if self.use_serial and self.ser and self.ser.is_open:
            self.stop()
            time.sleep(0.1)
            self.ser.close()
            print("✅ 아두이노 종료")


# ===========================
# YOLO 검출기
# ===========================

class ConeDetector:
    """
    YOLO 기반 콘 검출
    클래스: 'red_cone', 'green_cone'
    """
    def __init__(self, model_path: str = MODEL_PATH, conf: float = CONFIDENCE_THRESHOLD):
        print(f"📦 YOLO 모델 로딩: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        print("✅ YOLO 로딩 완료")
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        콘 검출
        
        Returns:
            (red_cones, green_cones)
            각 콘: {'x', 'y', 'w', 'h', 'cx', 'cy', 'bottom_y', 'area'}
        """
        results = self.model.predict(frame, conf=self.conf, verbose=False, device='cpu')
        
        red_cones = []
        green_cones = []
        
        for r in results:
            if r.boxes is None:
                continue
            
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                if area < MIN_AREA:
                    continue
                
                cone = {
                    'x': x1,
                    'y': y1,
                    'w': w,
                    'h': h,
                    'cx': x1 + w // 2,
                    'cy': y1 + h // 2,
                    'bottom_y': y2,
                    'area': area,
                    'conf': float(box.conf[0])
                }
                
                if cls_name == 'red_cone':
                    red_cones.append(cone)
                elif cls_name == 'green_cone':
                    green_cones.append(cone)
        
        return red_cones, green_cones


# ===========================
# 게이트 검출 로직
# ===========================

def find_best_gate_pair(red_cones: List[Dict], 
                        green_cones: List[Dict]) -> Optional[Tuple[Dict, Dict]]:
    """
    가장 가까운 유효 게이트 쌍 찾기
    
    조건:
    1. 좌측 = 초록, 우측 = 빨강
    2. 수평 정렬 (Y좌표 차이 < Y_ALIGNMENT_THRESHOLD)
    3. 면적 + 화면 하단 위치 기준으로 가장 가까운 것
    
    Returns:
        (red_cone, green_cone) 또는 None
    """
    if not red_cones or not green_cones:
        return None
    
    valid_pairs = []
    
    for green in green_cones:
        gcx, gcy = green['cx'], green['cy']
        
        for red in red_cones:
            rcx, rcy = red['cx'], red['cy']
            
            # 조건 1: 좌측 = 초록, 우측 = 빨강
            if gcx >= rcx:
                continue
            
            # 조건 2: 수평 정렬
            if abs(gcy - rcy) > Y_ALIGNMENT_THRESHOLD:
                continue
            
            # 거리 점수 계산 (면적 70% + Y좌표 30%)
            avg_area = (green['area'] + red['area']) / 2
            avg_y = (green['bottom_y'] + red['bottom_y']) / 2
            
            # 정규화를 위한 최대값
            max_area = 50000  # 가정
            max_y = COLOR_H
            
            area_score = min(avg_area / max_area, 1.0)
            y_score = avg_y / max_y
            
            # 종합 점수 (가까울수록 높음)
            score = 0.7 * area_score + 0.3 * y_score
            
            valid_pairs.append((red, green, score))
    
    if not valid_pairs:
        return None
    
    # 점수가 가장 높은 쌍 선택
    valid_pairs.sort(key=lambda x: x[2], reverse=True)
    best_red, best_green, _ = valid_pairs[0]
    
    return (best_red, best_green)


def check_collision_risk(cone: Dict, frame_cx: int, side: str) -> Tuple[str, str]:
    """
    충돌 위험 감지
    
    Args:
        cone: 콘 정보
        frame_cx: 화면 중심 X
        side: 'left' (초록) 또는 'right' (빨강)
    
    Returns:
        (위험도, 회피방향)
        위험도: 'none', 'warning', 'danger'
        회피방향: 'none', 'left', 'right'
    """
    cx = cone['cx']
    bottom_y = cone['bottom_y']
    area = cone['area']
    
    # 화면 상단에 있으면 위험 없음
    if bottom_y < COLLISION_THRESHOLD_Y:
        return 'none', 'none'
    
    # 좌측 콘 (초록)
    if side == 'left':
        # 화면 좌측 끝에 너무 가까움
        if cx < COLLISION_DANGER_X:
            return 'danger', 'right'
        elif cx < COLLISION_WARNING_X:
            return 'warning', 'right'
    
    # 우측 콘 (빨강)
    elif side == 'right':
        # 화면 우측 끝에 너무 가까움
        if cx > (COLOR_W - COLLISION_DANGER_X):
            return 'danger', 'left'
        elif cx > (COLOR_W - COLLISION_WARNING_X):
            return 'warning', 'left'
    
    # 면적이 너무 큰 경우 (너무 가까움)
    if area > COLLISION_AREA_THRESHOLD:
        if side == 'left':
            return 'danger', 'right'
        else:
            return 'danger', 'left'
    
    return 'none', 'none'


# ===========================
# ROS2 노드
# ===========================

class YoloGateNavigator(Node):
    def __init__(self):
        super().__init__('yolo_gate_navigator')
        
        # 모터 & 검출기 초기화
        self.motor = ArduinoMotorController()
        self.detector = ConeDetector()
        
        # ROS2 구독
        self.bridge = CvBridge()
        self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )
        
        # 상태
        self.color_img = None
        self.gates_passed = 0
        self.mission_complete = False
        
        # 탐색 관련
        self.last_gate_seen = time.time()
        self.last_scan_time = 0
        self.scan_direction = 'right'
        
        # FPS 추적
        self._t_prev = time.time()
        self._fps_smooth = None
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("🚢 YOLO Gate Navigator 시작")
        self.get_logger().info(f"   총 {TOTAL_GATES}개 게이트 통과 목표")
        self.get_logger().info("=" * 70)
    
    def color_callback(self, msg: Image):
        self.color_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.process_frame()
    
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
        if self.color_img is None or self.mission_complete:
            return
        
        frame = self.color_img.copy()
        frame_cx = COLOR_W // 2
        
        # YOLO 검출
        red_cones, green_cones = self.detector.detect(frame)
        
        # 완료 확인
        if self.gates_passed >= TOTAL_GATES:
            cv2.putText(frame, "🎉 MISSION COMPLETE! 🎉", (100, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            self.motor.stop()
            self.mission_complete = True
            self.get_logger().info("=" * 70)
            self.get_logger().info(f"🎉 미션 완료! {self.gates_passed}/{TOTAL_GATES} 게이트 통과!")
            self.get_logger().info("=" * 70)
            cv2.imshow("YOLO Gate Navigator", frame)
            cv2.waitKey(1)
            return
        
        # 화면 정보
        fps = self._update_fps()
        cv2.putText(frame, f"Gates: {self.gates_passed}/{TOTAL_GATES}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, f"Red: {len(red_cones)} | Green: {len(green_cones)}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"{fps:5.1f} FPS", 
                   (20, COLOR_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        
        # 시각화
        for cone in red_cones:
            x, y, w, h = cone['x'], cone['y'], cone['w'], cone['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f"R:{cone['area']}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        for cone in green_cones:
            x, y, w, h = cone['x'], cone['y'], cone['w'], cone['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"G:{cone['area']}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 중앙선 표시
        cv2.line(frame, (frame_cx, 0), (frame_cx, COLOR_H), (255, 255, 255), 1)
        
        # ⭐ 게이트 찾기
        gate_pair = find_best_gate_pair(red_cones, green_cones)
        
        if gate_pair:
            # 게이트 발견!
            self.last_gate_seen = time.time()
            self.navigate_through_gate(gate_pair, frame, frame_cx)
        else:
            # 게이트 없음 → 탐색
            self.search_gate(red_cones, green_cones, frame, frame_cx)
        
        cv2.imshow("YOLO Gate Navigator", frame)
        cv2.waitKey(1)
    
    def navigate_through_gate(self, gate_pair: Tuple[Dict, Dict], 
                             frame: np.ndarray, frame_cx: int):
        """⭐ 게이트 통과 로직 (충돌 회피 포함)"""
        red_cone, green_cone = gate_pair
        
        rcx, rcy = red_cone['cx'], red_cone['cy']
        gcx, gcy = green_cone['cx'], green_cone['cy']
        
        # 게이트 중심점
        gate_cx = (rcx + gcx) // 2
        gate_cy = (rcy + gcy) // 2
        
        # 시각화
        rx, ry, rw, rh = red_cone['x'], red_cone['y'], red_cone['w'], red_cone['h']
        gx, gy, gw, gh = green_cone['x'], green_cone['y'], green_cone['w'], green_cone['h']
        
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 0, 255), 3)
        cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (255, 0, 255), 3)
        cv2.line(frame, (gate_cx, 0), (gate_cx, COLOR_H), (0, 255, 255), 2)
        cv2.circle(frame, (gate_cx, gate_cy), 15, (0, 255, 255), -1)
        
        cv2.putText(frame, f"GATE #{self.gates_passed+1}", 
                   (gate_cx - 60, gate_cy - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 중앙 오차
        error = gate_cx - frame_cx
        
        # ⭐⭐⭐ 1단계: 충돌 위험 체크
        green_risk, green_avoid = check_collision_risk(green_cone, frame_cx, 'left')
        red_risk, red_avoid = check_collision_risk(red_cone, frame_cx, 'right')
        
        # 위험도 시각화
        if green_risk == 'danger':
            cv2.putText(frame, "⚠️ GREEN DANGER!", (gx, gy-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif green_risk == 'warning':
            cv2.putText(frame, "⚠ GREEN WARNING", (gx, gy-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        
        if red_risk == 'danger':
            cv2.putText(frame, "⚠️ RED DANGER!", (rx, ry-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif red_risk == 'warning':
            cv2.putText(frame, "⚠ RED WARNING", (rx, ry-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        
        # ⭐⭐⭐ 2단계: 충돌 회피 우선 처리
        if green_risk == 'danger':
            # 좌측 초록 위험! → 우회전
            self.get_logger().warn(f"🚨 좌측 초록 충돌 위험! → 우회전 회피")
            cv2.putText(frame, "AVOIDING GREEN → TURN RIGHT", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            self.motor.turn_right(PWM_TURN)
            time.sleep(0.35)
            self.motor.forward(PWM_SLOW)
            time.sleep(0.2)
            self.motor.stop()
            return
        
        if red_risk == 'danger':
            # 우측 빨강 위험! → 좌회전
            self.get_logger().warn(f"🚨 우측 빨강 충돌 위험! → 좌회전 회피")
            cv2.putText(frame, "AVOIDING RED → TURN LEFT", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            self.motor.turn_left(PWM_TURN)
            time.sleep(0.35)
            self.motor.forward(PWM_SLOW)
            time.sleep(0.2)
            self.motor.stop()
            return
        
        # 경고 수준 회피
        if green_risk == 'warning':
            self.get_logger().info(f"⚠️  좌측 초록 근접 → 우측 보정")
            cv2.putText(frame, "Adjust RIGHT (avoid green)", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            
            self.motor.pivot_right(PWM_TURN // 2)
            time.sleep(0.2)
            self.motor.stop()
            return
        
        if red_risk == 'warning':
            self.get_logger().info(f"⚠️  우측 빨강 근접 → 좌측 보정")
            cv2.putText(frame, "Adjust LEFT (avoid red)", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            
            self.motor.pivot_left(PWM_TURN // 2)
            time.sleep(0.2)
            self.motor.stop()
            return
        
        # ⭐⭐⭐ 3단계: 통과 판단
        avg_area = (red_cone['area'] + green_cone['area']) / 2
        is_close_enough = (gate_cy > GATE_PASS_Y_THRESHOLD or 
                          avg_area > GATE_PASS_AREA_THRESHOLD)
        
        if is_close_enough:
            # 게이트 통과!
            self.get_logger().info(f"🚪 게이트 #{self.gates_passed+1} 통과 중...")
            cv2.putText(frame, f"PASSING GATE #{self.gates_passed+1}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 마지막 미세 조정
            if abs(error) > 60:
                if error > 0:
                    self.motor.pivot_right(PWM_TURN // 3)
                    time.sleep(0.15)
                else:
                    self.motor.pivot_left(PWM_TURN // 3)
                    time.sleep(0.15)
            
            # 통과!
            self.motor.forward(PWM_FORWARD)
            time.sleep(PASS_TIME)
            self.motor.stop()
            
            self.gates_passed += 1
            self.get_logger().info(f"✅ 게이트 #{self.gates_passed}/{TOTAL_GATES} 통과 완료!")
            time.sleep(0.3)
        
        # ⭐⭐⭐ 4단계: 접근 & 정렬
        else:
            cv2.putText(frame, f"Error: {error:+d}px", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            if abs(error) <= DEADZONE:
                # 중앙 정렬 완료 → 직진
                self.get_logger().info(f"→ 중앙 정렬 OK → 직진")
                cv2.putText(frame, "ALIGNED - FORWARD", (20, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                self.motor.forward(PWM_FORWARD)
                time.sleep(FORWARD_TIME * 1.5)
                self.motor.stop()
            
            else:
                # 중앙 보정 필요
                self.get_logger().info(f"→ 중앙 보정 (오차: {error:+d}px)")
                
                if error > 0:
                    # 게이트가 오른쪽 → 우회전
                    cv2.putText(frame, "Align RIGHT", (20, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    self.motor.turn_right(PWM_TURN)
                else:
                    # 게이트가 왼쪽 → 좌회전
                    cv2.putText(frame, "Align LEFT", (20, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    self.motor.turn_left(PWM_TURN)
                
                # 오차에 비례한 회전 시간
                turn_duration = min(abs(error) / 200.0, 1.0) * TURN_TIME
                time.sleep(turn_duration)
                
                # 전진
                self.motor.forward(PWM_SLOW)
                time.sleep(FORWARD_TIME)
                self.motor.stop()
    
    def search_gate(self, red_cones: List[Dict], green_cones: List[Dict],
                   frame: np.ndarray, frame_cx: int):
        """⭐ 게이트 탐색 로직"""
        
        current_time = time.time()
        
        # 한쪽만 보이는 경우
        if red_cones and not green_cones:
            cv2.putText(frame, "RED only - Scanning LEFT for GREEN", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if current_time - self.last_scan_time > 1.0:
                self.get_logger().info("🔴 빨강만 보임 → 좌회전으로 초록 찾기")
                self.motor.turn_left(PWM_TURN)
                time.sleep(SCAN_TIME * 0.8)
                self.motor.stop()
                self.last_scan_time = current_time
        
        elif green_cones and not red_cones:
            cv2.putText(frame, "GREEN only - Scanning RIGHT for RED", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if current_time - self.last_scan_time > 1.0:
                self.get_logger().info("🟢 초록만 보임 → 우회전으로 빨강 찾기")
                self.motor.turn_right(PWM_TURN)
                time.sleep(SCAN_TIME * 0.8)
                self.motor.stop()
                self.last_scan_time = current_time
        
        # 아무것도 안 보이는 경우
        else:
            cv2.putText(frame, f"Searching Gate #{self.gates_passed+1}...", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 최근 본 적 있으면 직진
            if current_time - self.last_gate_seen < GATE_LOST_TIMEOUT:
                self.get_logger().info("최근 게이트 봤음 → 직진")
                self.motor.forward(PWM_SLOW)
                time.sleep(0.2)
                self.motor.stop()
            
            # 오래 못 봤으면 좌우 스캔
            elif current_time - self.last_scan_time > SCAN_INTERVAL:
                self.get_logger().info(f"🔍 [{self.scan_direction.upper()}] 스캔")
                
                if self.scan_direction == 'left':
                    self.motor.turn_left(PWM_TURN)
                    time.sleep(SCAN_TIME)
                    self.scan_direction = 'right'
                else:
                    self.motor.turn_right(PWM_TURN)
                    time.sleep(SCAN_TIME)
                    self.scan_direction = 'left'
                
                self.motor.stop()
                self.last_scan_time = current_time
    
    def destroy_node(self):
        self.motor.close()
        cv2.destroyAllWindows()
        super().destroy_node()


# ===========================
# 메인
# ===========================

def main(args=None):
    print("\n" + "=" * 70)
    print("🚢 KABOAT YOLO Gate Navigator")
    print(f"   - 아두이노 시리얼: {SERIAL_PORT} @ {BAUD_RATE}bps")
    print(f"   - YOLO 모델: {MODEL_PATH}")
    print(f"   - 목표: {TOTAL_GATES}개 게이트 통과")
    print("=" * 70 + "\n")
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ YOLO 모델 파일 없음: {MODEL_PATH}")
        return
    
    rclpy.init(args=args)
    node = YoloGateNavigator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n⚠️  사용자 중단")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("✅ 종료 완료")


if __name__ == '__main__':
    main()