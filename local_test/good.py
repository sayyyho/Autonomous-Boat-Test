#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KABOAT Phase1: Complete Safe Navigation with YOLO
- 좌우 스캔으로 게이트 쌍 찾기
- 두 콘의 중앙으로 직진
- 좌측 초록 부딪힐 것 같으면 우회전
- 우측 빨강 부딪힐 것 같으면 좌회전
- S자 코스 대응
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
BAUD_RATE = 9600
DEFAULT_SPEED = '7'

TOTAL_GATES = int(input("통과할 게이트 수: ") or "5")
print(f"✅ 총 {TOTAL_GATES}개의 게이트를 통과합니다.")

MODEL_PATH = 'cone.pt'
CONFIDENCE_THRESHOLD = 0.5

COLOR_W, COLOR_H = 640, 480

# 타이밍
FORWARD_TIME = 0.2
TURN_TIME = 0.15
SCAN_TIME = 1.2  # 스캔 회전 시간
PASS_TIME = 1.5

# 파라미터
MIN_AREA = 300
DEADZONE = 100
Y_ALIGNMENT_THRESHOLD = 150  # 수평 정렬 허용 오차

# ⭐ 충돌 방지 파라미터
COLLISION_THRESHOLD_Y = COLOR_H * 0.75  # 화면 하단 75%
COLLISION_THRESHOLD_AREA = 15000  # 면적 임계값
COLLISION_SIDE_THRESHOLD = COLOR_W * 0.35  # 좌우 구분 기준 (35%)

# 스캔 설정
SCAN_INTERVAL = 2.0  # 스캔 주기
GATE_LOST_TIMEOUT = 3.0  # 게이트 미발견 시간

# ===========================
# 모터
# ===========================

class Motor:
    def __init__(self):
        self.ser = None
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)
            self.ser.write(DEFAULT_SPEED.encode())
            time.sleep(0.01)
            self.ser.write(b'x')
            print("✅ 모터 연결")
        except:
            print("❌ 모터 연결 실패")
    
    def cmd(self, c: bytes):
        if self.ser and self.ser.is_open:
            self.ser.write(c)
            time.sleep(0.01)
    
    def forward(self):
        self.cmd(b'w')
    
    def left(self):
        self.cmd(b'a')
    
    def right(self):
        self.cmd(b'd')
    
    def stop(self):
        self.cmd(b'x')
    
    def close(self):
        if self.ser:
            self.stop()
            self.ser.close()


# ===========================
# YOLO
# ===========================

class Detector:
    def __init__(self):
        print("📦 YOLO 로딩...")
        self.model = YOLO(MODEL_PATH)
        self.conf = CONFIDENCE_THRESHOLD
        print("✅ 완료")
    
    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf, verbose=False, device='cpu')
        
        reds = []
        greens = []
        
        for r in results:
            for box in r.boxes:
                cls = r.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                
                if w * h < MIN_AREA:
                    continue
                
                cone = {
                    'x': x1,
                    'y': y1,
                    'w': w,
                    'h': h,
                    'cx': x1 + w//2,
                    'cy': y1 + h//2,
                    'bottom_y': y2,
                    'area': w * h
                }
                
                if cls == 'red_cone':
                    reds.append(cone)
                elif cls == 'green_cone':
                    greens.append(cone)
        
        return reds, greens


# ===========================
# 충돌 감지
# ===========================

def check_collision_risk(cone: Dict, frame_cx: int) -> Tuple[bool, str]:
    """
    충돌 위험 감지
    
    Returns:
        (위험여부, 회피방향)
        회피방향: 'none', 'left', 'right'
    """
    # 화면 하단에 너무 가까움
    if cone['bottom_y'] > COLLISION_THRESHOLD_Y:
        if cone['cx'] < frame_cx:
            return True, 'right'  # 좌측 콘 → 우회전
        else:
            return True, 'left'   # 우측 콘 → 좌회전
    
    # 면적이 너무 큼
    if cone['area'] > COLLISION_THRESHOLD_AREA:
        if cone['cx'] < frame_cx:
            return True, 'right'
        else:
            return True, 'left'
    
    # 중앙에 너무 가까이 있고 크기가 큼
    distance_from_center = abs(cone['cx'] - frame_cx)
    if distance_from_center < 80 and cone['area'] > 8000:
        if cone['cx'] < frame_cx:
            return True, 'right'
        else:
            return True, 'left'
    
    return False, 'none'


# ===========================
# 게이트 검출
# ===========================

def find_best_gate(reds: List[Dict], greens: List[Dict]) -> Optional[Tuple[Dict, Dict]]:
    """
    가장 가까운 유효 게이트 쌍
    조건: 좌=초록, 우=빨강, 수평 정렬
    """
    if not reds or not greens:
        return None
    
    # 면적+Y좌표 기준으로 가장 가까운 것 선택
    all_cones = reds + greens
    max_area = max(c['area'] for c in all_cones)
    max_y = max(c['bottom_y'] for c in all_cones)
    
    best_gate = None
    best_score = -1
    
    for green in greens:
        gcx, gcy = green['center'] if 'center' in green else (green['cx'], green['cy'])
        
        for red in reds:
            rcx, rcy = red['center'] if 'center' in red else (red['cx'], red['cy'])
            
            # 조건 1: 좌=초록, 우=빨강
            if gcx >= rcx:
                continue
            
            # 조건 2: Y좌표 수평 정렬
            if abs(gcy - rcy) > Y_ALIGNMENT_THRESHOLD:
                continue
            
            # 깊이 점수 (면적 70% + Y좌표 30%)
            g_score = 0.7 * green['area'] / max_area + 0.3 * green['bottom_y'] / max_y
            r_score = 0.7 * red['area'] / max_area + 0.3 * red['bottom_y'] / max_y
            score = (g_score + r_score) / 2.0
            
            if score > best_score:
                best_score = score
                best_gate = (red, green)
    
    return best_gate


# ===========================
# ROS2 노드
# ===========================

class CompleteSafeNavigator(Node):
    def __init__(self):
        super().__init__('complete_safe_navigator')
        
        self.motor = Motor()
        self.detector = Detector()
        
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.callback,
            10
        )
        
        self.img = None
        self.gates = 0
        self.done = False
        
        # 스캔 관련
        self.last_gate_seen = time.time()
        self.last_scan_time = 0
        self.scan_direction = 'right'
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🚢 Complete Safe Navigator 시작")
        self.get_logger().info("=" * 60)
    
    def callback(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.process()
    
    def process(self):
        if self.img is None or self.done:
            return
        
        frame = self.img.copy()
        frame_cx = COLOR_W // 2
        
        # YOLO 검출
        reds, greens = self.detector.detect(frame)
        
        # 완료 확인
        if self.gates >= TOTAL_GATES:
            cv2.putText(frame, "MISSION COMPLETE!", (200, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            self.motor.stop()
            self.done = True
            cv2.imshow("Navigator", frame)
            cv2.waitKey(1)
            return
        
        # 정보 표시
        cv2.putText(frame, f"Gates: {self.gates}/{TOTAL_GATES}",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, f"Red: {len(reds)} | Green: {len(greens)}",
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 시각화
        for cone in reds:
            x, y, w, h = cone['x'], cone['y'], cone['w'], cone['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f"R {cone['area']}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        for cone in greens:
            x, y, w, h = cone['x'], cone['y'], cone['w'], cone['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"G {cone['area']}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # ⭐⭐⭐ 게이트 찾기
        gate = find_best_gate(reds, greens)
        
        if gate:
            # 게이트 발견!
            self.last_gate_seen = time.time()
            self.navigate_through_gate(gate, frame, frame_cx)
        else:
            # 게이트 없음 → 스캔 또는 탐색
            self.search_gate(reds, greens, frame, frame_cx)
        
        cv2.imshow("Navigator", frame)
        cv2.waitKey(1)
    
    def navigate_through_gate(self, gate: Tuple[Dict, Dict], frame: np.ndarray, frame_cx: int):
        """⭐ 게이트 항법 (충돌 회피 포함)"""
        red, green = gate
        
        rcx, rcy = red['cx'], red['cy']
        gcx, gcy = green['cx'], green['cy']
        
        # 게이트 중점
        gate_cx = (rcx + gcx) // 2
        gate_cy = (rcy + gcy) // 2
        
        # 시각화
        rx, ry, rw, rh = red['x'], red['y'], red['w'], red['h']
        gx, gy, gw, gh = green['x'], green['y'], green['w'], green['h']
        
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 0, 255), 3)
        cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (255, 0, 255), 3)
        cv2.line(frame, (gate_cx, 0), (gate_cx, COLOR_H), (0, 255, 255), 3)
        cv2.circle(frame, (gate_cx, gate_cy), 25, (0, 255, 255), -1)
        
        error = gate_cx - frame_cx
        
        # ⭐⭐⭐ 충돌 위험 체크
        green_collision, green_avoid = check_collision_risk(green, frame_cx)
        red_collision, red_avoid = check_collision_risk(red, frame_cx)
        
        # 경고 표시
        if green_collision:
            cv2.putText(frame, "GREEN TOO CLOSE!", (gx, gy-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if red_collision:
            cv2.putText(frame, "RED TOO CLOSE!", (rx, ry-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # ⭐⭐⭐ 충돌 회피 우선
        if green_collision:
            # 좌측 초록 부딪힐 것 같음 → 우회전
            self.get_logger().warn(f"⚠️  좌측 초록 충돌 위험! → 우회전")
            cv2.putText(frame, "AVOIDING GREEN - TURN RIGHT", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            self.motor.right()
            time.sleep(0.3)
            self.motor.forward()
            time.sleep(0.15)
            self.motor.stop()
            return
        
        if red_collision:
            # 우측 빨강 부딪힐 것 같음 → 좌회전
            self.get_logger().warn(f"⚠️  우측 빨강 충돌 위험! → 좌회전")
            cv2.putText(frame, "AVOIDING RED - TURN LEFT", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            self.motor.left()
            time.sleep(0.3)
            self.motor.forward()
            time.sleep(0.15)
            self.motor.stop()
            return
        
        # ⭐ 통과 판단
        avg_area = (red['area'] + green['area']) / 2
        
        if gate_cy > COLOR_H * 0.65 or avg_area > 12000:
            self.get_logger().info(f"🚪 게이트 #{self.gates+1} 통과!")
            
            cv2.putText(frame, f"PASSING GATE #{self.gates+1}", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 마지막 보정
            if abs(error) > 50:
                if error > 0:
                    self.motor.right()
                    time.sleep(0.1)
                else:
                    self.motor.left()
                    time.sleep(0.1)
            
            # 통과
            self.motor.forward()
            time.sleep(PASS_TIME)
            self.motor.stop()
            
            self.gates += 1
            self.get_logger().info(f"✅ {self.gates}/{TOTAL_GATES} 완료!")
            time.sleep(0.5)
        
        # ⭐ 정상 접근
        else:
            cv2.putText(frame, f"Error: {error}px", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            if abs(error) <= DEADZONE:
                # 중앙 정렬 → 직진
                self.get_logger().info(f"→ 중앙 정렬 → 직진")
                cv2.putText(frame, "ALIGNED - FORWARD", (20, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                self.motor.forward()
                time.sleep(FORWARD_TIME * 2)
            else:
                # 보정
                self.get_logger().info(f"→ 보정 (오차: {error}px)")
                cv2.putText(frame, "ADJUSTING", (20, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if error > 0:
                    self.motor.right()
                else:
                    self.motor.left()
                
                time.sleep(TURN_TIME * min(abs(error)/100, 1.5))
                
                self.motor.forward()
                time.sleep(FORWARD_TIME)
            
            self.motor.stop()
    
    def search_gate(self, reds: List[Dict], greens: List[Dict], 
                   frame: np.ndarray, frame_cx: int):
        """⭐ 게이트 탐색"""
        
        current_time = time.time()
        
        # 한쪽만 보이는 경우
        if reds and not greens:
            cv2.putText(frame, "Found RED only - Scanning for GREEN", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 빨강만 보임 → 좌측(초록) 찾기
            if current_time - self.last_scan_time > 1.0:
                self.get_logger().info("🔴 빨강만 보임 → 좌회전으로 초록 찾기")
                self.motor.left()
                time.sleep(SCAN_TIME * 0.7)
                self.motor.stop()
                self.last_scan_time = current_time
            
        elif greens and not reds:
            cv2.putText(frame, "Found GREEN only - Scanning for RED", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 초록만 보임 → 우측(빨강) 찾기
            if current_time - self.last_scan_time > 1.0:
                self.get_logger().info("🟢 초록만 보임 → 우회전으로 빨강 찾기")
                self.motor.right()
                time.sleep(SCAN_TIME * 0.7)
                self.motor.stop()
                self.last_scan_time = current_time
        
        # 아무것도 안 보이는 경우
        else:
            cv2.putText(frame, f"Searching Gate #{self.gates+1}...", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 최근 본 적 있으면 직진
            if current_time - self.last_gate_seen < 2.0:
                self.get_logger().info("최근 게이트 봤음 → 직진")
                self.motor.forward()
                time.sleep(0.15)
                self.motor.stop()
            
            # 오래 못 봤으면 좌우 스캔
            elif current_time - self.last_scan_time > SCAN_INTERVAL:
                self.get_logger().info(f"🔍 [{self.scan_direction}] 스캔")
                
                if self.scan_direction == 'left':
                    self.motor.left()
                    time.sleep(SCAN_TIME)
                    self.scan_direction = 'right'
                else:
                    self.motor.right()
                    time.sleep(SCAN_TIME)
                    self.scan_direction = 'left'
                
                self.motor.stop()
                self.last_scan_time = current_time


def main(args=None):
    print("\n" + "=" * 60)
    print("🚢 KABOAT Complete Safe Navigator")
    print("=" * 60)
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return
    
    rclpy.init(args=args)
    node = CompleteSafeNavigator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.motor.close()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()