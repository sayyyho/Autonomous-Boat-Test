#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KABOAT Phase1: Continuous Rotation Search
- 초록만 보임 → 빨강 잡힐 때까지 계속 우회전
- 빨강만 보임 → 초록 잡힐 때까지 계속 좌회전
- 둘 다 보임 → 즉시 게이트 중앙 직진
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
DEFAULT_SPEED = '5'

TOTAL_GATES = int(input("통과할 게이트 수: ") or "5")
print(f"✅ 총 {TOTAL_GATES}개의 게이트를 통과합니다.")

MODEL_PATH = 'cone.pt'
CONFIDENCE_THRESHOLD = 0.4

COLOR_W, COLOR_H = 640, 480

# 타이밍
FORWARD_TIME = 0.25
PASS_TIME = 2.0

# 파라미터
MIN_AREA = 300
DEADZONE = 100
Y_ALIGNMENT_THRESHOLD = 150

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
# 게이트 검출
# ===========================

def find_best_gate(reds: List[Dict], greens: List[Dict]) -> Optional[Tuple[Dict, Dict]]:
    """가장 가까운 게이트 쌍"""
    if not reds or not greens:
        return None
    
    all_cones = reds + greens
    max_area = max(c['area'] for c in all_cones)
    max_y = max(c['bottom_y'] for c in all_cones)
    
    best_gate = None
    best_score = -1
    
    for green in greens:
        gcx, gcy = green['cx'], green['cy']
        
        for red in reds:
            rcx, rcy = red['cx'], red['cy']
            
            # 조건 1: 좌=초록, 우=빨강
            if gcx >= rcx:
                continue
            
            # 조건 2: Y좌표 수평 정렬
            if abs(gcy - rcy) > Y_ALIGNMENT_THRESHOLD:
                continue
            
            # 깊이 점수
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

class ContinuousSearchNavigator(Node):
    def __init__(self):
        super().__init__('continuous_search_navigator')
        
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
        
        # ⭐ 상태 변수
        self.state = 'SEARCHING'  # 'SEARCHING', 'GATE_MODE'
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🚢 Continuous Search Navigator 시작")
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
        cv2.putText(frame, f"Gates: {self.gates}/{TOTAL_GATES} | State: {self.state}",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, f"Red: {len(reds)} | Green: {len(greens)}",
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 시각화
        for cone in reds:
            x, y, w, h = cone['x'], cone['y'], cone['w'], cone['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(frame, "RED", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        for cone in greens:
            x, y, w, h = cone['x'], cone['y'], cone['w'], cone['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, "GREEN", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # ⭐⭐⭐ 핵심 로직
        has_red = len(reds) > 0
        has_green = len(greens) > 0
        
        # 케이스 1: 둘 다 보임 → 게이트 모드
        if has_red and has_green:
            gate = find_best_gate(reds, greens)
            
            if gate:
                self.state = 'GATE_MODE'
                cv2.putText(frame, "GATE MODE - Both cones visible!", (20, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                self.navigate_through_gate(gate, frame, frame_cx)
            else:
                # 유효한 게이트 아님 → 직진
                cv2.putText(frame, "Cones visible but no valid gate - Forward",
                           (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                self.motor.forward()
                time.sleep(FORWARD_TIME)
                self.motor.stop()
        
        # ⭐ 케이스 2: 초록만 보임 → 빨강 찾을 때까지 계속 우회전
        elif has_green and not has_red:
            self.state = 'SEARCHING'
            cv2.putText(frame, "GREEN ONLY - Turning RIGHT to find RED...", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
            cv2.putText(frame, ">>> CONTINUOUS RIGHT TURN >>>", (20, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            self.get_logger().info("🟢 초록만 보임 → 우회전 계속...")
            
            # ⭐ 계속 우회전 (stop 없음)
            self.motor.right()
            # time.sleep 없음 - 다음 프레임에서 계속 체크
        
        # ⭐ 케이스 3: 빨강만 보임 → 초록 찾을 때까지 계속 좌회전
        elif has_red and not has_green:
            self.state = 'SEARCHING'
            cv2.putText(frame, "RED ONLY - Turning LEFT to find GREEN...", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.putText(frame, "<<< CONTINUOUS LEFT TURN <<<", (20, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            self.get_logger().info("🔴 빨강만 보임 → 좌회전 계속...")
            
            # ⭐ 계속 좌회전 (stop 없음)
            self.motor.left()
            # time.sleep 없음 - 다음 프레임에서 계속 체크
        
        # 케이스 4: 아무것도 안 보임 → 천천히 직진
        else:
            self.state = 'SEARCHING'
            cv2.putText(frame, "No cones - Slow forward", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            self.get_logger().info("⚠️  콘 없음 → 천천히 직진")
            
            self.motor.forward()
            time.sleep(0.15)
            self.motor.stop()
        
        cv2.imshow("Navigator", frame)
        cv2.waitKey(1)
    
    def navigate_through_gate(self, gate: Tuple[Dict, Dict], frame: np.ndarray, frame_cx: int):
        """게이트 통과"""
        red, green = gate
        
        rcx, rcy = red['cx'], red['cy']
        gcx, gcy = green['cx'], green['cy']
        
        # 게이트 중점
        gate_cx = (rcx + gcx) // 2
        gate_cy = (rcy + gcy) // 2
        
        # 시각화
        rx, ry, rw, rh = red['x'], red['y'], red['w'], red['h']
        gx, gy, gw, gh = green['x'], green['y'], green['w'], green['h']
        
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 0, 255), 4)
        cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (255, 0, 255), 4)
        cv2.line(frame, (gate_cx, 0), (gate_cx, COLOR_H), (0, 255, 255), 3)
        cv2.circle(frame, (gate_cx, gate_cy), 25, (0, 255, 255), -1)
        
        error = gate_cx - frame_cx
        
        # 통과 판단
        avg_area = (red['area'] + green['area']) / 2
        
        if gate_cy > COLOR_H * 0.65 or avg_area > 12000:
            self.get_logger().info(f"🚪 게이트 #{self.gates+1} 통과!")
            
            cv2.putText(frame, f"PASSING GATE #{self.gates+1}", (20, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
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
        
        # 접근 중
        else:
            cv2.putText(frame, f"Approaching | Error: {error}px", (20, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if abs(error) <= DEADZONE:
                # 중앙 정렬 → 직진
                self.get_logger().info(f"→ 중앙 정렬 → 직진")
                self.motor.forward()
                time.sleep(FORWARD_TIME * 2)
                self.motor.stop()
            else:
                # 보정
                self.get_logger().info(f"→ 보정 (오차: {error}px)")
                
                if error > 0:
                    self.motor.right()
                else:
                    self.motor.left()
                
                time.sleep(0.15 * min(abs(error)/100, 1.5))
                
                self.motor.forward()
                time.sleep(FORWARD_TIME)
                self.motor.stop()


def main(args=None):
    print("\n" + "=" * 60)
    print("🚢 KABOAT Continuous Search Navigator")
    print("=" * 60)
    print("📋 로직:")
    print("  1. 초록만 → 빨강 잡힐 때까지 계속 우회전")
    print("  2. 빨강만 → 초록 잡힐 때까지 계속 좌회전")
    print("  3. 둘 다 → 즉시 게이트 중앙 직진")
    print("=" * 60)
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return
    
    rclpy.init(args=args)
    node = ContinuousSearchNavigator()
    
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