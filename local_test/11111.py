#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KABOAT Phase1: YOLO + ROS2 + Simple Motor Control
- Document 5의 동작하는 모터 제어 방식 사용
- YOLO 게이트 검출
- 깊이 기반 거리 추정
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
from ultralytics import YOLO
from pathlib import Path

# ===========================
# 설정 파라미터
# ===========================

# 하드웨어
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
DEFAULT_SPEED = '9'

# 미션
TOTAL_GATES = int(input("통과할 게이트 수: ") or "5")
print(f"✅ 총 {TOTAL_GATES}개의 게이트를 통과합니다.")

# YOLO
MODEL_PATH = 'cone.pt'
CONFIDENCE_THRESHOLD = 0.5

# 카메라
COLOR_W, COLOR_H = 640, 480

# 게이트 검출
Y_ALIGNMENT_THRESHOLD = 100
MIN_CONE_AREA = 400
GATE_CENTER_DEADZONE = 50

# 깊이 가중치
AREA_WEIGHT = 0.6
Y_WEIGHT = 0.4

# ⭐ 타이밍 (실제 동작하는 방식)
FORWARD_TIME = 0.3
TURN_TIME = 0.4
SCAN_TURN_TIME = 1.2
APPROACH_TIME = 0.6
GATE_PASS_TIME = 2.0

# ===========================
# ⭐ Document 5 스타일 모터 제어
# ===========================

class ArduinoMotorController:
    """단순하고 확실한 모터 제어"""
    
    def __init__(self, port: str = SERIAL_PORT, baudrate: int = BAUD_RATE):
        self.ser = None
        self.current_command = b'x'
        
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            self.set_speed(DEFAULT_SPEED)
            self.stop()
            print(f"✅ 아두이노 연결: {port}")
        except Exception as e:
            print(f"❌ 연결 실패: {e}")
            self.ser = None
    
    def send_command(self, command: bytes):
        """명령 즉시 전송"""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(command)
                self.current_command = command
                time.sleep(0.01)
            except Exception as e:
                print(f"전송 실패: {e}")
        else:
            print(f"[MOTOR] {command.decode('utf-8', errors='ignore')}")
    
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
            time.sleep(0.1)
            self.ser.close()
            print("✅ 아두이노 종료")


# 전역 모터 컨트롤러
motor_controller = None

def init_motor():
    global motor_controller
    motor_controller = ArduinoMotorController()
    return motor_controller

def motor_action(action: str):
    """간단한 모터 제어"""
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


# ===========================
# YOLO 검출기
# ===========================

class YOLOConeDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ 모델 없음: {model_path}")
        
        print(f"📦 YOLO 로딩: {model_path}")
        self.model = YOLO(str(model_path))
        self.device = 'cpu'
        print(f"✅ 로드 완료")
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False
        )
        
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
                bottom_y = y2
                
                cone_data = {
                    'bbox': (x1, y1, w, h),
                    'conf': confidence,
                    'center': (cx, cy),
                    'area': area,
                    'bottom_y': bottom_y
                }
                
                if cls_name == 'red_cone':
                    red_cones.append(cone_data)
                elif cls_name == 'green_cone':
                    green_cones.append(cone_data)
        
        return red_cones, green_cones


# ===========================
# 게이트 검출
# ===========================

def calculate_depth_score(cone: Dict, max_area: float, max_y: float) -> float:
    area_score = cone['area'] / max_area if max_area > 0 else 0
    y_score = cone['bottom_y'] / max_y if max_y > 0 else 0
    return AREA_WEIGHT * area_score + Y_WEIGHT * y_score


def find_nearest_gate_pair(red_cones: List[Dict], 
                           green_cones: List[Dict],
                           frame_width: int,
                           frame_height: int) -> Optional[Tuple[Dict, Dict, float]]:
    if not red_cones or not green_cones:
        return None
    
    all_cones = red_cones + green_cones
    max_area = max(c['area'] for c in all_cones)
    max_y = max(c['bottom_y'] for c in all_cones)
    
    best_gate = None
    best_depth = -1
    
    for green in green_cones:
        green_cx, green_cy = green['center']
        
        for red in red_cones:
            red_cx, red_cy = red['center']
            
            if green_cx >= red_cx:
                continue
            
            y_diff = abs(green_cy - red_cy)
            if y_diff > Y_ALIGNMENT_THRESHOLD:
                continue
            
            green_depth = calculate_depth_score(green, max_area, max_y)
            red_depth = calculate_depth_score(red, max_area, max_y)
            avg_depth = (green_depth + red_depth) / 2.0
            
            gate_cx = (green_cx + red_cx) // 2
            center_distance = abs(gate_cx - frame_width // 2)
            center_bonus = 1.0 - (center_distance / frame_width) * 0.2
            
            final_score = avg_depth * center_bonus
            
            if final_score > best_depth:
                best_depth = final_score
                best_gate = (red, green, final_score)
    
    return best_gate


# ===========================
# ROS2 노드
# ===========================

class SimpleGateNavigatorNode(Node):
    """단순하고 확실한 게이트 네비게이터"""
    
    def __init__(self):
        super().__init__('simple_gate_navigator')
        
        # 모터 초기화
        self.motor = init_motor()
        
        # YOLO 초기화
        self.get_logger().info(f"YOLO 로딩: {MODEL_PATH}")
        self.detector = YOLOConeDetector(MODEL_PATH, CONFIDENCE_THRESHOLD)
        
        # ROS2 구독
        self.bridge = CvBridge()
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )
        
        # 상태 변수
        self.color_img = None
        self.mission_stage = 'NAVIGATION'
        self.gates_passed = 0
        self.gate_state = 'SEARCHING'
        
        self.last_gate_seen = time.time()
        self.scan_direction = 'right'
        self.last_scan_time = 0
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🚢 Simple Gate Navigator 시작")
        self.get_logger().info("=" * 60)
    
    def color_callback(self, msg: Image):
        """컬러 이미지 수신"""
        self.color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_frame()
    
    def process_frame(self):
        """프레임 처리"""
        if self.color_img is None:
            return
        
        frame = self.color_img.copy()
        
        # YOLO 검출
        red_cones, green_cones = self.detector.detect(frame)
        
        # 게이트 찾기
        gate_info = find_nearest_gate_pair(
            red_cones, green_cones,
            frame.shape[1], frame.shape[0]
        )
        
        # 시각화
        self.visualize(frame, red_cones, green_cones, gate_info)
        
        # 항법 로직
        if self.mission_stage == 'NAVIGATION':
            self.navigation_logic(gate_info, frame)
        elif self.mission_stage == 'COMPLETE':
            self.complete_logic(frame)
        
        cv2.imshow("Simple Navigator", frame)
        cv2.waitKey(1)
    
    def visualize(self, frame, red_cones, green_cones, gate_info):
        """시각화"""
        
        selected_red = gate_info[0] if gate_info else None
        selected_green = gate_info[1] if gate_info else None
        
        # 초록 콘
        for cone in green_cones:
            x, y, w, h = cone['bbox']
            is_selected = (selected_green and cone == selected_green)
            color = (0, 255, 255) if is_selected else (0, 255, 0)
            thickness = 4 if is_selected else 2
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            cv2.putText(frame, f"G {cone['conf']:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 빨강 콘
        for cone in red_cones:
            x, y, w, h = cone['bbox']
            is_selected = (selected_red and cone == selected_red)
            color = (0, 255, 255) if is_selected else (0, 0, 255)
            thickness = 4 if is_selected else 2
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            cv2.putText(frame, f"R {cone['conf']:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 게이트 강조
        if gate_info:
            red, green, depth_score = gate_info
            red_cx, red_cy = red['center']
            green_cx, green_cy = green['center']
            
            gate_cx = (red_cx + green_cx) // 2
            gate_cy = (red_cy + green_cy) // 2
            
            cv2.line(frame, (gate_cx, 0), (gate_cx, frame.shape[0]),
                    (0, 255, 255), 3)
            cv2.line(frame, (green_cx, green_cy), (red_cx, red_cy),
                    (255, 0, 255), 3)
            cv2.circle(frame, (gate_cx, gate_cy), 12, (0, 255, 255), -1)
            
            label = f"GATE #{self.gates_passed+1} | D:{depth_score:.2f}"
            cv2.putText(frame, label, (gate_cx-70, gate_cy-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 정보 표시
        cv2.putText(frame, 
                   f"Stage: {self.mission_stage} | Gates: {self.gates_passed}/{TOTAL_GATES}",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"State: {self.gate_state}",
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def navigation_logic(self, gate_info, frame):
        """항법 로직 (Document 5 스타일)"""
        
        if self.gates_passed >= TOTAL_GATES:
            self.mission_stage = 'COMPLETE'
            return
        
        if gate_info:
            self.last_gate_seen = time.time()
            self.approach_and_pass_gate(gate_info, frame)
        else:
            self.search_gate()
    
    def approach_and_pass_gate(self, gate_info, frame):
        """⭐ 게이트 접근 및 통과 (Document 5 방식)"""
        
        red, green, depth_score = gate_info
        red_cx, red_cy = red['center']
        green_cx, green_cy = green['center']
        
        gate_cx = (red_cx + green_cx) // 2
        gate_cy = (red_cy + green_cy) // 2
        
        frame_cx = frame.shape[1] // 2
        error = gate_cx - frame_cx
        
        # 통과 단계
        if depth_score > 0.6 or gate_cy > frame.shape[0] * 0.65:
            if self.gate_state != 'PASSING':
                self.gate_state = 'PASSING'
                self.get_logger().info(f"🚪 게이트 #{self.gates_passed+1} 통과")
            
            # 최종 조정
            if abs(error) > GATE_CENTER_DEADZONE // 2:
                if error > 0:
                    motor_action('right')
                    time.sleep(TURN_TIME * 0.3)
                else:
                    motor_action('left')
                    time.sleep(TURN_TIME * 0.3)
            
            # 직진 통과
            motor_action('forward')
            time.sleep(GATE_PASS_TIME)
            motor_action('stop')
            
            self.gates_passed += 1
            self.get_logger().info(f"✅ 게이트 {self.gates_passed}/{TOTAL_GATES} 통과!")
            
            self.gate_state = 'SEARCHING'
            time.sleep(0.5)
        
        # 접근 단계
        else:
            self.gate_state = 'APPROACHING'
            
            if abs(error) <= GATE_CENTER_DEADZONE:
                self.get_logger().info("→ 중앙 정렬 → 전진")
                motor_action('forward')
                time.sleep(APPROACH_TIME)
            elif error > 0:
                self.get_logger().info(f"→ 우측 {error}px")
                motor_action('right')
                time.sleep(TURN_TIME * min(abs(error)/100, 1.0))
                motor_action('forward')
                time.sleep(APPROACH_TIME * 0.5)
            else:
                self.get_logger().info(f"→ 좌측 {abs(error)}px")
                motor_action('left')
                time.sleep(TURN_TIME * min(abs(error)/100, 1.0))
                motor_action('forward')
                time.sleep(APPROACH_TIME * 0.5)
            
            motor_action('stop')
    
    def search_gate(self):
        """게이트 탐색"""
        self.gate_state = 'SEARCHING'
        
        # 최근에 봤으면 직진
        if time.time() - self.last_gate_seen < 2.0:
            motor_action('forward')
            time.sleep(FORWARD_TIME)
            motor_action('stop')
            return
        
        # 스캔
        if time.time() - self.last_scan_time >= 2.0:
            self.scan_for_gate()
        else:
            motor_action('stop')
    
    def scan_for_gate(self):
        """좌우 스캔"""
        self.last_scan_time = time.time()
        self.get_logger().info(f"🔍 [{self.scan_direction}] 스캔")
        
        if self.scan_direction == 'left':
            motor_action('left')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'right'
        else:
            motor_action('right')
            time.sleep(SCAN_TURN_TIME)
            self.scan_direction = 'left'
        
        motor_action('stop')
    
    def complete_logic(self, frame):
        """완료"""
        cv2.putText(frame, "MISSION COMPLETE!",
                   (150, 240), cv2.FONT_HERSHEY_SIMPLEX,
                   1.5, (0, 255, 0), 3)
        motor_action('stop')
        self.get_logger().info("🎉 완료!")
    
    def cleanup(self):
        """정리"""
        global motor_controller
        if motor_controller:
            motor_controller.close()
        cv2.destroyAllWindows()


# ===========================
# 메인
# ===========================

def main(args=None):
    print("\n" + "=" * 60)
    print("🚢 KABOAT Simple Navigator")
    print("=" * 60)
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return
    
    rclpy.init(args=args)
    
    node = None
    try:
        node = SimpleGateNavigatorNode()
        rclpy.spin(node)
    
    except KeyboardInterrupt:
        print("\n⚠️  중단")
    
    finally:
        if node:
            node.cleanup()
        
        if rclpy.ok():
            rclpy.shutdown()
        
        print("=" * 60)
        print("✅ 종료")
        print("=" * 60)


if __name__ == '__main__':
    main()