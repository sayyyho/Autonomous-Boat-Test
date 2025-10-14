#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import serial
import sys, termios, tty, select
import time
import numpy as np
import cv2
from collections import deque
import threading

class ConeDetector:
    """
    LiDAR로 꼬깔(삼각뿔) 형태 감지
    """
    def __init__(self, logger):
        self.logger = logger
        self.min_cone_points = 5  # 최소 포인트 수
        self.max_cone_width = 0.5  # 최대 폭 (미터)
        self.angle_tolerance = 15  # 각도 허용 범위
        
    def detect_cones(self, ranges, angle_min, angle_increment):
        """
        LiDAR 스캔에서 꼬깔 형태 객체 감지
        
        Returns:
            List[Dict]: [{'angle': -20, 'distance': 5.2, 'width': 0.3, 'is_cone': True}, ...]
        """
        ranges = np.array(ranges)
        valid_mask = ~(np.isinf(ranges) | np.isnan(ranges) | (ranges <= 0.1))
        
        if not np.any(valid_mask):
            return []
        
        # 클러스터링
        clusters = self._cluster_points(ranges, valid_mask, angle_min, angle_increment)
        
        # 각 클러스터가 꼬깔인지 판단
        cones = []
        for cluster in clusters:
            if self._is_cone_shaped(cluster):
                cone_info = self._compute_cone_center(cluster)
                cones.append(cone_info)
        
        return cones
    
    def _cluster_points(self, ranges, valid_mask, angle_min, angle_increment):
        """
        거리 기반 클러스터링
        """
        clusters = []
        current_cluster = []
        
        indices = np.where(valid_mask)[0]
        
        for i, idx in enumerate(indices):
            distance = ranges[idx]
            angle = angle_min + idx * angle_increment
            
            point = {
                'index': idx,
                'distance': distance,
                'angle': np.degrees(angle)
            }
            
            if not current_cluster:
                current_cluster.append(point)
            else:
                # 이전 점과의 각도/거리 차이 확인
                prev = current_cluster[-1]
                angle_diff = abs(point['angle'] - prev['angle'])
                dist_diff = abs(point['distance'] - prev['distance'])
                
                # 같은 클러스터 조건: 각도 5도 이내, 거리 0.3m 이내
                if angle_diff < 5 and dist_diff < 0.3:
                    current_cluster.append(point)
                else:
                    # 클러스터 완성
                    if len(current_cluster) >= self.min_cone_points:
                        clusters.append(current_cluster)
                    current_cluster = [point]
        
        # 마지막 클러스터
        if len(current_cluster) >= self.min_cone_points:
            clusters.append(current_cluster)
        
        return clusters
    
    def _is_cone_shaped(self, cluster):
        """
        클러스터가 꼬깔(원뿔) 형태인지 판단
        
        원뿔 특징:
        - 중앙이 가장 가까움 (또는 끝이 가장 가까움)
        - 폭이 0.3~0.5m 정도
        - 점들이 연속적
        """
        if len(cluster) < self.min_cone_points:
            return False
        
        distances = np.array([p['distance'] for p in cluster])
        angles = np.array([p['angle'] for p in cluster])
        
        # 1. 거리 변화 패턴 확인 (V자 또는 역V자)
        min_idx = np.argmin(distances)
        is_v_shape = (min_idx > 0 and min_idx < len(distances) - 1)
        
        # 2. 각도 범위 (너무 넓지 않아야 함)
        angle_span = abs(angles[-1] - angles[0])
        if angle_span > self.angle_tolerance:
            return False
        
        # 3. 폭 계산 (양 끝점의 실제 거리)
        if len(cluster) >= 2:
            left = cluster[0]
            right = cluster[-1]
            
            # 극좌표 → 직교좌표
            left_x = left['distance'] * np.sin(np.radians(left['angle']))
            left_y = left['distance'] * np.cos(np.radians(left['angle']))
            right_x = right['distance'] * np.sin(np.radians(right['angle']))
            right_y = right['distance'] * np.cos(np.radians(right['angle']))
            
            width = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
            
            # 폭이 적절한 범위인지
            if width > self.max_cone_width:
                return False
        
        return True
    
    def _compute_cone_center(self, cluster):
        """
        꼬깔의 중심 각도/거리 계산
        """
        angles = np.array([p['angle'] for p in cluster])
        distances = np.array([p['distance'] for p in cluster])
        
        # 중심 각도 (평균)
        center_angle = np.mean(angles)
        
        # 중심 거리 (최소값 가중)
        center_distance = np.min(distances) * 0.6 + np.mean(distances) * 0.4
        
        # 폭 계산
        left = cluster[0]
        right = cluster[-1]
        left_x = left['distance'] * np.sin(np.radians(left['angle']))
        right_x = right['distance'] * np.sin(np.radians(right['angle']))
        width = abs(right_x - left_x)
        
        return {
            'angle': center_angle,
            'distance': center_distance,
            'width': width,
            'is_cone': True,
            'point_count': len(cluster)
        }


class ColorRegionClassifier:
    """
    색 공간 이분법 분류기
    HSV Hue 중간값(90도)을 기준으로 RED/GREEN 영역 판단
    """
    def __init__(self, logger):
        self.logger = logger
        self.hue_boundary = 90  # 빨강-초록 경계
        
    def classify_region_at_angle(self, frame, target_angle, camera_fov=87):
        """
        특정 각도 방향의 색 영역 판단
        
        Parameters:
            frame: BGR 이미지
            target_angle: LiDAR 각도 (-43.5 ~ +43.5)
            camera_fov: 카메라 수평 FOV (기본 87도)
        
        Returns:
            'RED' or 'GREEN' or 'UNKNOWN'
        """
        h, w = frame.shape[:2]
        
        # 각도 → 픽셀 변환
        normalized = (target_angle + camera_fov / 2) / camera_fov
        x_pixel = int(normalized * w)
        x_pixel = np.clip(x_pixel, 0, w - 1)
        
        # ROI 설정 (세로로 길게, 가로로 좁게)
        x_start = max(0, x_pixel - 25)
        x_end = min(w, x_pixel + 25)
        y_start = h // 4
        y_end = 3 * h // 4
        
        roi = frame[y_start:y_end, x_start:x_end]
        
        if roi.size == 0:
            return 'UNKNOWN'
        
        # HSV 변환
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # 채도/명도 필터 (회색/검정 제외)
        valid_mask = (saturation > 50) & (value > 50)
        
        if not np.any(valid_mask):
            return 'UNKNOWN'
        
        # 유효한 Hue 값들의 평균
        valid_hues = hue[valid_mask]
        avg_hue = np.mean(valid_hues)
        
        # 이분법 판단
        if avg_hue < self.hue_boundary:
            # 0~90: 빨강 영역
            return 'RED'
        else:
            # 90~180: 초록 영역
            return 'GREEN'


class GateNavigator:
    """
    LiDAR 꼬깔 감지 + 색상 이분법 통합 항법
    """
    def __init__(self, logger, node=None):
        self.logger = logger
        self.node = node
        
        self.cone_detector = ConeDetector(logger)
        self.color_classifier = ColorRegionClassifier(logger)
        
        # 카메라 초기화
        self.cap = self.find_camera()
        self.camera_available = (self.cap is not None and self.cap.isOpened())
        
        if self.camera_available:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.logger.info("카메라 활성화")
        else:
            self.logger.warning("카메라 없음 - LiDAR 단독 모드")
        
        # ROS2 퍼블리셔
        if self.node and self.camera_available:
            self.bridge = CvBridge()
            self.debug_pub = self.node.create_publisher(Image, '/gate/debug', 10)
        
        # 게이트 상태
        self.detected_gates = []
        self.target_gate = None
        self.color_rule = None  # {'left': 'GREEN', 'right': 'RED'}
        
        # 플래그
        self.left_cone_flag = False
        self.right_cone_flag = False
        
        self.logger.info("게이트 네비게이터 초기화 완료")
    
    def find_camera(self):
        """RGB 카메라 찾기"""
        for index in range(10):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and len(frame.shape) == 3 and frame.shape[2] == 3:
                    self.logger.info(f"✅ RGB 카메라 발견: video{index}")
                    return cap
                cap.release()
        return None
    
    def update(self, lidar_msg):
        """
        메인 업데이트 (LiDAR + 카메라 융합)
        
        Parameters:
            lidar_msg: sensor_msgs.msg.LaserScan
        """
        # 1. LiDAR로 꼬깔 감지
        cones = self.cone_detector.detect_cones(
            lidar_msg.ranges,
            lidar_msg.angle_min,
            lidar_msg.angle_increment
        )
        
        if len(cones) == 0:
            self.left_cone_flag = False
            self.right_cone_flag = False
            self.target_gate = None
            return
        
        # 2. 카메라 프레임 획득
        frame = None
        if self.camera_available:
            ret, frame = self.cap.read()
            if not ret:
                frame = None
        
        # 3. 각 꼬깔에 색상 레이블 부여
        for cone in cones:
            if frame is not None:
                color_region = self.color_classifier.classify_region_at_angle(
                    frame, cone['angle']
                )
                cone['color'] = color_region
            else:
                cone['color'] = 'UNKNOWN'
        
        # 4. 좌/우 플래그 업데이트
        self._update_cone_flags(cones)
        
        # 5. 유효한 게이트 찾기
        self.detected_gates = self._find_valid_gates(cones)
        
        # 6. 첫 게이트로 색 규칙 학습
        if self.detected_gates and self.color_rule is None:
            self._learn_color_rule(self.detected_gates[0])
        
        # 7. 가장 가까운 게이트 선택
        if self.detected_gates:
            self.target_gate = min(self.detected_gates, key=lambda g: g['distance'])
        else:
            self.target_gate = None
        
        # 8. 디버그 시각화
        if frame is not None and self.node:
            self._publish_debug_image(frame, cones)
    
    def _update_cone_flags(self, cones):
        """
        좌/우 꼬깔 플래그 업데이트
        조건: 각도 기준 좌(-)/우(+) + 색상 일치 + 지속적 감지
        """
        left_cones = [c for c in cones if c['angle'] < -5]  # 왼쪽
        right_cones = [c for c in cones if c['angle'] > 5]  # 오른쪽
        
        # 왼쪽 플래그
        if left_cones:
            # 색상이 있고, 색 규칙과 일치하면 플래그 ON
            if self.color_rule:
                left_match = any(c['color'] == self.color_rule['left'] for c in left_cones)
                self.left_cone_flag = left_match
            else:
                # 색 규칙 없으면 일단 감지만으로 플래그 ON
                self.left_cone_flag = True
        else:
            self.left_cone_flag = False
        
        # 오른쪽 플래그
        if right_cones:
            if self.color_rule:
                right_match = any(c['color'] == self.color_rule['right'] for c in right_cones)
                self.right_cone_flag = right_match
            else:
                self.right_cone_flag = True
        else:
            self.right_cone_flag = False
    
    def _find_valid_gates(self, cones):
        """
        RED-GREEN 쌍으로 유효한 게이트 찾기
        """
        red_cones = [c for c in cones if c['color'] == 'RED']
        green_cones = [c for c in cones if c['color'] == 'GREEN']
        
        if not red_cones or not green_cones:
            return []
        
        gates = []
        for red in red_cones:
            for green in green_cones:
                angle_diff = abs(red['angle'] - green['angle'])
                
                # 게이트 조건: 15~60도 사이
                if 15 < angle_diff < 60:
                    left_cone = red if red['angle'] < green['angle'] else green
                    right_cone = green if red['angle'] < green['angle'] else red
                    
                    mid_angle = (red['angle'] + green['angle']) / 2
                    mid_distance = (red['distance'] + green['distance']) / 2
                    
                    gates.append({
                        'left': left_cone,
                        'right': right_cone,
                        'mid_angle': mid_angle,
                        'distance': mid_distance
                    })
        
        return gates
    
    def _learn_color_rule(self, first_gate):
        """
        첫 게이트로 좌우 색 규칙 학습
        """
        self.color_rule = {
            'left': first_gate['left']['color'],
            'right': first_gate['right']['color']
        }
        self.logger.info(f"🎓 색 규칙 학습: 왼쪽={self.color_rule['left']}, 오른쪽={self.color_rule['right']}")
    
    def _publish_debug_image(self, frame, cones):
        """
        디버그 이미지 퍼블리시 (Foxglove용)
        """
        debug_frame = frame.copy()
        h, w = debug_frame.shape[:2]
        
        # 중앙선
        cv2.line(debug_frame, (w//2, 0), (w//2, h), (128, 128, 128), 2)
        
        # 감지된 꼬깔 표시
        for cone in cones:
            angle = cone['angle']
            x = int((angle + 43.5) / 87 * w)
            
            color_map = {'RED': (0, 0, 255), 'GREEN': (0, 255, 0), 'UNKNOWN': (128, 128, 128)}
            color = color_map.get(cone['color'], (255, 255, 255))
            
            cv2.circle(debug_frame, (x, h//2), 15, color, -1)
            cv2.putText(debug_frame, f"{cone['distance']:.1f}m", 
                       (x-20, h//2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 타겟 게이트 표시
        if self.target_gate:
            left_x = int((self.target_gate['left']['angle'] + 43.5) / 87 * w)
            right_x = int((self.target_gate['right']['angle'] + 43.5) / 87 * w)
            mid_x = (left_x + right_x) // 2
            
            cv2.line(debug_frame, (left_x, h//2), (right_x, h//2), (255, 255, 0), 3)
            cv2.circle(debug_frame, (mid_x, h//2), 20, (255, 0, 255), -1)
        
        # 플래그 상태
        flag_text = f"L:{self.left_cone_flag} R:{self.right_cone_flag}"
        cv2.putText(debug_frame, flag_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 색 규칙
        if self.color_rule:
            rule_text = f"Rule: L={self.color_rule['left']} R={self.color_rule['right']}"
            cv2.putText(debug_frame, rule_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        try:
            msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self.debug_pub.publish(msg)
        except Exception as e:
            self.logger.error(f"디버그 이미지 퍼블리시 실패: {e}")
    
    def get_navigation_command(self):
        """
        항법 명령 반환
        
        Returns:
            'F'(직진), 'L'(좌회전), 'R'(우회전), 'S'(정지), None(미검출)
        """
        if not self.target_gate:
            return None
        
        # 양쪽 플래그 모두 ON이어야 유효
        if not (self.left_cone_flag and self.right_cone_flag):
            return None
        
        # 중앙 각도 기준 조향
        mid_angle = self.target_gate['mid_angle']
        
        if mid_angle < -8:
            return 'L'
        elif mid_angle > 8:
            return 'R'
        else:
            return 'F'
    
    def get_status(self):
        """
        현재 상태 정보
        """
        return {
            'left_flag': self.left_cone_flag,
            'right_flag': self.right_cone_flag,
            'gates_detected': len(self.detected_gates),
            'target_distance': self.target_gate['distance'] if self.target_gate else None,
            'target_angle': self.target_gate['mid_angle'] if self.target_gate else None
        }
    
    def cleanup(self):
        if self.camera_available and self.cap:
            self.cap.release()


class HybridBoatController(Node):
    def __init__(self):
        super().__init__('hybrid_boat_controller')

        self.emergency_stop_time = None
        self.is_in_emergency = False
        self.left_speed = 0
        self.right_speed = 0
        self.speed_step = 10
        self.arduino = None
        self.arduino_connected = False

        self.control_mode = 0  # 0:수동, 1:라이다, 2:색상(기존), 3:게이트
        self.emergency_stop = False

        self.danger_threshold = 0.7
        self.safe_threshold = 1.2
        self.emergency_threshold = 0.15
        
        self.auto_command = 'F'
        self.previous_auto_command = 'F'
        
        # 게이트 네비게이터 초기화
        self.gate_nav = GateNavigator(self.get_logger(), node=self)
        
        try:
            self.settings = termios.tcgetattr(sys.stdin)
        except Exception as e:
            self.get_logger().error(f"터미널 설정 실패: {e}")
            self.settings = None

        self.connect_arduino()

        # LiDAR 구독
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.auto_timer = self.create_timer(0.1, self.auto_control_update)

        self.print_instructions()

    def connect_arduino(self):
        possible_ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
        for port in possible_ports:
            try:
                self.arduino = serial.Serial(port, 115200, timeout=0.1)
                time.sleep(2)
                self.arduino_connected = True
                self.get_logger().info(f"아두이노 연결: {port}")
                break
            except:
                continue

        if not self.arduino_connected:
            self.get_logger().error("아두이노 연결 실패 - 시뮬레이션 모드")

    def print_instructions(self):
        status = "연결완료" if self.arduino_connected else "시뮬레이션"
        camera = "활성" if self.gate_nav.camera_available else "비활성"
        mode_names = ["수동", "라이다", "색상(구)", "게이트"]
        
        print(f"""
{status} - 하이브리드 보트
========================================
현재: {mode_names[self.control_mode]} | 카메라: {camera}

모드: 1(수동) 2(라이다) 3(게이트) x(긴급정지)
수동: w/s(전후) a/d(좌우) space(정지)

Foxglove 토픽:
  - /gate/debug (게이트 검출 시각화)

속도: L{self.left_speed} R{self.right_speed}
========================================
        """)

    def get_key(self):
        if not self.settings:
            return ''
        
        try:
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                key = sys.stdin.read(1)
                if key == '\x1b':
                    rlist2, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist2:
                        sys.stdin.read(2)
                    key = 'ESC'
            else:
                key = ''
        except:
            key = ''
        finally:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            except:
                pass
        return key

    def clamp_speed(self, speed):
        return max(-255, min(255, speed))

    def send_motor_command(self):
        if self.emergency_stop:
            self.left_speed = 0
            self.right_speed = 0

        if not self.arduino_connected:
            return

        try:
            self.arduino.flushInput()
            self.arduino.flushOutput()
            command = f"L{self.left_speed},R{self.right_speed}\n"
            self.arduino.write(command.encode('utf-8'))
            time.sleep(0.05)
        except Exception as e:
            self.get_logger().error(f"통신 에러: {e}")

    def lidar_callback(self, msg):
        """
        LiDAR 콜백 - 게이트 네비게이터 업데이트
        """
        if self.control_mode == 3:
            # 게이트 모드일 때만 업데이트
            self.gate_nav.update(msg)
        elif self.control_mode == 1:
            # 라이다 단독 모드 (기존 로직)
            self.enhanced_scan_callback(msg)

    def enhanced_scan_callback(self, msg):
        """기존 라이다 장애물 회피 로직"""
        try:
            ranges = np.array(msg.ranges)
            ranges = np.where(np.isinf(ranges) | np.isnan(ranges) | (ranges <= 0), 10.0, ranges)
        except:
            return
        
        # 간단한 전방/좌/우 체크
        total = len(ranges)
        front = np.min(ranges[0:30])
        left = np.min(ranges[30:120])
        right = np.min(ranges[total-120:total-30])
        
        if front < 0.5:
            self.auto_command = 'S'
        elif front < 1.0:
            self.auto_command = 'L' if left > right else 'R'
        else:
            self.auto_command = 'F'

    def auto_control_update(self):
        if self.control_mode == 0:
            return
        
        command = None
        
        if self.control_mode == 1:
            # 라이다 단독
            command = self.auto_command
        
        elif self.control_mode == 3:
            # 게이트 항법
            command = self.gate_nav.get_navigation_command()
            status = self.gate_nav.get_status()
            
            if command:
                if command != self.previous_auto_command:
                    self.get_logger().info(
                        f"[게이트] {command} - "
                        f"L:{status['left_flag']} R:{status['right_flag']} "
                        f"Gates:{status['gates_detected']} "
                        f"Dist:{status['target_distance']:.1f}m "
                        f"Angle:{status['target_angle']:.1f}°"
                    )
                    self.previous_auto_command = command
            else:
                command = 'S'
                if command != self.previous_auto_command:
                    self.get_logger().warning(
                        f"[게이트] 미검출 - "
                        f"L:{status['left_flag']} R:{status['right_flag']}"
                    )
                    self.previous_auto_command = command
        
        # 모터 제어
        speed_map = {
            'F': (190, -190),
            'B': (-190, 190),
            'L': (190, 190),
            'R': (-190, -190),
            'S': (0, 0)
        }
        
        if command in speed_map:
            self.left_speed, self.right_speed = speed_map[command]
            self.send_motor_command()

    def run(self):
        if not self.settings:
            return

        try:
            while True:
                key = self.get_key()

                if key == '1':
                    self.control_mode = 0
                    self.emergency_stop = False
                    self.left_speed = self.right_speed = 0
                    print("수동 모드")
                elif key == '2':
                    self.control_mode = 1
                    self.emergency_stop = False
                    print("라이다 모드")
                elif key == '3':
                    self.control_mode = 3
                    self.emergency_stop = False
                    print("게이트 네비게이션 모드")
                elif key == 'x':
                    self.emergency_stop = True
                    self.left_speed = self.right_speed = 0
                    print("긴급정지")
                elif key == '\x03':
                    break

                if self.emergency_stop and key != 'x':
                    continue

                if self.control_mode == 0 and not self.emergency_stop:
                    manual_map = {
                        'w': (175, -175), 's': (-175, 175),
                        'a': (175, 175), 'd': (-175, -175),
                        ' ': (0, 0), 'r': (0, 0)
                    }
                    
                    if key in manual_map:
                        self.left_speed, self.right_speed = manual_map[key]
                    elif key in ['q', 'z', 'e', 'c']:
                        delta = self.speed_step if key in ['q', 'e'] else -self.speed_step
                        if key in ['q', 'z']:
                            self.left_speed = self.clamp_speed(self.left_speed + delta)
                        else:
                            self.right_speed = self.clamp_speed(self.right_speed + delta)

                if key and key != '\x03' and self.control_mode == 0:
                    self.send_motor_command()

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            self.left_speed = self.right_speed = 0
            self.send_motor_command()
            if self.settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            if self.arduino_connected and self.arduino:
                self.arduino.close()
            self.gate_nav.cleanup()
            self.get_logger().info("시스템 종료")
        except Exception as e:
            self.get_logger().error(f"종료 에러: {e}")


def main(args=None):
    rclpy.init(args=args)
    controller = HybridBoatController()

    if not controller.settings:
        controller.destroy_node()
        rclpy.shutdown()
        return

    ros_thread = threading.Thread(target=rclpy.spin, args=(controller,))
    ros_thread.daemon = True
    ros_thread.start()

    try:
        controller.run()
    except Exception as e:
        controller.get_logger().error(f"실행 에러: {e}")
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()