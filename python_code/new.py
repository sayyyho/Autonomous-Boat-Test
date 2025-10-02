# 원격 되는 놈
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import serial
import sys, termios, tty, select
import time
import numpy as np
import cv2
from collections import deque
import threading

class ColorNavigator:
    """색상 기반 네비게이션 모듈"""
    def __init__(self, logger, camera_index=None):
        self.logger = logger
        
        # 카메라 초기화
        self.cap = self.find_camera() if camera_index is None else cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            self.logger.error("카메라 연결 실패!")
            self.camera_available = False
            return
        
        self.camera_available = True
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 색상 범위 설정
        self.setup_color_ranges()
        
        # 화면 설정
        self.display_width = 640
        self.display_height = 480
        
        # 안정화 영역
        self.stable_zones = {'green': None, 'red': None, 'blue': None}
        
        # 네비게이션 상태
        self.target_offset = 0  # 중점 오프셋 (-1 ~ +1)
        self.is_valid_setup = False
        self.last_detection_time = 0
        
        self.logger.info("✅ 색상 네비게이터 초기화 완료")
    
    def find_camera(self):
        """사용 가능한 카메라 찾기"""
        for index in range(5):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.logger.info(f"카메라 발견: index {index}")
                    return cap
                cap.release()
        return None
    
    def setup_color_ranges(self):
        """색상 HSV 범위 설정"""
        self.green_lower = np.array([30, 40, 40])
        self.green_upper = np.array([90, 255, 255])
        
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([12, 255, 255])
        self.red_lower2 = np.array([168, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.blue_lower = np.array([90, 40, 40])
        self.blue_upper = np.array([130, 255, 255])
    
    def detect_cones(self, frame, color_type):
        """색상 콘 검출"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # 마스크 생성
        if color_type == 'green':
            mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        elif color_type == 'blue':
            mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        else:  # red
            mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        
        # 노이즈 제거
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cones = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 800:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                if 0.8 < aspect_ratio < 3.0:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:
                            center_x = x + w // 2
                            center_y = y + h // 2
                            
                            cone_info = {
                                'color': color_type,
                                'pixel_pos': (center_x, center_y),
                                'area': area,
                                'bbox': (x, y, w, h)
                            }
                            cones.append(cone_info)
        
        return cones
    
    def get_best_cone(self, cones):
        """최적 콘 선택"""
        if not cones:
            return None
        
        def score_cone(cone):
            area_score = min(cone['area'] / 5000.0, 1.0)
            center_x = cone['pixel_pos'][0]
            center_distance = abs(center_x - self.display_width // 2)
            center_score = max(0, 1 - center_distance / (self.display_width // 2))
            return area_score * 0.6 + center_score * 0.4
        
        return max(cones, key=score_cone)
    
    def update(self):
        """네비게이션 상태 업데이트"""
        if not self.camera_available:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # 색상 검출
        green_cones = self.detect_cones(frame, 'green')
        red_cones = self.detect_cones(frame, 'red')
        blue_cones = self.detect_cones(frame, 'blue')
        
        best_green = self.get_best_cone(green_cones)
        best_red = self.get_best_cone(red_cones)
        best_blue = self.get_best_cone(blue_cones)
        
        # 오른쪽 마커 선택 (초록 우선, 없으면 파랑)
        right_cone = best_green if best_green else best_blue
        
        # 유효성 검증: 빨강(왼쪽) + 초록/파랑(오른쪽)
        if best_red and right_cone:
            red_x = best_red['pixel_pos'][0]
            right_x = right_cone['pixel_pos'][0]
            
            # 빨강이 왼쪽에 있고, 충분한 간격이 있는지 확인
            if red_x < right_x and abs(red_x - right_x) >= 50:
                self.is_valid_setup = True
                self.last_detection_time = time.time()
                
                # 중점 계산 및 정규화 (-1 ~ +1)
                mid_pixel_x = (red_x + right_x) // 2
                screen_center = self.display_width // 2
                self.target_offset = (mid_pixel_x - screen_center) / screen_center
            else:
                self.is_valid_setup = False
        else:
            self.is_valid_setup = False
        
        # 3초 이상 미검출 시 무효화
        if time.time() - self.last_detection_time > 3.0:
            self.is_valid_setup = False
    
    def get_navigation_command(self):
        """네비게이션 명령 반환"""
        if not self.is_valid_setup:
            return None
        
        # 오프셋 기반 명령 생성
        if self.target_offset < -0.15:  # 왼쪽으로 치우침
            return 'L'
        elif self.target_offset > 0.15:  # 오른쪽으로 치우침
            return 'R'
        else:
            return 'F'  # 직진
    
    def cleanup(self):
        """카메라 해제"""
        if self.camera_available:
            self.cap.release()


class HybridBoatController(Node):
    def __init__(self):
        super().__init__('hybrid_boat_controller')

        # 모터 속도 초기화
        self.emergency_stop_time = None
        self.is_in_emergency = False
        self.left_speed = 0
        self.right_speed = 0
        self.speed_step = 10
        self.arduino = None
        self.arduino_connected = False

        # 제어 모드 (0: 수동, 1: 라이다, 2: 색상)
        self.control_mode = 0
        self.emergency_stop = False

        # 라이다 회피 파라미터
        self.danger_threshold = 0.7
        self.safe_threshold = 1.2
        self.emergency_threshold = 0.15
        self.front_angle = 30
        self.side_angle = 90
        
        self.auto_command = 'F'
        self.previous_auto_command = 'F'
        
        # 색상 네비게이터 초기화
        self.color_nav = ColorNavigator(self.get_logger())
        
        # 색상 네비게이션 업데이트 스레드
        if self.color_nav.camera_available:
            self.color_update_thread = threading.Thread(target=self.color_update_loop, daemon=True)
            self.color_update_thread.start()

        # 터미널 설정
        try:
            self.settings = termios.tcgetattr(sys.stdin)
        except Exception as e:
            self.get_logger().error(f"터미널 설정 실패: {e}")
            self.settings = None

        # 아두이노 연결
        self.connect_arduino()

        # 라이다 구독
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.enhanced_scan_callback,
            10
        )

        # 자동 모드용 타이머
        self.auto_timer = self.create_timer(0.1, self.auto_control_update)

        self.print_instructions()

    def connect_arduino(self):
        possible_ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
        for port in possible_ports:
            try:
                self.arduino = serial.Serial(port, 115200, timeout=0.1)
                time.sleep(2)
                self.arduino_connected = True
                self.get_logger().info(f"아두이노 연결 성공! 포트: {port}")
                break
            except Exception as e:
                continue

        if not self.arduino_connected:
            self.get_logger().error("아두이노 연결 실패! 시뮬레이션 모드")

    def print_instructions(self):
        status = "연결완료!!" if self.arduino_connected else "시뮬레이션 모드"
        camera_status = "활성화" if self.color_nav.camera_available else "비활성화"
        
        mode_names = ["수동모드", "라이다 자동", "색상 네비게이션"]
        mode = mode_names[self.control_mode]
        
        print(f"""
{status} - 하이브리드 보트 제어 시스템 (색상 네비게이션 통합)
========================================
현재 모드: {mode}
카메라 상태: {camera_status}

모드 전환:
1 : 수동 모드 (키보드 조종)
2 : 라이다 자동 모드 (장애물 회피)
3 : 색상 네비게이션 모드 (빨강-초록/파랑)
x : 긴급 정지

=== 수동 모드 조작법 ===
w : 전진     s : 후진
a : 좌회전   d : 우회전
space : 정지

개별 모터 제어:
q/z : 좌측 모터 +/-
e/c : 우측 모터 +/-
k/l : 현재 방향 가속/감속

=== 라이다 자동 모드 ===
장애물 자동 회피
긴급정지: {self.emergency_threshold}m
위험거리: {self.danger_threshold}m
안전거리: {self.safe_threshold}m

=== 색상 네비게이션 모드 ===
🔴 빨강(왼쪽) + 🟢 초록 또는 🔵 파랑(오른쪽)
→ 중점을 향해 자동 주행

r : 리셋    Ctrl+C : 종료
========================================
현재 속도 - 좌측: {self.left_speed}, 우측: {self.right_speed}
        """)

    def color_update_loop(self):
        """색상 네비게이션 업데이트 루프"""
        while True:
            if self.control_mode == 2:  # 색상 모드일 때만 업데이트
                self.color_nav.update()
            time.sleep(0.1)  # 10Hz

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
                        next_chars = sys.stdin.read(2)
                        if next_chars == 'OP':
                            key = 'F1'
                        elif next_chars == 'OQ':
                            key = 'F2'
                        else:
                            key = 'ESC'
                    else:
                        key = 'ESC'
            else:
                key = ''
        except Exception as e:
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
        
            if self.arduino.in_waiting:
                response = self.arduino.readline().decode('utf-8', errors='ignore').strip()
        except Exception as e:
            self.get_logger().error(f"통신 에러: {e}")

    def apply_noise_filter(self, ranges):
        """라이다 노이즈 필터링"""
        filtered = np.copy(ranges)
        for i in range(1, len(ranges)-1):
            window = ranges[i-1:i+2]
            filtered[i] = np.median(window)
        return filtered

    def get_sector_distances(self, ranges, sector):
        """섹터별 거리 데이터 추출"""
        total_points = len(ranges)
        
        if sector == "FRONT":
            angle_range = 30
            front_start = max(0, total_points - angle_range)
            front_end = min(total_points, angle_range)
            return np.concatenate([ranges[0:front_end], ranges[front_start:]])
        
        elif sector == "LEFT":
            left_start = min(total_points - 1, 30)
            left_end = min(total_points, 120)
            return ranges[left_start:left_end]
        
        elif sector == "RIGHT":
            right_start = max(0, total_points - 120)
            right_end = max(0, total_points - 30)
            return ranges[right_start:right_end]
        
        return np.array([10.0])

    def calculate_representative_distance(self, distances):
        """영역의 대표 거리 계산"""
        if len(distances) == 0:
            return 10.0
        
        min_dist = np.min(distances)
        avg_dist = np.mean(distances)
        
        representative = min_dist * 0.7 + avg_dist * 0.3
        return representative

    def enhanced_scan_callback(self, msg):
        """라이다 데이터 처리"""
        if self.control_mode != 1:  # 라이다 모드가 아니면 무시
            return

        try:
            ranges = np.array(msg.ranges)
            ranges = np.where(np.isinf(ranges), 10.0, ranges)
            ranges = np.where(np.isnan(ranges), 10.0, ranges)
            ranges = np.where(ranges <= 0, 10.0, ranges)
            ranges = self.apply_noise_filter(ranges)
            
        except Exception as e:
            self.get_logger().error(f"라이다 데이터 처리 에러: {e}")
            return
        
        total_points = len(ranges)
        if total_points == 0:
            return
        
        front_distances = self.get_sector_distances(ranges, "FRONT")
        left_distances = self.get_sector_distances(ranges, "LEFT") 
        right_distances = self.get_sector_distances(ranges, "RIGHT")
        
        front_min = self.calculate_representative_distance(front_distances)
        left_min = self.calculate_representative_distance(left_distances)
        right_min = self.calculate_representative_distance(right_distances)
        
        new_command = self.decide_auto_movement(front_min, left_min, right_min)
        
        if new_command != self.previous_auto_command:
            direction_map = {'F': '직진', 'B': '후진', 'L': '좌회전', 'R': '우회전', 'S': '정지'}
            print(f"[라이다] {direction_map.get(new_command, new_command)}")
            self.previous_auto_command = new_command
        
        self.auto_command = new_command

    def decide_auto_movement(self, front, left, right):
        """장애물 회피 로직"""
        # 긴급 정지
        if front < self.emergency_threshold or left < self.emergency_threshold or right < self.emergency_threshold:
            if not self.is_in_emergency:
                self.emergency_stop_time = time.time()
                self.is_in_emergency = True
            if time.time() - self.emergency_stop_time >= 0.3:
                self.is_in_emergency = False
                if left > right and left > front:
                    return 'L'
                elif right > left and right > front:
                    return 'R'
                else:
                    return 'B'
            return 'S'

        # 직진 가능 여부
        if front > self.safe_threshold:
            return 'F'
        elif front > self.danger_threshold:
            return 'F'
        
        # 회피
        if left > right:
            return 'L'
        else:
            return 'R'

    def auto_control_update(self):
        """자동 제어 업데이트"""
        if self.control_mode == 0:  # 수동 모드
            return
        
        command = None
        
        if self.control_mode == 1:  # 라이다 모드
            command = self.auto_command
        
        elif self.control_mode == 2:  # 색상 네비게이션 모드
            command = self.color_nav.get_navigation_command()
            
            if command:
                if command != self.previous_auto_command:
                    direction_map = {'F': '직진', 'L': '좌회전', 'R': '우회전'}
                    valid_status = "✅" if self.color_nav.is_valid_setup else "❌"
                    print(f"[색상] {valid_status} {direction_map.get(command, command)} (오프셋: {self.color_nav.target_offset:.2f})")
                    self.previous_auto_command = command
            else:
                # 색상 미검출 시 정지
                command = 'S'
                if self.previous_auto_command != 'S':
                    print("[색상] ⚠️ 콘 미검출 - 정지")
                    self.previous_auto_command = 'S'
        
        # 명령을 모터 속도로 변환
        if command == 'F':
            self.left_speed = 190
            self.right_speed = -190
        elif command == 'B':
            self.left_speed = -190
            self.right_speed = 190
        elif command == 'L':
            self.left_speed = 190
            self.right_speed = 190
        elif command == 'R':
            self.left_speed = -190
            self.right_speed = -190
        elif command == 'S':
            self.left_speed = 0
            self.right_speed = 0

        self.send_motor_command()

    def run(self):
        if not self.settings:
            self.get_logger().error("터미널 설정 실패")
            return

        try:
            while True:
                key = self.get_key()

                # 모드 전환
                if key == '1':
                    self.control_mode = 0
                    self.emergency_stop = False
                    self.left_speed = 0
                    self.right_speed = 0
                    print("🎮 수동 모드")
                    
                elif key == '2':
                    self.control_mode = 1
                    self.emergency_stop = False
                    print("🎯 라이다 자동 모드")
                    
                elif key == '3':
                    if self.color_nav.camera_available:
                        self.control_mode = 2
                        self.emergency_stop = False
                        print("🎨 색상 네비게이션 모드")
                    else:
                        print("❌ 카메라를 사용할 수 없습니다")
                    
                elif key == 'x':
                    self.emergency_stop = True
                    self.left_speed = 0
                    self.right_speed = 0
                    print("🚨 긴급 정지")

                elif key == '\x03':  # Ctrl+C
                    break

                # 긴급정지 상태에서는 키 입력 무시
                if self.emergency_stop and key != 'x':
                    continue

                # 수동 모드에서만 키보드 조작
                if self.control_mode == 0 and not self.emergency_stop:
                    if key == 'w':
                        self.left_speed = 175
                        self.right_speed = -175
                    elif key == 's':
                        self.left_speed = -175
                        self.right_speed = 175
                    elif key == 'a':
                        self.left_speed = 175
                        self.right_speed = 175
                    elif key == 'd':
                        self.left_speed = -175
                        self.right_speed = -175
                    elif key == ' ':
                        self.left_speed = 0
                        self.right_speed = 0
                    elif key == 'r':
                        self.left_speed = 0
                        self.right_speed = 0
                    elif key == 'q':
                        self.left_speed = self.clamp_speed(self.left_speed + self.speed_step)
                    elif key == 'z':
                        self.left_speed = self.clamp_speed(self.left_speed - self.speed_step)
                    elif key == 'e':
                        self.right_speed = self.clamp_speed(self.right_speed + self.speed_step)
                    elif key == 'c':
                        self.right_speed = self.clamp_speed(self.right_speed - self.speed_step)
                    elif key == 'k':
                        if self.left_speed > 0:
                            self.left_speed = self.clamp_speed(self.left_speed + 10)
                        elif self.left_speed < 0:
                            self.left_speed = self.clamp_speed(self.left_speed - 10)
                        if self.right_speed > 0:
                            self.right_speed = self.clamp_speed(self.right_speed + 10)
                        elif self.right_speed < 0:
                            self.right_speed = self.clamp_speed(self.right_speed - 10)
                    elif key == 'l':
                        if self.left_speed < 0:
                            self.left_speed = self.clamp_speed(self.left_speed + 10)
                        elif self.left_speed > 0:
                            self.left_speed = self.clamp_speed(self.left_speed - 10)
                        if self.right_speed < 0:
                            self.right_speed = self.clamp_speed(self.right_speed + 10)
                        elif self.right_speed > 0:
                            self.right_speed = self.clamp_speed(self.right_speed - 10)

                if key and key != '\x03' and self.control_mode == 0:
                    self.send_motor_command()

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        """종료 시 정리"""
        try:
            self.left_speed = 0
            self.right_speed = 0
            self.send_motor_command()

            if self.settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            
            if self.arduino_connected and self.arduino:
                self.arduino.close()
            
            # 카메라 해제
            self.color_nav.cleanup()
                
            self.get_logger().info("시스템 종료")
        except Exception as e:
            self.get_logger().error(f"종료 처리 중 에러: {e}")

def main(args=None):
    rclpy.init(args=args)
    controller = HybridBoatController()

    if not controller.settings:
        controller.destroy_node()
        rclpy.shutdown()
        return

    import threading
    ros_thread = threading.Thread(target=rclpy.spin, args=(controller,))
    ros_thread.daemon = True
    ros_thread.start()

    try:
        controller.run()
    except Exception as e:
        controller.get_logger().error(f"실행 중 에러: {e}")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()