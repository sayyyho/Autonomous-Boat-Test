# 1002 테스트본
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

class ColorNavigator:
    def __init__(self, logger, camera_index=None, node=None):
        self.logger = logger
        self.node = node
        
        if camera_index is None:
            self.cap = self.find_camera()
        else:
            self.cap = cv2.VideoCapture(camera_index)
    
        # 카메라 없으면 바로 종료
        if self.cap is None or not self.cap.isOpened():
            self.logger.error("카메라 연결 실패!")
            self.camera_available = False
            self.cap = None
            return
    
        self.camera_available = True
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # RealSense 카메라 설정 조정
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        self.logger.info("카메라 설정: 화이트밸런스 수동, 노출 조정")
        
        # 이미지 퍼블리셔 (Foxglove용)
        if self.node:
            self.bridge = CvBridge()
            self.image_pub = self.node.create_publisher(Image, '/camera/color/image_raw', 10)
            self.debug_pub = self.node.create_publisher(Image, '/camera/color/debug', 10)
            self.logger.info("카메라 토픽 퍼블리시: /camera/color/image_raw, /camera/color/debug")
        
        # 엄격한 색상 범위 설정
        self.setup_strict_color_ranges()
        
        # 화면 설정
        self.display_width = 640
        self.display_height = 480
        
        # 네비게이션 상태
        self.target_offset = 0.0
        self.is_valid_setup = False
        self.last_detection_time = 0
        
        # 검출 데이터
        self.gb_data = {'detected': False}
        self.red_data = {'detected': False}
        
        self.logger.info("색상 네비게이터 초기화 완료")
    
    def find_camera(self):
        """사용 가능한 RGB 카메라 찾기"""
        self.logger.info("RGB 카메라 검색 중...")
        
        for index in range(30):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    shape = frame.shape
                    dtype = frame.dtype
                    self.logger.info(f"video{index}: shape={shape}, dtype={dtype}")
                    
                    if len(shape) == 3 and shape[2] == 3:
                        mean_color = frame.mean(axis=(0,1))
                        self.logger.info(f"✅ RGB 카메라 발견! video{index}, Mean BGR={mean_color}")
                        return cap
                    else:
                        self.logger.info(f"video{index}는 RGB가 아님")
                        cap.release()
                else:
                    self.logger.info(f"video{index}: 프레임 읽기 실패")
                    cap.release()
            else:
                self.logger.info(f"video{index}: 열기 실패")
        
        self.logger.warning("RGB 카메라를 찾지 못했습니다")
        return None
    
    def setup_strict_color_ranges(self):
        """엄격한 색상 HSV 범위"""
        # self.green_blue_lower = np.array([35, 80, 80])
        # self.green_blue_upper = np.array([125, 255, 255])
        
        # self.red_lower1 = np.array([0, 150, 150])
        # self.red_upper1 = np.array([8, 255, 255])
        # self.red_lower2 = np.array([172, 150, 150])
        # self.red_upper2 = np.array([180, 255, 255])

        self.green_blue_lower = np.array([80, 80, 80])   # 하늘색부터
        self.green_blue_upper = np.array([140, 255, 255]) # 보라 전까지

        self.red_lower1 = np.array([0, 150, 150])
        self.red_upper1 = np.array([8, 255, 255])
        self.red_lower2 = np.array([172, 150, 150])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.logger.info("STRICT MODE: GB[35-125 S80+ V80+], RED[0-8|172-180 S150+ V150+]")
    
    def color_correction(self, frame):
        """초록끼 보정"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        b = cv2.add(b, 10)
        lab = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return corrected

    def find_color_center(self, frame, color_type):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        if color_type == 'green_blue':
            mask = cv2.inRange(hsv, self.green_blue_lower, self.green_blue_upper)
        else:
            mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        moments = cv2.moments(mask)
        screen_area = self.display_width * self.display_height
        area = moments["m00"]
        coverage_percent = (area / screen_area) * 100
        coverage_percent = min(coverage_percent, 100.0)
        
        if coverage_percent >= 0.5:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            
            return {
                'center': (center_x, center_y),
                'area': area,
                'coverage': coverage_percent,
                'detected': True
            }
        
        return {
            'detected': False,
            'area': area,
            'coverage': coverage_percent
        }
    
    def update(self):
        """네비게이션 상태 업데이트"""
        if not self.camera_available or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # 원본 이미지 퍼블리시 (Foxglove용)
        if self.node:
            try:
                img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.image_pub.publish(img_msg)
            except Exception as e:
                self.logger.error(f"이미지 퍼블리시 실패: {e}")
        
        frame = self.color_correction(frame)
        
        # 색상 중심점 검출
        self.gb_data = self.find_color_center(frame, 'green_blue')
        self.red_data = self.find_color_center(frame, 'red')
        
        # 디버그 이미지 생성 및 퍼블리시
        if self.node:
            debug_frame = frame.copy()
            
            # 검출된 색상 표시
            if self.gb_data['detected']:
                cv2.circle(debug_frame, self.gb_data['center'], 15, (0, 255, 255), -1)
                cv2.putText(debug_frame, f"GB:{self.gb_data['coverage']:.1f}%", 
                           (self.gb_data['center'][0]-40, self.gb_data['center'][1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            if self.red_data['detected']:
                cv2.circle(debug_frame, self.red_data['center'], 15, (0, 0, 255), -1)
                cv2.putText(debug_frame, f"RED:{self.red_data['coverage']:.1f}%", 
                           (self.red_data['center'][0]-40, self.red_data['center'][1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 중앙선 및 타겟 표시
            if self.gb_data['detected'] and self.red_data['detected']:
                mid_x = (self.gb_data['center'][0] + self.red_data['center'][0]) // 2
                mid_y = (self.gb_data['center'][1] + self.red_data['center'][1]) // 2
                cv2.line(debug_frame, self.gb_data['center'], self.red_data['center'], (255, 255, 255), 2)
                cv2.circle(debug_frame, (mid_x, mid_y), 10, (255, 255, 0), -1)
            
            # 화면 중앙선
            screen_center = self.display_width // 2
            cv2.line(debug_frame, (screen_center, 0), (screen_center, self.display_height), (128, 128, 128), 2)
            
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
                self.debug_pub.publish(debug_msg)
            except Exception as e:
                self.logger.error(f"디버그 이미지 퍼블리시 실패: {e}")
        
        # 양쪽 모두 검출되었는지 확인
        if self.gb_data['detected'] and self.red_data['detected']:
            gb_x = self.gb_data['center'][0]
            red_x = self.red_data['center'][0]
            
            if abs(gb_x - red_x) >= 50:
                self.is_valid_setup = True
                self.last_detection_time = time.time()
                
                mid_pixel_x = (gb_x + red_x) // 2
                screen_center = self.display_width // 2
                self.target_offset = (mid_pixel_x - screen_center) / screen_center
            else:
                self.is_valid_setup = False
        else:
            self.is_valid_setup = False
        
        if time.time() - self.last_detection_time > 3.0:
            self.is_valid_setup = False
    
    def get_navigation_command(self):
        if not self.is_valid_setup:
            return None
        
        if self.target_offset < -0.15:
            return 'L'
        elif self.target_offset > 0.15:
            return 'R'
        else:
            return 'F'
    
    def get_detection_info(self):
        return {
            'valid': self.is_valid_setup,
            'gb_coverage': self.gb_data.get('coverage', 0),
            'red_coverage': self.red_data.get('coverage', 0),
            'offset': self.target_offset
        }
    
    def cleanup(self):
        if self.camera_available and self.cap is not None:
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

        self.control_mode = 0
        self.emergency_stop = False

        self.danger_threshold = 0.7
        self.safe_threshold = 1.2
        self.emergency_threshold = 0.15
        
        self.auto_command = 'F'
        self.previous_auto_command = 'F'
        self.color_log_counter = 0
        
        # 색상 네비게이터 초기화 (node 전달)
        self.color_nav = ColorNavigator(self.get_logger(), node=self)
        
        if self.color_nav.camera_available:
            self.color_update_thread = threading.Thread(target=self.color_update_loop, daemon=True)
            self.color_update_thread.start()

        try:
            self.settings = termios.tcgetattr(sys.stdin)
        except Exception as e:
            self.get_logger().error(f"터미널 설정 실패: {e}")
            self.settings = None

        self.connect_arduino()

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.enhanced_scan_callback,
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
        camera = "활성" if self.color_nav.camera_available else "비활성"
        mode_names = ["수동", "라이다", "색상네비"]
        
        print(f"""
{status} - 하이브리드 보트
========================================
현재: {mode_names[self.control_mode]} | 카메라: {camera}

모드: 1(수동) 2(라이다) 3(색상) x(긴급정지)
수동: w/s(전후) a/d(좌우) space(정지)

Foxglove 토픽:
  - /camera/color/image_raw (원본)
  - /camera/color/debug (검출 표시)

속도: L{self.left_speed} R{self.right_speed}
========================================
        """)

    def color_update_loop(self):
        while True:
            if self.color_nav.camera_available:
                self.color_nav.update()
            time.sleep(0.1)

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

    def apply_noise_filter(self, ranges):
        filtered = np.copy(ranges)
        for i in range(1, len(ranges)-1):
            filtered[i] = np.median(ranges[i-1:i+2])
        return filtered

    def get_sector_distances(self, ranges, sector):
        total = len(ranges)
        if sector == "FRONT":
            return np.concatenate([ranges[0:30], ranges[total-30:]])
        elif sector == "LEFT":
            return ranges[30:120]
        elif sector == "RIGHT":
            return ranges[total-120:total-30]
        return np.array([10.0])

    def calculate_representative_distance(self, distances):
        if len(distances) == 0:
            return 10.0
        return np.min(distances) * 0.7 + np.mean(distances) * 0.3

    def enhanced_scan_callback(self, msg):
        if self.control_mode != 1:
            return

        try:
            ranges = np.array(msg.ranges)
            ranges = np.where(np.isinf(ranges) | np.isnan(ranges) | (ranges <= 0), 10.0, ranges)
            ranges = self.apply_noise_filter(ranges)
        except:
            return
        
        front = self.calculate_representative_distance(self.get_sector_distances(ranges, "FRONT"))
        left = self.calculate_representative_distance(self.get_sector_distances(ranges, "LEFT"))
        right = self.calculate_representative_distance(self.get_sector_distances(ranges, "RIGHT"))
        
        new_cmd = self.decide_auto_movement(front, left, right)
        
        if new_cmd != self.previous_auto_command:
            direction_map = {'F':'직진','B':'후진','L':'좌회전','R':'우회전','S':'정지'}
            print(f"[라이다] {direction_map[new_cmd]}")
            self.previous_auto_command = new_cmd
        
        self.auto_command = new_cmd

    def decide_auto_movement(self, front, left, right):
        if front < self.emergency_threshold or left < self.emergency_threshold or right < self.emergency_threshold:
            if not self.is_in_emergency:
                self.emergency_stop_time = time.time()
                self.is_in_emergency = True
            if time.time() - self.emergency_stop_time >= 0.3:
                self.is_in_emergency = False
                return 'L' if left > max(right, front) else 'R' if right > front else 'B'
            return 'S'

        if front > self.safe_threshold:
            return 'F'
        return 'L' if left > right else 'R'

    def auto_control_update(self):
        if self.control_mode == 0:
            return
        
        command = None
        
        if self.control_mode == 1:
            command = self.auto_command
        
        elif self.control_mode == 2:
            self.color_log_counter += 1
            
            command = self.color_nav.get_navigation_command()
            info = self.color_nav.get_detection_info()
            
            should_log = (self.color_log_counter >= 30) or (command != self.previous_auto_command)
            
            if command:
                if should_log:
                    status = "OK" if info['valid'] else "X"
                    print(f"[색상] {status} {command} - GB:{info['gb_coverage']:.1f}% RED:{info['red_coverage']:.1f}% Offset:{info['offset']:.2f}")
                    self.color_log_counter = 0
            else:
                command = 'S'
                if should_log:
                    print(f"[색상] 미검출 - GB:{info['gb_coverage']:.1f}% RED:{info['red_coverage']:.1f}%")
                    self.color_log_counter = 0
            
            self.previous_auto_command = command
        
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
                    if self.color_nav.camera_available:
                        self.control_mode = 2
                        self.emergency_stop = False
                        print("색상 네비게이션")
                    else:
                        print("카메라 없음")
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
            self.color_nav.cleanup()
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