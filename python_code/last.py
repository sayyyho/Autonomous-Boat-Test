#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import serial
import sys, termios, tty, select
import time
import numpy as np

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

        # 제어 모드 (True: 자동, False: 수동)
        self.auto_mode = False
        self.emergency_stop = False

        # 라이다 회피 파라미터
        self.danger_threshold = 1
        self.safe_threshold = 1.2
        self.emergency_threshold = 0.15  # 긴급 정지 거리 추가
        self.front_angle = 45         # 전방 감지 각도 (±45도)
        self.side_angle = 90          # 좌우 감지 각도
        
        self.auto_command = 'F'
        self.command_count = 0
        self.previous_auto_command = 'F'
        
        # 최소한의 출력 제어
        self.last_simple_log = 0
        self.simple_log_interval = 10.0  # 10초마다 간단한 상태만 출력
        
        # 즉시 반응 설정
        self.fast_response = True
        self.stability_threshold = 0  # 즉시 반응

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
            self.scan_callback,
            10
        )

        # 자동 모드용 타이머 (즉시 반응을 위해 0.02초)
        self.auto_timer = self.create_timer(0.1, self.auto_control_update)
        
        # 상태 표시 타이머 (5초마다)
        self.status_timer = self.create_timer(5.0, self.update_status_display)

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
        mode = "자동모드" if self.auto_mode else "수동모드"
        print(f"""
{status} - 하이브리드 보트 제어 시스템
========================================
현재 모드: {mode}

모드 전환:
1 : 수동 모드 (키보드 조종)
2 : 자동 모드 (라이다 회피)
x : 긴급 정지

=== 수동 모드 조작법 ===
w : 전진     s : 후진
a : 좌회전   d : 우회전
space : 정지

개별 모터 제어:
q/z : 좌측 모터 +/-
e/c : 우측 모터 +/-
k/l : 현재 방향 가속/감속

=== 자동 모드 ===
라이다로 장애물 자동 회피
긴급정지: {self.emergency_threshold}m
위험거리: {self.danger_threshold}m
안전거리: {self.safe_threshold}m

r : 리셋    Ctrl+C : 종료
========================================
현재 속도 - 좌측: {self.left_speed}, 우측: {self.right_speed}
        """)

    def get_key(self):
        if not self.settings:
            return ''
        
        try:
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                key = sys.stdin.read(1)
                # F1, F2, ESC 키 처리
                if key == '\x1b':  # ESC 시퀀스 시작
                    # 추가 문자가 있는지 확인
                    rlist2, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist2:
                        next_chars = sys.stdin.read(2)
                        if next_chars == 'OP':  # F1
                            key = 'F1'
                        elif next_chars == 'OQ':  # F2
                            key = 'F2'
                        else:  # 다른 ESC 시퀀스
                            key = 'ESC'
                    else:  # ESC만 눌림
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
        # 버퍼 비우기
            self.arduino.flushInput()
            self.arduino.flushOutput()
        
            command = f"L{self.left_speed},R{self.right_speed}\n"
            self.arduino.write(command.encode('utf-8'))
        
        # 아두이노 응답 대기 (중요!)
            time.sleep(0.05)  # 50ms 대기
        
        # 응답 확인
            if self.arduino.in_waiting:
                response = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                # print(response)
        except Exception as e:
            self.get_logger().error(f"통신 에러: {e}")

    def scan_callback(self, msg):
        if not self.auto_mode:
            return

        # 안전한 배열 처리
        try:
            ranges = np.array(msg.ranges)
            ranges = np.where(np.isinf(ranges), 10.0, ranges)
            ranges = np.where(np.isnan(ranges), 10.0, ranges)
            ranges = np.where(ranges <= 0, 10.0, ranges)  # 0 이하 값 처리
        except Exception as e:
            self.get_logger().error(f"라이다 데이터 처리 에러: {e}")
            return
        
        total_points = len(ranges)
        if total_points == 0:
            return
        
        # 각도별 영역 계산 (경계값 안전 처리)
        front_start = max(0, total_points - self.front_angle)
        front_end = min(total_points, self.front_angle)
        
        left_start = min(total_points - 1, self.front_angle)
        left_end = min(total_points, self.side_angle + self.front_angle)
        
        right_start = max(0, total_points - (self.side_angle + self.front_angle))
        right_end = max(0, total_points - self.front_angle)
        
        # 각 영역의 최소 거리 (안전한 슬라이싱)
        try:
            front_ranges = np.concatenate([
                ranges[0:front_end] if front_end > 0 else np.array([]),
                ranges[front_start:] if front_start < total_points else np.array([])
            ]) if front_start != front_end else ranges[0:front_end]
            
            left_ranges = ranges[left_start:left_end] if left_start < left_end else np.array([10.0])
            right_ranges = ranges[right_start:right_end] if right_start < right_end else np.array([10.0])
            
            front_min = np.min(front_ranges) if len(front_ranges) > 0 else 10.0
            left_min = np.min(left_ranges) if len(left_ranges) > 0 else 10.0
            right_min = np.min(right_ranges) if len(right_ranges) > 0 else 10.0
        except Exception as e:
            self.get_logger().error(f"거리 계산 에러: {e}")
            return
        
        # 자동 회피 결정
        new_command = self.decide_auto_movement(front_min, left_min, right_min)
        decision_reason = self.get_decision_reason(front_min, left_min, right_min, new_command)
        
        # 명령 안정화 제거 - 즉시 반응
        self.auto_command = new_command  # 바로 적용
        
        command_changed = new_command != self.previous_auto_command
        self.previous_auto_command = new_command
        
        command_changed = new_command != self.previous_auto_command
        
        # 출력 최소화 - 명령 변경 시에만
        if command_changed:
            direction_map = {'F': '직진', 'B': '후진', 'L': '좌회전', 'R': '우회전', 'S': '정지'}
            print(f"{direction_map.get(new_command, new_command)}")
        
        # 10초마다 간단한 상태
        current_time = time.time()
        if current_time - self.last_simple_log >= self.simple_log_interval:
            self.last_simple_log = current_time
            print(f"AUTO: {self.auto_command}")

    def get_status_text(self, distance):
        if distance < self.danger_threshold:
            return "위험"
        elif distance < self.safe_threshold:
            return "주의"
        else:
            return "안전"

    def print_detailed_status(self, front_min, left_min, right_min, new_command, command_changed, decision_reason):
        print("\n" + "="*80)
        print(f"자동 모드 - 라이다 스캔 분석")
        print("-"*80)
        
        # 위험도 평가
        front_status = self.get_status_text(front_min)
        left_status = self.get_status_text(left_min)
        right_status = self.get_status_text(right_min)
        
        print(f"위험도 평가:")
        print(f"  전방: {front_status} (기준: 위험<{self.danger_threshold}m, 안전>{self.safe_threshold}m)")
        print(f"  좌측: {left_status}")
        print(f"  우측: {right_status}")
        
        print(f"의사결정:")
        print(f"  이전 명령: {self.previous_auto_command}")
        print(f"  새 명령: {new_command}")
        print(f"  명령 변경: {'예' if command_changed else '아니오'}")
        print(f"  안정화 카운트: {self.command_count}/{self.stability_threshold}")
        print(f"  최종 명령: {self.auto_command}")
        print(f"  결정 이유: {decision_reason}")

    def get_decision_reason(self, front, left, right, command):
        """의사결정 이유를 반환"""
        if front < self.emergency_threshold or left < self.emergency_threshold or right < self.emergency_threshold:
            min_dist = min(front, left, right)
            return f"긴급정지 - 장애물 너무 가까움({min_dist:.2f}m < {self.emergency_threshold}m)"

        is_surrounded = (front < self.danger_threshold and left < self.danger_threshold and right < self.danger_threshold)
        if is_surrounded:
            return f"사방 막힘(수조 코너) → 좌회전으로 탈출 시도"
            
        if front < self.danger_threshold:
            return f"전방 위험({front:.2f}m) → 더 안전한 방향({('좌' if left > right else '우')})으로 회전"
        
        if left < self.danger_threshold and right < self.danger_threshold:
            return f"좁은 통로 → 전방({front:.2f}m)이 안전하므로 전진"
        
        if left < self.danger_threshold:
            return f"좌측 위험({left:.2f}m) → 우회전"
        if right < self.danger_threshold:
            return f"우측 위험({right:.2f}m) → 좌회전"
            
        if front > self.safe_threshold:
            return f"모든 방향 안전 → 직진"
        else:
            return f"전방 주의({front:.2f}m) → 더 넓은 방향으로 선회"

    def decide_auto_movement(self, front, left, right):
        # 1. 긴급 정지 (최우선 순위)
        # 장애물이 너무 가까우면(emergency_threshold) 일단 정지 후 후진합니다.
        if front < self.emergency_threshold or left < self.emergency_threshold or right < self.emergency_threshold:
            if not self.is_in_emergency:
                self.emergency_stop_time = time.time()
                self.is_in_emergency = True
            if time.time() - self.emergency_stop_time >= 0.5:
                self.is_in_emergency = False
                return 'B'  # 공간 확보를 위해 후진
            return 'S'  # 1.5초간 정지

        # ========================================================== #
        # ========== 요청하신 '사방이 벽일 때' 로직 추가 ========== #
        # ========================================================== #
        # 2. 수조 코너 등 사방이 막힌 경우 (두 번째 우선 순위)
        # front, left, right 모두 위험 거리 안쪽이면 '좌회전'을 기본 동작으로 지정합니다.
        is_surrounded = (front < self.danger_threshold and
                         left < self.danger_threshold and
                         right < self.danger_threshold)
        if is_surrounded:
            return 'L'  # 무조건 좌회전하여 벽을 따라 이동 시작
        # ========================================================== #

        # 3. 일반 장애물 회피 로직 (기존 로직 재구성)
        
        # 3-1. 전방이 위험할 때 (하지만 사방이 막히진 않았을 때)
        if front < self.danger_threshold:
            # 좌/우 중 더 넓은 공간으로 회전합니다.
            return 'L' if left > right else 'R'

        # 3-2. 좌/우만 위험할 때 (좁은 길, 앞은 안전)
        if left < self.danger_threshold and right < self.danger_threshold:
            # 앞이 뚫려 있으므로 전진합니다.
            return 'F'
        
        # 3-3. 한쪽만 위험할 때
        if left < self.danger_threshold:
            return 'R'  # 좌측이 위험하면 우회전
        if right < self.danger_threshold:
            return 'L'  # 우측이 위험하면 좌회전

        # 4. 안전한 주행
        if front > self.safe_threshold:
            return 'F'
        else:
            # 전방이 '주의' 구간일 경우, 더 넓은 쪽으로 선회
            return 'L' if left > right else 'R'

    def auto_control_update(self):
        if not self.auto_mode:
            return

        # 자동 명령을 모터 속도로 변환
        prev_left = self.left_speed
        prev_right = self.right_speed
        
        if self.auto_command == 'F':      # 전진
            self.left_speed = 175
            self.right_speed = -175
        elif self.auto_command == 'B':    # 후진
            self.left_speed = -175
            self.right_speed = 175
        elif self.auto_command == 'L':    # 좌회전
            self.left_speed = 175
            self.right_speed = 175
        elif self.auto_command == 'R':    # 우회전
            self.left_speed = -175
            self.right_speed = -175
        elif self.auto_command == 'S':    # 정지
            self.left_speed = 0
            self.right_speed = 0

        # 속도가 변경된 경우에만 간단히 출력
        if prev_left != self.left_speed or prev_right != self.right_speed:
            direction_map = {'F': '전진', 'B': '후진', 'L': '좌회전', 'R': '우회전', 'S': '정지'}
            print(f"{direction_map.get(self.auto_command, self.auto_command)}")

        self.send_motor_command()

    def update_status_display(self):
        """최소한의 상태 표시"""
        if self.auto_mode and not self.emergency_stop:
            print(f"AUTO: {self.auto_command}")

    def run(self):
        if not self.settings:
            self.get_logger().error("터미널 설정 실패")
            return

        try:
            while True:
                key = self.get_key()

                # 모드 전환 키 변경 (단순화)
                if key == '1':
                    self.auto_mode = False
                    self.emergency_stop = False
                    self.left_speed = 0
                    self.right_speed = 0
                    print("수동 모드")
                    
                elif key == '2':
                    self.auto_mode = True
                    self.emergency_stop = False
                    print("자동 모드")
                    
                elif key == 'x':
                    self.emergency_stop = True
                    self.left_speed = 0
                    self.right_speed = 0
                    print("긴급 정지")

                elif key == '\x03':  # Ctrl+C
                    break

                # 긴급정지 상태에서는 키 입력 무시 (x로 해제)
                if self.emergency_stop and key != 'x':
                    continue

                # 수동 모드에서만 키보드 조작 허용
                if not self.auto_mode and not self.emergency_stop:
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
                    elif key == 'k':  # 가속
                        if self.left_speed > 0:
                            self.left_speed = self.clamp_speed(self.left_speed + 10)
                        elif self.left_speed < 0:
                            self.left_speed = self.clamp_speed(self.left_speed - 10)
                        if self.right_speed > 0:
                            self.right_speed = self.clamp_speed(self.right_speed + 10)
                        elif self.right_speed < 0:
                            self.right_speed = self.clamp_speed(self.right_speed - 10)
                    elif key == 'l':  # 감속
                        if self.left_speed < 0:
                            self.left_speed = self.clamp_speed(self.left_speed + 10)
                        elif self.left_speed > 0:
                            self.left_speed = self.clamp_speed(self.left_speed - 10)
                        if self.right_speed < 0:
                            self.right_speed = self.clamp_speed(self.right_speed + 10)
                        elif self.right_speed > 0:
                            self.right_speed = self.clamp_speed(self.right_speed - 10)

                # 수동 모드에서 키 입력 시 명령 전송
                if key and key != '\x03' and not self.auto_mode:
                    self.send_motor_command()
                    
                # 키 입력 시 현재 속도 표시
                if key and key != '\x03':
                    mode_indicator = "자동" if self.auto_mode else "수동"
                    emergency_indicator = " 긴급정지" if self.emergency_stop else ""
                    auto_cmd_info = f" [{self.auto_command}]" if self.auto_mode else ""
                    print(f"\r{mode_indicator} 좌측: {self.left_speed:4d}, 우측: {self.right_speed:4d}{auto_cmd_info}{emergency_indicator} | 키: {key}", end='', flush=True)

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        """종료 시 정리 작업"""
        try:
            # 모터 정지
            self.left_speed = 0
            self.right_speed = 0
            self.send_motor_command()

            # 터미널 설정 복원
            if self.settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            
            # 시리얼 연결 종료
            if self.arduino_connected and self.arduino:
                self.arduino.close()
                
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

    # ROS2 스핀을 별도 스레드로 실행
    import threading
    ros_thread = threading.Thread(target=rclpy.spin, args=(controller,))
    ros_thread.daemon = True
    ros_thread.start()

    # 메인 스레드에서 키보드 처리
    try:
        controller.run()
    except Exception as e:
        controller.get_logger().error(f"실행 중 에러: {e}")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()