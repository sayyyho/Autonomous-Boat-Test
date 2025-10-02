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
        self.slalom_mode = False  # 슬라럼 모드 추가

        # 보트 크기 및 안전 여유 설정
        self.boat_length = 1.0           # 보트 세로 길이 (m)
        self.boat_width = 1            # 보트 가로 폭 (m)
        self.safety_margin = 1         # 안전 여유 공간 (m)
        
        # 라이다 회피 파라미터 (보트 크기 기반)
        self.danger_threshold = self.boat_width/2 + self.safety_margin  # 1.05m (35cm + 70cm)
        self.safe_threshold = self.danger_threshold + 0.5               # 1.55m
        self.emergency_threshold = self.boat_width/2 + 0.1              # 0.45m (35cm + 10cm)
        self.front_angle = 40            # 전방 감지 각도 (±30도)
        self.side_angle = 90             # 좌우 감지 각도
        
        # 슬라럼 모드 설정
        self.slalom_state = "APPROACH"  # APPROACH, TURNING_LEFT, TURNING_RIGHT
        self.last_turn_direction = None
        self.turn_completion_threshold = 0.8  # 회전 완료 기준 시간
        self.turn_start_time = None
        
        # 슬라럼 파라미터
        self.slalom_detection_distance = 2.0    # 꼬깔 감지 거리
        self.slalom_turn_distance = 1.3         # 회전 시작 거리
        self.slalom_clear_distance = 1.8        # 통과 완료 거리
        
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
            self.enhanced_scan_callback,  # 향상된 스캔 콜백 사용
            10
        )

        # 자동 모드용 타이머
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
        mode = "슬라럼모드" if self.slalom_mode else ("자동모드" if self.auto_mode else "수동모드")
        print(f"""
{status} - 하이브리드 보트 제어 시스템 (슬라럼 지원)
========================================
현재 모드: {mode}

보트 크기: {self.boat_length}m x {self.boat_width}m (세로 x 가로)
안전 여유: {self.safety_margin}m

모드 전환:
1 : 수동 모드 (키보드 조종)
2 : 자동 모드 (직진 최우선)
3 : 슬라럼 모드 (S자 코스)
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
긴급정지: {self.emergency_threshold:.2f}m
위험거리: {self.danger_threshold:.2f}m  
안전거리: {self.safe_threshold:.2f}m

=== 슬라럼 모드 ===
꼬깔콘 사이 S자 항해
감지거리: {self.slalom_detection_distance:.1f}m
회전거리: {self.slalom_turn_distance:.1f}m
통과거리: {self.slalom_clear_distance:.1f}m

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

    def apply_noise_filter(self, ranges):
        """라이다 노이즈 필터링"""
        # 중앙값 필터 적용 (3점 윈도우)
        filtered = np.copy(ranges)
        for i in range(1, len(ranges)-1):
            window = ranges[i-1:i+2]
            filtered[i] = np.median(window)
        return filtered

    def get_sector_distances(self, ranges, sector):
        """섹터별 거리 데이터 추출 (더 정밀하게)"""
        total_points = len(ranges)
        
        if sector == "FRONT":
            # 전방 ±30도 영역
            angle_range = 30
            front_start = max(0, total_points - angle_range)
            front_end = min(total_points, angle_range)
            return np.concatenate([ranges[0:front_end], ranges[front_start:]])
        
        elif sector == "LEFT":
            # 좌측 30-120도 영역  
            left_start = min(total_points - 1, 30)
            left_end = min(total_points, 120)
            return ranges[left_start:left_end]
        
        elif sector == "RIGHT":
            # 우측 240-330도 영역
            right_start = max(0, total_points - 120)
            right_end = max(0, total_points - 30)
            return ranges[right_start:right_end]
        
        return np.array([10.0])

    def calculate_representative_distance(self, distances):
        """영역의 대표 거리 계산 (최소값과 평균의 가중합)"""
        if len(distances) == 0:
            return 10.0
        
        min_dist = np.min(distances)
        avg_dist = np.mean(distances)
        
        # 최소값에 더 큰 가중치 (안전 우선)
        representative = min_dist * 0.7 + avg_dist * 0.3
        return representative

    def enhanced_scan_callback(self, msg):
        """향상된 라이다 데이터 처리"""
        if not self.auto_mode:
            return

        try:
            ranges = np.array(msg.ranges)
            ranges = np.where(np.isinf(ranges), 10.0, ranges)
            ranges = np.where(np.isnan(ranges), 10.0, ranges)
            ranges = np.where(ranges <= 0, 10.0, ranges)
            
            # 노이즈 필터링 (급격한 변화 제거)
            ranges = self.apply_noise_filter(ranges)
            
        except Exception as e:
            self.get_logger().error(f"라이다 데이터 처리 에러: {e}")
            return
        
        total_points = len(ranges)
        if total_points == 0:
            return
        
        # 더 정밀한 영역 분할
        front_distances = self.get_sector_distances(ranges, "FRONT")
        left_distances = self.get_sector_distances(ranges, "LEFT") 
        right_distances = self.get_sector_distances(ranges, "RIGHT")
        
        # 각 영역의 대표값 계산 (최소값 + 평균의 가중합)
        front_min = self.calculate_representative_distance(front_distances)
        left_min = self.calculate_representative_distance(left_distances)
        right_min = self.calculate_representative_distance(right_distances)
        
        # 향상된 의사결정
        new_command = self.decide_auto_movement(front_min, left_min, right_min)
        decision_reason = self.get_decision_reason(front_min, left_min, right_min, new_command)
        
        # 명령 즉시 적용
        self.auto_command = new_command
        
        # 명령 변경 시에만 출력
        if new_command != self.previous_auto_command:
            direction_map = {'F': '직진', 'B': '후진', 'L': '좌회전', 'R': '우회전', 'S': '정지'}
            print(f"{direction_map.get(new_command, new_command)} | {decision_reason}")
            self.previous_auto_command = new_command

    def detect_slalom_cone(self, front, left, right):
        """꼬깔콘 패턴 감지"""
        
        # 전방에 장애물이 있고, 좌우 중 한쪽이 뚫려있으면 꼬깔콘으로 판단
        cone_detected = (front < self.slalom_detection_distance and 
                        (left > self.slalom_clear_distance or right > self.slalom_clear_distance))
        
        if cone_detected:
            # 더 넓은 쪽으로 회전할 방향 결정
            if left > right:
                preferred_direction = "LEFT"
            else:
                preferred_direction = "RIGHT"
                
            # 교대 패턴 강제 (이전과 다른 방향 선호)
            if self.last_turn_direction == "LEFT":
                if right > self.danger_threshold:  # 우측이 최소한 안전하면
                    preferred_direction = "RIGHT"
            elif self.last_turn_direction == "RIGHT":
                if left > self.danger_threshold:   # 좌측이 최소한 안전하면
                    preferred_direction = "LEFT"
                    
            return True, preferred_direction
        
        return False, None

    def decide_slalom_movement(self, front, left, right):
        """슬라럼 전용 항해 로직"""
        
        # 1. 꼬깔콘 감지
        cone_detected, turn_direction = self.detect_slalom_cone(front, left, right)
        
        if not cone_detected:
            # 꼬깔콘이 없으면 직진
            self.slalom_state = "APPROACH"
            return 'F'
        
        # 2. 회전 상태 관리
        current_time = time.time()
        
        if self.slalom_state == "APPROACH":
            # 회전 시작
            if front < self.slalom_turn_distance:
                if turn_direction == "LEFT":
                    self.slalom_state = "TURNING_LEFT"
                    self.last_turn_direction = "LEFT"
                    self.turn_start_time = current_time
                    return 'L'
                else:
                    self.slalom_state = "TURNING_RIGHT" 
                    self.last_turn_direction = "RIGHT"
                    self.turn_start_time = current_time
                    return 'R'
            else:
                return 'F'  # 아직 접근 중
                
        elif self.slalom_state == "TURNING_LEFT":
            # 좌회전 지속 조건 체크
            if (current_time - self.turn_start_time > self.turn_completion_threshold and 
                front > self.slalom_clear_distance):
                # 회전 완료, 직진으로 전환
                self.slalom_state = "APPROACH"
                return 'F'
            else:
                return 'L'  # 계속 좌회전
                
        elif self.slalom_state == "TURNING_RIGHT":
            # 우회전 지속 조건 체크
            if (current_time - self.turn_start_time > self.turn_completion_threshold and 
                front > self.slalom_clear_distance):
                # 회전 완료, 직진으로 전환
                self.slalom_state = "APPROACH"
                return 'F'
            else:
                return 'R'  # 계속 우회전
        
        return 'F'  # 기본값

    def check_forward_clearance(self, front_distance):
        """
        전방 통행 가능성을 보트 크기 기준으로 분석
        """
        # 매우 안전한 거리 (안전거리 이상)
        if front_distance > self.safe_threshold:  # 1.55m 이상
            return "CLEAR"
        
        # 충분한 거리 (위험거리 + 여유)
        elif front_distance > self.danger_threshold + 0.3:  # 1.35m 이상
            return "CLEAR"
        
        # 주의하지만 통과 가능 (위험거리 + 약간 여유)
        elif front_distance > self.danger_threshold + 0.1:  # 1.15m 이상
            return "TIGHT_BUT_PASSABLE"
        
        # 위험 - 회피 필요 (위험거리 미달)
        else:
            return "BLOCKED"

    def analyze_turning_space(self, distance, direction):
        """
        회전 공간의 충분함을 보트 크기 기준으로 평가
        """
        # 보트 폭의 절반 + 안전 여유를 기준으로 평가
        min_required = self.boat_width/2 + 0.2  # 최소 필요 공간: 55cm
        
        if distance > self.safe_threshold:  # 1.55m 이상
            return 100  # 매우 넓음
        elif distance > self.danger_threshold:  # 1.05m 이상 
            return 80   # 넓음
        elif distance > self.danger_threshold * 0.8:  # 0.84m 이상
            return 60   # 보통
        elif distance > min_required:  # 0.55m 이상
            return 30   # 좁지만 회전 가능
        else:
            return 0    # 막힘

    def calculate_avoidance_strategy(self, front, left, right):
        """
        직진이 불가능할 때의 스마트 회피 전략
        """
        
        # 좌우 공간 분석
        left_space = self.analyze_turning_space(left, "LEFT")
        right_space = self.analyze_turning_space(right, "RIGHT")
        
        # 양쪽 모두 막힌 경우
        if left < self.danger_threshold and right < self.danger_threshold:
            if front > self.danger_threshold:
                # 좁은 통로지만 앞이 상대적으로 안전 → 조심스럽게 직진
                return 'F'
            else:
                return "TRAPPED"
        
        # 한쪽만 막힌 경우 - 간단한 회피
        if left < self.danger_threshold and right >= self.danger_threshold:
            return 'R'  # 우회전
        if right < self.danger_threshold and left >= self.danger_threshold:
            return 'L'  # 좌회전
        
        # 양쪽 모두 어느 정도 안전한 경우 - 더 넓은 쪽 선택
        if left_space > right_space:
            return 'L'
        elif right_space > left_space:
            return 'R'
        else:
            # 동일하면 좌회전 (일관성)
            return 'L'

    def escape_dead_end(self, front, left, right):
        """
        막다른 길 탈출 전략
        """
        # 가장 넓은 방향 찾기
        max_distance = max(front, left, right)
        
        if max_distance == front and front > self.emergency_threshold:
            return 'F'  # 앞이 가장 넓으면 직진
        elif max_distance == left:
            return 'L'  # 좌측이 가장 넓으면 좌회전
        elif max_distance == right:
            return 'R'  # 우측이 가장 넓으면 우회전
        else:
            return 'B'  # 모든 방향이 막혔으면 후진

    def decide_auto_movement(self, front, left, right):
        """
        향상된 의사결정 - 슬라럼 모드 포함
        """
        
        # 긴급정지는 모든 모드에서 최우선
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
        
        # 슬라럼 모드 활성화 시 전용 로직 사용
        if self.slalom_mode:
            return self.decide_slalom_movement(front, left, right)
        
        # 기존 직진 우선 로직
        else:
            front_clearance = self.check_forward_clearance(front)
            
            if front_clearance == "CLEAR":
                return 'F'
            elif front_clearance == "TIGHT_BUT_PASSABLE":
                return 'F'
            
            avoidance_strategy = self.calculate_avoidance_strategy(front, left, right)
            
            if avoidance_strategy == "TRAPPED":
                return self.escape_dead_end(front, left, right)
            
            return avoidance_strategy

    def get_decision_reason(self, front, left, right, command):
        """슬라럼 상태 포함한 의사결정 이유"""
        
        if front < self.emergency_threshold or left < self.emergency_threshold or right < self.emergency_threshold:
            min_dist = min(front, left, right)
            return f"긴급정지 - 충돌 위험({min_dist:.2f}m < {self.emergency_threshold:.2f}m)"

        if self.slalom_mode:
            cone_detected, turn_dir = self.detect_slalom_cone(front, left, right)
            if cone_detected:
                return f"슬라럼: {self.slalom_state} | 꼬깔 감지, {turn_dir} 회전 예정"
            else:
                return f"슬라럼: {self.slalom_state} | 직진 중"

        front_clearance = self.check_forward_clearance(front)
        
        if command == 'F':
            if front_clearance == "CLEAR":
                return f"직진 유지 - 전방 안전({front:.2f}m)"
            elif front_clearance == "TIGHT_BUT_PASSABLE":
                return f"조심스럽게 직진 - 좁지만 통과 가능({front:.2f}m)"
            else:
                return f"직진 시도 - 좁은 통로 통과({front:.2f}m)"
        
        elif command == 'L':
            if right < self.danger_threshold and left >= self.danger_threshold:
                return f"우측 차단 회피 - 좌회전({right:.2f}m < {self.danger_threshold:.2f}m)"
            else:
                left_score = self.analyze_turning_space(left, "LEFT")
                right_score = self.analyze_turning_space(right, "RIGHT")
                return f"최적 경로 선택 - 좌측이 더 넓음(L:{left_score} vs R:{right_score})"
        
        elif command == 'R':
            if left < self.danger_threshold and right >= self.danger_threshold:
                return f"좌측 차단 회피 - 우회전({left:.2f}m < {self.danger_threshold:.2f}m)"
            else:
                left_score = self.analyze_turning_space(left, "LEFT")
                right_score = self.analyze_turning_space(right, "RIGHT")
                return f"최적 경로 선택 - 우측이 더 넓음(R:{right_score} vs L:{left_score})"
        
        elif command == 'B':
            return f"막다른 길 탈출 - 후진으로 공간 확보"
        
        elif command == 'S':
            return f"안전 확보 - 잠시 정지 후 상황 재평가"
        
        return f"기본 전략 - {command}"

    def auto_control_update(self):
        if not self.auto_mode:
            return

        # 자동 명령을 모터 속도로 변환
        prev_left = self.left_speed
        prev_right = self.right_speed
        
        if self.auto_command == 'F':      # 전진
            self.left_speed = 190
            self.right_speed = -190
        elif self.auto_command == 'B':    # 후진
            self.left_speed = -190
            self.right_speed = 190
        elif self.auto_command == 'L':    # 좌회전
            self.left_speed = 190
            self.right_speed = 190
        elif self.auto_command == 'R':    # 우회전
            self.left_speed = -190
            self.right_speed = -190
        elif self.auto_command == 'S':    # 정지
            self.left_speed = 0
            self.right_speed = 0

        self.send_motor_command()

    def update_status_display(self):
        """최소한의 상태 표시"""
        if self.auto_mode and not self.emergency_stop:
            mode_info = f"SLALOM:{self.slalom_state}" if self.slalom_mode else "AUTO"
            print(f"{mode_info}: {self.auto_command}")

    def run(self):
        if not self.settings:
            self.get_logger().error("터미널 설정 실패")
            return

        try:
            while True:
                key = self.get_key()

                # 모드 전환 키
                if key == '1':
                    self.auto_mode = False
                    self.slalom_mode = False
                    self.emergency_stop = False
                    self.left_speed = 0
                    self.right_speed = 0
                    print("수동 모드")
                    
                elif key == '2':
                    self.auto_mode = True
                    self.slalom_mode = False
                    self.emergency_stop = False
                    print("자동 모드 (직진 최우선)")
                    
                elif key == '3':  # 슬라럼 모드 추가
                    self.auto_mode = True
                    self.slalom_mode = True
                    self.emergency_stop = False
                    self.slalom_state = "APPROACH"  # 상태 초기화
                    self.last_turn_direction = None
                    print("슬라럼 모드 (S자 항해)")
                    
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
                    mode_indicator = "슬라럼" if self.slalom_mode else ("자동" if self.auto_mode else "수동")
                    emergency_indicator = " 긴급정지" if self.emergency_stop else ""
                    auto_cmd_info = f" [{self.auto_command}]" if self.auto_mode else ""
                    slalom_info = f" {self.slalom_state}" if self.slalom_mode else ""
                    print(f"\r{mode_indicator} 좌측: {self.left_speed:4d}, 우측: {self.right_speed:4d}{auto_cmd_info}{slalom_info}{emergency_indicator} | 키: {key}", end='', flush=True)

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