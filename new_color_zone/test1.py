#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan  # LiDAR만 구독
# Image, CvBridge는 사용 안 함
import subprocess
import time
import cv2  # OpenCV 직접 사용
import numpy as np

# --- GPIO 설정 (기존과 동일) ---
CHIP = 'gpiochip4'
GPIOSET_PATH = '/usr/bin/gpioset'
MOTOR_A_FRONT = 19
MOTOR_A_BACK = 26
MOTOR_B_FRONT = 21
MOTOR_B_BACK = 20


def set_motor_state(a_f, a_b, b_f, b_b, label=""):
    """모터 상태를 gpioset으로 직접 제어"""
    try:
        cmd = [GPIOSET_PATH, CHIP,
               f"{MOTOR_A_FRONT}={a_f}", f"{MOTOR_A_BACK}={a_b}",
               f"{MOTOR_B_FRONT}={b_f}", f"{MOTOR_B_BACK}={b_b}"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Motor Command: {label}")
    except subprocess.CalledProcessError as e:
        print(f"❌ GPIO 설정 실패: {e}")
    except FileNotFoundError:
        print(f"❌ '{GPIOSET_PATH}' 명령을 찾을 수 없습니다.")


def find_camera(max_index=10):
    """
    0번부터 max_index까지 카메라를 확인하고,
    '3채널 컬러 프레임'을 반환하는 첫 번째 cap 객체를 찾습니다.
    """
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # [!!] 프레임이 3채널(컬러)인지 확인
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    print(f"✅ 3채널 컬러 카메라 찾음! 인덱스 {i}번을 사용합니다.")
                    return cap  # 3채널 컬러 카메라만 반환
                else:
                    print(f"❌ 인덱스 {i}번: 1채널(흑백/IR) 카메라입니다. (무시)")
                    cap.release()
            else:
                print(f"❌ 인덱스 {i}번: 열렸으나 프레임 읽기 실패.")
                cap.release()
        else:
            print(f"❌ 인덱스 {i}번: 열기 실패.")
            cap.release()
            
    return None  # 10번까지 모두 실패


class GateNavigator(Node):
    def __init__(self):
        super().__init__('gate_navigator_hybrid')
        self.get_logger().info("✅ Gate Navigation (LiDAR Topic + Local Camera)")

        # --- LiDAR 설정 (ROS Topic 구독) ---
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',  # LiDAR 노드(sllidar_ros2)가 켜져 있어야 함
            self.scan_callback,
            10
        )
        self.lidar_threshold = 1.2
        self.segment_size = 320
        self.obstacle_detected = False
        self.obstacle_command = 'S'

        # --- 카메라 설정 (로컬 하드웨어 직접 제어) ---
        self.cap = find_camera(10)  # 0~10번 인덱스 탐색
        if self.cap is None:
            self.get_logger().error("🚨 치명적 오류: 0~10번에서 컬러 카메라를 찾지 못했습니다. 노드를 종료합니다.")
            rclpy.shutdown()  # 카메라 못 찾으면 노드 종료
            return

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_center_x = self.frame_width // 2
        self.camera_ready = True
        self.get_logger().info(f"로컬 카메라 준비 완료. Frame: {self.frame_width}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, Center: {self.frame_center_x}")

        # --- HSV 범위 (로컬 코드와 동일) ---
        self.lower_green = np.array([35, 50, 50])
        self.upper_green = np.array([85, 255, 255])
        self.lower_red1 = np.array([0, 120, 100])
        self.upper_red1 = np.array([5, 255, 255])
        self.lower_red2 = np.array([175, 120, 100])
        self.upper_red2 = np.array([179, 255, 255])
        self.min_area_green = 500
        self.min_area_red = 500

        # --- 상태 변수 ---
        self.gate_command = 'S'

        # --- 제어 루프 (모터 결정) ---
        self.timer = self.create_timer(0.05, self.control_loop)  # 0.05초마다 제어

    def scan_callback(self, msg):
        """LiDAR 콜백: 장애물 상태만 업데이트 (기존과 동일)"""
        left_segment = msg.ranges[0:self.segment_size]
        right_segment = msg.ranges[len(msg.ranges) - self.segment_size: len(msg.ranges)]

        left_valid = [r for r in left_segment if msg.range_min < r < self.lidar_threshold]
        right_valid = [r for r in right_segment if msg.range_min < r < self.lidar_threshold]
        
        left_valid_cnt = len(left_valid)
        right_valid_cnt = len(right_valid)

        if right_valid_cnt == 0 and left_valid_cnt == 0:
            self.obstacle_detected = False
        elif right_valid_cnt >= left_valid_cnt:
            self.obstacle_detected = True
            self.obstacle_command = 'HARD_L'
        else:
            self.obstacle_detected = True
            self.obstacle_command = 'HARD_R'

    # [!!] image_callback은 삭제됨 (로컬 카메라를 쓰므로)

    def find_gate_command(self, frame):
        """(로컬 코드와 동일) 카메라 영상으로 게이트 "쌍"을 찾아 조향 명령을 반환"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 마스크 생성
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        red_mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        red_mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Contours 찾기
        contours_green, _ = cv2.findContours(green_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_green_contour = max(contours_green, key=cv2.contourArea, default=None)
        largest_red_contour = max(contours_red, key=cv2.contourArea, default=None)

        green_cx = -1
        red_cx = -1

        # 초록색 객체(우) 처리
        if largest_green_contour is not None and cv2.contourArea(largest_green_contour) > self.min_area_green:
            M = cv2.moments(largest_green_contour)
            if M["m00"] != 0:
                green_cx = int(M["m10"] / M["m00"])
                x, y, w, h = cv2.boundingRect(largest_green_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, "Green (R)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 빨간색 객체(좌) 처리
        if largest_red_contour is not None and cv2.contourArea(largest_red_contour) > self.min_area_red:
            M = cv2.moments(largest_red_contour)
            if M["m00"] != 0:
                red_cx = int(M["m10"] / M["m00"])
                x, y, w, h = cv2.boundingRect(largest_red_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(frame, "Red (L)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 항로 추종 로직 (V2와 동일)
        if red_cx != -1 and green_cx != -1:
            if red_cx >= green_cx:
                cv2.putText(frame, "Error: Gate Crossed?", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                return 'S'

            gate_center_x = (green_cx + red_cx) // 2
            error = self.frame_center_x - gate_center_x
            cv2.line(frame, (gate_center_x, 240), (self.frame_center_x, 240), (255, 0, 0), 3)

            deadzone = 40
            gentle_turn_zone = 150

            if abs(error) < deadzone:
                return 'F'
            elif error > 0:
                return 'GENTLE_L' if error < gentle_turn_zone else 'HARD_L'
            else:
                return 'GENTLE_R' if abs(error) < gentle_turn_zone else 'HARD_R'
        
        elif red_cx != -1 and green_cx == -1:
            cv2.putText(frame, "Searching for Green(R)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return 'HARD_R'
        
        elif red_cx == -1 and green_cx != -1:
            cv2.putText(frame, "Searching for Red(L)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return 'HARD_L'
        else:
            cv2.putText(frame, "No Gate Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return 'S'

    def control_loop(self):
        """0.05초마다 카메라/LiDAR의 '최신 상태'를 읽어 모터 제어"""
        
        # [!!] 1. 로컬 카메라에서 프레임 읽기
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("⚠️ 로컬 카메라 프레임 읽기 실패")
            set_motor_state(0, 0, 0, 0, label="[ERROR] Camera Read Fail")
            return

        # [!!] 2. 읽은 프레임으로 게이트 명령 계산
        self.gate_command = self.find_gate_command(frame)

        # [!!] 3. 디버그 창 표시
        cv2.imshow("Gate Navigation (Hybrid)", frame)
        cv2.waitKey(1)

        final_command = 'S'
        label_prefix = ""

        # 4. 최우선: LiDAR 장애물 확인 (scan_callback에서 업데이트된 최신 값 사용)
        if self.obstacle_detected:
            final_command = self.obstacle_command  # 'HARD_L' 또는 'HARD_R'
            label_prefix = "[AVOID]"
        else:
        # 5. 차선: 게이트 추종
            final_command = self.gate_command
            label_prefix = "[GATE]"

        # 6. 모터 제어 (5-State)
        cmd = final_command.strip()
        if cmd == 'F':
            set_motor_state(1, 0, 1, 0, label=f"{label_prefix} Forward")
        elif cmd == 'GENTLE_L':
            set_motor_state(0, 0, 1, 0, label=f"{label_prefix} Gentle Left")
        elif cmd == 'GENTLE_R':
            set_motor_state(1, 0, 0, 0, label=f"{label_prefix} Gentle Right")
        elif cmd == 'HARD_L':
            set_motor_state(0, 1, 1, 0, label=f"{label_prefix} Hard Left")
        elif cmd == 'HARD_R':
            set_motor_state(1, 0, 0, 1, label=f"{label_prefix} Hard Right")
        else:  # 'S'
            set_motor_state(0, 0, 0, 0, label=f"{label_prefix} Stop")
            
    def cleanup(self):
        """노드 종료 시 호출될 정리 함수"""
        self.get_logger().info("🛑 노드 종료... 모터 정지 및 카메라/창 해제")
        set_motor_state(0, 0, 0, 0, label="Shutdown Stop")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = GateNavigator()
    
    if rclpy.ok(): # 카메라 초기화 실패 시 node가 생성되지 않을 수 있음
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            # 종료 시 모터 정지 및 자원 해제
            node.cleanup()
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("🚨 카메라 초기화 실패로 ROS 2 노드를 시작하지 못했습니다.")


if __name__ == '__main__':
    main()