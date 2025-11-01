#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import subprocess
import time

# --- GPIO 설정 ---
CHIP = 'gpiochip4'
GPIOSET_PATH = '/usr/bin/gpioset'

# BCM 핀 번호 (보드 배선에 맞게 조정)
MOTOR_A_FRONT = 19  # 왼쪽 모터 전진
MOTOR_A_BACK  = 26  # 왼쪽 모터 후진
MOTOR_B_FRONT = 21  # 오른쪽 모터 전진
MOTOR_B_BACK  = 20  # 오른쪽 모터 후진

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


class LaserScanDriver(Node):
    def __init__(self):
        super().__init__('laser_scan_driver')

        self.get_logger().info("✅ GPIO 기반 LaserScan Driver 시작")
        self.left_message = ""
        self.right_message = ""

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.threshold = 1.2  # 장애물 거리 기준 (미터)
        self.current_command = 'F'  # 초기 명령
        self.segment_size = 320
        self.range_count = 3200

        # 0.05초마다 명령 반복 실행
        self.timer = self.create_timer(0.05, self.send_command)

    def scan_callback(self, msg):
        left_segment = msg.ranges[0:self.segment_size]
        right_segment = msg.ranges[len(msg.ranges) - self.segment_size : len(msg.ranges)]

        self.left_message = f"Left segment sample: {left_segment[::30]}"
        self.right_message = f"Right segment sample: {right_segment[::-1][::30]}"

        # 유효한 값만 필터링
        left_valid = [r for r in left_segment if msg.range_min < r < self.threshold]
        right_valid = [r for r in right_segment if msg.range_min < r < self.threshold]

        left_valid_cnt = len(left_valid)
        right_valid_cnt = len(right_valid)

        if right_valid_cnt == left_valid_cnt == 0:
            self.current_command = 'F'  # 전진
        elif right_valid_cnt >= left_valid_cnt:
            self.current_command = 'L'  # 좌회전
        elif right_valid_cnt < left_valid_cnt:
            self.current_command = 'R'  # 우회전
        # else: 필요시 정지 로직 추가 가능

    def send_command(self):
        cmd = self.current_command.strip()
        if cmd == 'F':
            set_motor_state(1, 0, 1, 0, label="Forward")
        elif cmd == 'L':
            set_motor_state(0, 1, 1, 0, label="Turn Left")
        elif cmd == 'R':
            set_motor_state(1, 0, 0, 1, label="Turn Right")
        elif cmd == 'S':
            set_motor_state(0, 0, 0, 0, label="Stop")
        else:
            set_motor_state(0, 0, 0, 0, label="Idle/Unknown")


def main(args=None):
    rclpy.init(args=args)
    node = LaserScanDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 키보드 인터럽트로 종료")
    finally:
        set_motor_state(0, 0, 0, 0, label="Shutdown Stop")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()