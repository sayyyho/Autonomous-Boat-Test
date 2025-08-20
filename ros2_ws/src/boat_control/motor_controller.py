#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
import sys, termios, tty, select
import time

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        
        # 모터 속도 초기화 (좌측, 우측)
        self.left_speed = 0    # -255 ~ 255
        self.right_speed = 0   # -255 ~ 255
        self.speed_step = 10   # 속도 증감 단위 (더 크게)
        self.arduino = None
        self.arduino_connected = False
        
        # 터미널 설정
        try:
            self.settings = termios.tcgetattr(sys.stdin)
        except Exception as e:
            self.get_logger().error(f"터미널 설정 실패: {e}")
            self.settings = None
        
        # 아두이노 연결 시도
        possible_ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
        for port in possible_ports:
            try:
                self.arduino = serial.Serial(port, 115200, timeout=0.1)
                time.sleep(2)
                self.arduino_connected = True
                self.get_logger().info(f"🟢 아두이노 연결 성공! 포트: {port}")
                break
            except Exception as e:
                continue
        
        if not self.arduino_connected:
            self.get_logger().error("🛑 아두이노 연결 실패! 시뮬레이션 모드 전환")
            
        self.print_instructions()
        
    def print_instructions(self):
        status = "연결완료!!" if self.arduino_connected else "시뮬레이션 모드"
        print(f"""
{status} - 보트 키보드 조종
=========================
기본 조종:
w : 전진 (좌측:시계, 우측:반시계)
s : 후진 (좌측:반시계, 우측:시계)
a : 좌회전 (양쪽 반시계)
d : 우회전 (양쪽 시계)
space : 정지

좌측 모터 개별:
q : 좌측 모터 속도 +
z : 좌측 모터 속도 -

우측 모터 개별:
e : 우측 모터 속도 +
c : 우측 모터 속도 -

가속/감속:
k : 현재 방향 유지하며 10 가속
l : 현재 방향 유지하며 10 감속

r : 리셋 (양쪽 속도 0)
Ctrl+C : 종료

현재 속도 - 좌측: {self.left_speed}, 우측: {self.right_speed}
        """)

    def get_key(self):
        if not self.settings:
            return ''
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def clamp_speed(self, speed):
        return max(-255, min(255, speed))

    def send_motor_command(self):
        if not self.arduino_connected:
            self.get_logger().info(f"시뮬레이션: 좌측={self.left_speed}, 우측={self.right_speed}")
            return
            
        try:
            command = f"L{self.left_speed},R{self.right_speed}\n"
            self.arduino.write(command.encode())
            self.get_logger().info(f"모터 상태: 좌측={self.left_speed}, 우측={self.right_speed}")
            
            time.sleep(0.01)
            if self.arduino.in_waiting:
                response = self.arduino.readline().decode().strip()
                self.get_logger().info(f"📡 응답: {response}")
        except Exception as e:
            self.get_logger().error(f"통신 에러: {e}")

    def run(self):
        if not self.settings:
            self.get_logger().error("터미널 설정 실패")
            return
            
        try:
            while True:
                key = self.get_key()
                
                # 기본 조종
                if key == 'w':  
                    self.left_speed = 175  
                    self.right_speed = -175
                    
                elif key == 's':  
                    self.left_speed = -175 
                    self.right_speed = 175 
                    
                elif key == 'a':  # 좌회전 
                    self.left_speed = -175  
                    self.right_speed = -175
                    
                elif key == 'd':  # 우회전
                    self.left_speed = 175   
                    self.right_speed = 175
                    
                elif key == ' ':  # 정지
                    self.left_speed = 0
                    self.right_speed = 0
                    
                elif key == 'r':  # 리셋
                    self.left_speed = 0
                    self.right_speed = 0
                    
                # 좌측 모터 개별 제어
                elif key == 'q':  # 좌측 속도 증가
                    self.left_speed = self.clamp_speed(self.left_speed + self.speed_step)
                    
                elif key == 'z':  # 좌측 속도 감소
                    self.left_speed = self.clamp_speed(self.left_speed - self.speed_step)
                    
                # 우측 모터 개별 제어
                elif key == 'e':  # 우측 속도 증가
                    self.right_speed = self.clamp_speed(self.right_speed + self.speed_step)
                    
                elif key == 'c':  # 우측 속도 감소
                    self.right_speed = self.clamp_speed(self.right_speed - self.speed_step)
                
                elif key == 'k':  # 현재 방향 유지하며 가속
                    if self.left_speed > 0:
                        self.left_speed = self.clamp_speed(self.left_speed + 10)
                    elif self.left_speed < 0:
                        self.left_speed = self.clamp_speed(self.left_speed - 10)
                    
                    if self.right_speed > 0:
                        self.right_speed = self.clamp_speed(self.right_speed + 10)
                    elif self.right_speed < 0:
                        self.right_speed = self.clamp_speed(self.right_speed - 10)

                elif key == 'l':  # 현재 방향 유지하며 감속
                    if self.left_speed < 0:
                        self.left_speed = self.clamp_speed(self.left_speed + 10)
                    elif self.left_speed > 0:
                        self.left_speed = self.clamp_speed(self.left_speed - 10)
                    
                    if self.right_speed < 0:
                        self.right_speed = self.clamp_speed(self.right_speed + 10)
                    elif self.right_speed > 0:
                        self.right_speed = self.clamp_speed(self.right_speed - 10)
                    

                elif key == '\x03':  # Ctrl+C
                    break
                
                # 명령 전송
                if key and key != '\x03':
                    self.send_motor_command()
                    print(f"\r현재 속도 - 좌측: {self.left_speed:4d}, 우측: {self.right_speed:4d} | 키: {key}", end='', flush=True)
                    
        except KeyboardInterrupt:
            pass
        finally:
            # 종료 시 모터 정지
            self.left_speed = 0
            self.right_speed = 0
            self.send_motor_command()
            
            if self.settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            self.get_logger().info("🛑 보트 정지")

def main(args=None):
    rclpy.init(args=args)
    teleop = KeyboardTeleop()
    
    if not teleop.settings:
        teleop.destroy_node()
        rclpy.shutdown()
        return
        
    teleop.run()
    teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()