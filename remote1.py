#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import serial
import threading
from typing import Optional

# ===========================================================
# CONFIG
# ===========================================================

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
PWM_MAX = 255

DEFAULT_SPEED = 1.0   # 초기 속도 (0.5로 시작)
SPEED_STEP = 0.1      # +/- 키로 속도 조절 단위
SEND_INTERVAL = 0.02  # 50Hz

RAMP_RATE = 0.20      # 가속 속도 (초당 10 단위 → 0.1초에 1.0 도달)
DECAY_RATE = 0.12     # 브레이크 감쇠 속도
RELEASE_STOP_TIME = 0.5  # 손 떼면 0.5초에 정지
RELEASE_DECAY = SEND_INTERVAL / RELEASE_STOP_TIME 

# 환경변수 오버라이드
import os
_env_ramp = os.getenv('RAMP_RATE')
if _env_ramp:
    try:
        RAMP_RATE = float(_env_ramp)
    except Exception:
        pass
_env_decay = os.getenv('DECAY_RATE')
if _env_decay:
    try:
        DECAY_RATE = float(_env_decay)
    except Exception:
        pass
_env_release = os.getenv('RELEASE_STOP_TIME')
if _env_release:
    try:
        RELEASE_STOP_TIME = float(_env_release)
        RELEASE_DECAY = SEND_INTERVAL / RELEASE_STOP_TIME
    except Exception:
        pass

# ===========================================================
# 유틸
# ===========================================================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def priority_mixing(v: float, w: float) -> tuple:
    """선형/각속도 (v, w)를 좌우 모터 PWM 값으로 변환"""
    throttle = clamp(v, -1.0, 1.0)
    steer = clamp(w, -1.0, 1.0)
    
    left = throttle - steer
    right = throttle + steer
    
    max_val = max(abs(left), abs(right))
    if max_val > 1.0:
        left /= max_val
        right /= max_val
    
    left_pwm = int(left * PWM_MAX)
    right_pwm = int(right * PWM_MAX)
    
    return left_pwm, right_pwm

# ===========================================================
# Keyboard
# ===========================================================

try:
    import keyboard as _kb
    _USE_KEYBOARD_MODULE = True
    print("✅ keyboard 모듈 로드 성공")
except Exception as e:
    _USE_KEYBOARD_MODULE = False
    print(f"⚠️ keyboard 모듈 없음 (fallback 사용): {e}")

MOVEMENT_KEYS = ['w','a','s','d','r','f',' ']

if _USE_KEYBOARD_MODULE:
    class Keyboard:
        def __init__(self):
            self._keys = ['w','a','s','d','r','f',' ', '+', '=', '-', '_', 'm','t','p','esc']

        def get(self) -> Optional[str]:
            for k in self._keys:
                try:
                    if _kb.is_pressed(k):
                        return k
                except Exception:
                    continue
            return None

        def get_pressed_keys(self):
            pressed = []
            for k in self._keys:
                try:
                    if _kb.is_pressed(k):
                        pressed.append(k)
                except Exception:
                    continue
            return pressed

        def close(self):
            pass

        def is_any_movement_pressed(self) -> bool:
            try:
                for k in MOVEMENT_KEYS:
                    if _kb.is_pressed(k):
                        return True
            except Exception:
                return False
            return False

else:
    try:
        import termios
        import tty

        class Keyboard:
            def __init__(self):
                self.fd = sys.stdin.fileno()
                self.old = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)

            def get(self) -> Optional[str]:
                import select
                if select.select([sys.stdin], [], [], 0)[0]:
                    return sys.stdin.read(1)
                return None

            def close(self):
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

            def is_any_movement_pressed(self) -> bool:
                return False

            def get_pressed_keys(self):
                k = self.get()
                return [k] if k else []

    except ImportError:
        import msvcrt

        class Keyboard:
            def get(self):
                if msvcrt.kbhit():
                    return msvcrt.getch().decode("utf-8", errors="ignore")
                return None

            def close(self):
                pass

            def is_any_movement_pressed(self) -> bool:
                return False

            def get_pressed_keys(self):
                k = self.get()
                return [k] if k else []

# ===========================================================
# Arduino
# ===========================================================

class Arduino:
    def __init__(self, port: str = SERIAL_PORT, baud: int = BAUD_RATE):
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            print(f"✅ 시리얼 연결: {port}")
            print("⏳ 아두이노 부팅 대기...")
            time.sleep(2.5)
            while self.ser.in_waiting:
                try:
                    self.ser.readline().decode('utf-8', errors='ignore').strip()
                except:
                    pass
            print("✅ 아두이노 준비 완료")
            
        except Exception as e:
            print(f"❌ 연결 실패: {e}")
            sys.exit(1)

    def send_motion(self, l: int, r: int):
        l = clamp(l, -PWM_MAX, PWM_MAX)
        r = clamp(r, -PWM_MAX, PWM_MAX)
        pkt = f"<L:{l:+04d},R:{r:+04d}>\n"
        try:
            self.ser.write(pkt.encode('ascii'))
            self.ser.flush()
        except Exception as e:
            if not hasattr(self, '_err_count'): 
                self._err_count = 0
            self._err_count += 1
            if self._err_count % 100 == 0:
                print(f"❌ 전송 실패: {e}")

    def send_cmd(self, packet: str):
        try:
            self.ser.write(f"{packet}\n".encode('ascii'))
            self.ser.flush()
            print(f"📤 명령: {packet}")
        except Exception as e:
            print(f"❌ 명령 실패: {e}")

    def close(self):
        print("⏹️  정지...")
        self.send_motion(0, 0)
        time.sleep(0.2)
        self.ser.close()

# ===========================================================
# Remote
# ===========================================================

class Remote:
    def __init__(self):
        self.key = Keyboard()
        self.ino = Arduino()
        
        self.running = True
        self.speed = DEFAULT_SPEED
        
        self.target_v = 0.0
        self.target_w = 0.0
        self.current_v = 0.0
        self.current_w = 0.0
        
        self.last_key_time = time.time()
        self.release_input_timeout = 0.15  # 더 짧게 변경 (반응성 향상)
        self.decay_mode = None
        
        # 중복 출력 방지용
        self.last_printed_target = (0.0, 0.0)
        
        print("🟡 AUTO 모드로 시작...")
        self.ino.send_cmd("<MODE:A>")
        time.sleep(0.5)
        
        self.sender = threading.Thread(target=self._loop, daemon=True)
        self.sender.start()
        self._print_guide()

    def _loop(self):
        """가속/감쇠 계산 및 명령 전송"""
        while self.running:
            if abs(self.target_v) > 0.0 or abs(self.target_w) > 0.0:
                rate = RAMP_RATE
            else:
                if self.decay_mode == 'brake':
                    rate = DECAY_RATE
                else:
                    rate = RELEASE_DECAY
            
            # 선형 속도 업데이트
            dv = self.target_v - self.current_v
            if abs(dv) > rate:
                self.current_v += rate * (1.0 if dv > 0 else -1.0)
            else:
                self.current_v = self.target_v
            
            # 각속도 업데이트
            dw = self.target_w - self.current_w
            if abs(dw) > rate:
                self.current_w += rate * (1.0 if dw > 0 else -1.0)
            else:
                self.current_w = self.target_w
                
            # PWM 계산 및 전송
            l_pwm, r_pwm = priority_mixing(self.current_v, self.current_w)
            self.ino.send_motion(l_pwm, r_pwm)
            
            time.sleep(SEND_INTERVAL)

    def process(self, k):
        """키 입력 처리"""
        if isinstance(k, str):
            keys = [k.lower()]
        else:
            keys = [str(x).lower() for x in k if x]

        if not keys:
            return True

        # 속도 제어
        if any(x in keys for x in ['+', '=']):
            self.speed = min(1.0, self.speed + SPEED_STEP)
            self._update_target_motion()
            print(f"⚡ 속도: {self.speed:.2f}")
            self.last_key_time = time.time()
            return True

        if any(x in keys for x in ['-', '_']):
            self.speed = max(0.0, self.speed - SPEED_STEP)
            self._update_target_motion()
            print(f"🔋 속도: {self.speed:.2f}")
            self.last_key_time = time.time()
            return True

        # 모드 전환
        if 'm' in keys:
            self.ino.send_cmd("<MODE:M>")
            self.target_v = self.target_w = 0.0
            self.current_v = self.current_w = 0.0
            self.decay_mode = 'brake'
            print("🟢 MANUAL Mode")
            self.last_key_time = time.time()
            return True

        if 't' in keys:
            self.ino.send_cmd("<MODE:A>")
            print("🟡 AUTO Mode")
            self.last_key_time = time.time()
            return True

        if 'p' in keys:
            self.ino.send_cmd("<CLEAR>")
            print("✅ CLEAR")
            self.last_key_time = time.time()
            return True

        if 'esc' in keys or '\x1b' in keys:
            print("👋 종료")
            self.running = False
            return False

        # 정지
        if ' ' in keys:
            self.target_v = 0.0
            self.target_w = 0.0
            self.decay_mode = 'brake'
            print("🛑 정지")
            self.last_key_time = time.time()
            return True

        # 제자리 회전
        if 'r' in keys and 'f' not in keys and 'w' not in keys and 's' not in keys:
            self.target_v = 0.0
            self.target_w = self.speed * 1.0
            self.decay_mode = None
            self._print_target_once(f"⟲ 좌회전")
            self.last_key_time = time.time()
            return True
            
        if 'f' in keys and 'r' not in keys and 'w' not in keys and 's' not in keys:
            self.target_v = 0.0
            self.target_w = -self.speed * 1.0
            self.decay_mode = None
            self._print_target_once(f"⟳ 우회전")
            self.last_key_time = time.time()
            return True

        # 방향 제어
        forward = 'w' in keys
        backward = 's' in keys
        left = 'a' in keys
        right = 'd' in keys

        if forward and not backward:
            if left and not right:
                self.target_v = self.speed * 0.7
                self.target_w = self.speed * 0.7
                self._print_target_once("↖ 전진-좌")
            elif right and not left:
                self.target_v = self.speed * 0.7
                self.target_w = -self.speed * 0.7
                self._print_target_once("↗ 전진-우")
            else:
                self.target_v = self.speed
                self.target_w = 0.0
                self._print_target_once("↑ 전진")
                
        elif backward and not forward:
            if left and not right:
                self.target_v = -self.speed * 0.7
                self.target_w = self.speed * 0.7
                self._print_target_once("↙ 후진-좌")
            elif right and not left:
                self.target_v = -self.speed * 0.7
                self.target_w = -self.speed * 0.7
                self._print_target_once("↘ 후진-우")
            else:
                self.target_v = -self.speed
                self.target_w = 0.0
                self._print_target_once("↓ 후진")
                
        else:
            if left and not right:
                self.target_v = 0.0
                self.target_w = self.speed * 0.7
                self._print_target_once("← 좌회전")
            elif right and not left:
                self.target_v = 0.0
                self.target_w = -self.speed * 0.7
                self._print_target_once("→ 우회전")
            else:
                self.last_key_time = time.time()
                return True

        self.decay_mode = None
        self.last_key_time = time.time()
        return True

    def _print_target_once(self, msg: str):
        """목표 속도가 변경되었을 때만 출력 (중복 방지)"""
        current = (round(self.target_v, 2), round(self.target_w, 2))
        if current != self.last_printed_target:
            print(f"{msg} (V={self.target_v:.2f}, W={self.target_w:.2f})")
            self.last_printed_target = current

    def _update_target_motion(self):
        """속도(+/-) 변경 시 목표 속도 업데이트"""
        if abs(self.target_v) > 0.0:
            target_v_sign = 1.0 if self.target_v > 0 else -1.0
            if self.target_v != 0:
                target_w_ratio = self.target_w / self.target_v
            else:
                target_w_ratio = 0
            self.target_v = target_v_sign * self.speed
            self.target_w = self.target_v * target_w_ratio
            
        elif abs(self.target_w) > 0.0:
            target_w_sign = 1.0 if self.target_w > 0 else -1.0
            self.target_w = target_w_sign * self.speed
            self.target_v = 0.0
            
    def _print_guide(self):
        print("=" * 60)
        print("🚢 KABOAT 키보드 원격 제어")
        print("=" * 60)
        print("⚠️  AUTO 모드에서만 작동합니다!")
        print("W/S: 전진/후진 | A/D: 좌우회전")
        print("R/F: 제자리 좌/우 회전")
        print(f"Space: 정지 | +/-: 속도 조절 (현재: {self.speed:.2f})")
        print("T: AUTO | M: MANUAL | P: CLEAR")
        print("ESC: 종료")
        print("=" * 60)

    def run(self):
        try:
            print(f"🔍 키보드 모듈: {'keyboard' if _USE_KEYBOARD_MODULE else 'terminal'}")
            while self.running:
                now = time.time()
                pressed_keys = []
                
                try:
                    pressed_keys = self.key.get_pressed_keys()
                except Exception as e:
                    # fallback
                    k = self.key.get()
                    if k:
                        pressed_keys = [k]

                if pressed_keys:
                    if not self.process(pressed_keys):
                        break
                else:
                    # 키가 안 눌림 - 릴리스 체크
                    pressed = False
                    try:
                        pressed = self.key.is_any_movement_pressed()
                    except Exception:
                        pressed = False

                    if pressed:
                        self.last_key_time = now
                    else:
                        if now - self.last_key_time > self.release_input_timeout:
                            if (self.target_v != 0.0) or (self.target_w != 0.0):
                                self.target_v = 0.0
                                self.target_w = 0.0
                                self.decay_mode = 'release'

                time.sleep(0.005)  # 5ms 폴링
                
        except KeyboardInterrupt:
            print("\n⚠️  Ctrl+C")
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n종료 중...")
        self.running = False
        if self.sender.is_alive():
            self.sender.join(0.5)
        self.ino.close()
        self.key.close()
        print("✅ 종료 완료")

# ===========================================================
# MAIN
# ===========================================================

def main():
    Remote().run()

if __name__ == "__main__":
    main()