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

# --- 제어 파라미터 (Control Parameters) ---
DEFAULT_SPEED = 1.0   # 최대 목표 선형 속도 (정규화된 값: 0.0 ~ 1.0)
SPEED_STEP = 0.1      # +/- 키로 속도 조절 단위

SEND_INTERVAL = 0.02  # 모터 명령 전송 주기 [s] (50Hz로 변경: 더 부드러운 제어)

# RAMP/DECAY: 원래는 루프당 절대값(step)으로 사용됩니다.
# - rate 값은 한 루프(=SEND_INTERVAL)에서 current_v/current_w에 더해지는 값입니다.
# - 예: SEND_INTERVAL=0.02, RAMP_RATE=0.05 -> 초당 약 2.5 단위 (1.0 도달에 0.4s)
# 권장 프리셋 (사용자 요청 기준, 반응성 향상):
#  - Smooth: 0.04  (부드럽고 느림)
#  - Responsive: 0.12  (권장 기본값 — 반응성 좋음)
#  - Aggressive: 0.2  (매우 빠름)

RAMP_RATE = 0.12      # 권장 기본값: 0.12 (더 빠른 가속 — 필요시 환경변수로 오버라이드 가능)
DECAY_RATE = 0.20     # space 등 강제 브레이크용(빠른 감쇠)

# 릴리스(손 때면)용 감쇠: 사용자가 손을 떼면 이 값으로 천천히 감속하여
# 약 `RELEASE_STOP_TIME` 초에 목표 0에 도달하게 합니다.
RELEASE_STOP_TIME = 1.0  # 손을 떼면 1초 정도에 자연스럽게 정지
RELEASE_DECAY = SEND_INTERVAL / RELEASE_STOP_TIME

# 환경변수로 오버라이드 허용 (편하게 실험 가능)
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
# INPUT_TIMEOUT 로직을 제거하여 키 연속 누름을 보장함

# ===========================================================
# 유틸 (Priority Mixing)
# ===========================================================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def priority_mixing(v: float, w: float) -> tuple:
    """선형/각속도 (v, w)를 좌우 모터 PWM 값으로 변환 (Arcade Drive + Priority Normalization)"""
    throttle = clamp(v, -1.0, 1.0)
    steer = clamp(w, -1.0, 1.0)
    
    # 믹싱 (ROS 규약: +w는 좌회전 -> 왼쪽 감속, 오른쪽 가속)
    left = throttle - steer
    right = throttle + steer
    
    # 정규화 (최대 PWM 255를 넘지 않도록 비율 유지하며 스케일링)
    max_val = max(abs(left), abs(right))
    if max_val > 1.0:
        left /= max_val
        right /= max_val
    
    # PWM 값으로 변환
    left_pwm = int(left * PWM_MAX)
    right_pwm = int(right * PWM_MAX)
    
    return left_pwm, right_pwm

# ===========================================================
# Keyboard (생략. 이전 코드와 동일)
# ===========================================================

# Keyboard input: prefer the `keyboard` package (detects holds reliably on Linux).
try:
    import keyboard as _kb  # optional dependency; on Linux may require sudo
    _USE_KEYBOARD_MODULE = True
except Exception:
    _USE_KEYBOARD_MODULE = False

MOVEMENT_KEYS = ['w','a','s','d','r','f',' ']  # movement keys we care about

if _USE_KEYBOARD_MODULE:
    class Keyboard:
        """Keyboard using the `keyboard` package to detect holds continuously.

        Note: On many Linux systems this requires root privileges (sudo) or
        appropriate udev permissions to read global key state.
        """
        def __init__(self):
            # keys we care about; lookup done in get()
            self._keys = ['w','a','s','d','r','f',' ', '+', '=', '-', '_', 'm','t','p','\x1b']

        def get(self) -> Optional[str]:
            # return the first pressed key we detect (continuous while held)
            for k in self._keys:
                try:
                    if _kb.is_pressed(k):
                        return k
                except Exception:
                    # ignore driver errors and continue
                    continue
            return None

        def get_pressed_keys(self):
            """Return list of all currently pressed keys from the interest set."""
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
    # Fallback: terminal-based reader (original behavior)
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
                # Terminal fallback cannot reliably report hold state;
                # return False to force timeout-based detection in Remote.run
                return False

            def get_pressed_keys(self):
                # Fallback: cannot detect simultaneous keys; return single key if available
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
# Arduino (생략. 이전 코드와 동일)
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
            if not hasattr(self, '_err_count'): self._err_count = 0
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
# Remote (수정된 제어 로직)
# ===========================================================

class Remote:
    def __init__(self):
        self.key = Keyboard()
        self.ino = Arduino()
        
        self.running = True
        self.speed = DEFAULT_SPEED
        
        # 목표 속도 (키 입력에 의해 즉시 변경됨)
        self.target_v = 0.0
        self.target_w = 0.0
        
        # 현재 속도 (_loop에서 점진적으로 변경됨)
        self.current_v = 0.0
        self.current_w = 0.0
        
        # ✅ 쓰레드 시작 전에 릴리스 관련 속성 먼저 초기화
        self.last_key_time = time.time()
        self.release_input_timeout = 1.0
        self.decay_mode = None  # None | 'release' | 'brake'
        
        print("🟡 AUTO 모드로 시작...")
        self.ino.send_cmd("<MODE:A>")
        time.sleep(0.5)
        
        # ✅ 모든 속성 초기화 후 쓰레드 시작
        self.sender = threading.Thread(target=self._loop, daemon=True)
        self.sender.start()
        self._print_guide()

    def _loop(self):
        """가속/감쇠 계산 및 명령 전송"""
        while self.running:
            
            # 1. 현재 속도 업데이트 (가속/감속)
            # 목표 속도(target_v, target_w)를 향해 현재 속도(current_v, current_w)를 변경
            # 목표가 0일 때는 decay_mode에 따라 자연 릴리스 감쇠 또는 브레이크 감쇠를 사용
            if abs(self.target_v) > 0.0 or abs(self.target_w) > 0.0:
                rate = RAMP_RATE
            else:
                if self.decay_mode == 'brake':
                    rate = DECAY_RATE
                else:
                    rate = RELEASE_DECAY
            
            # 선형 속도 (v) 업데이트
            dv = self.target_v - self.current_v
            if abs(dv) > rate:
                self.current_v += rate * (1.0 if dv > 0 else -1.0)
            else:
                self.current_v = self.target_v
            
            # 각속도 (w) 업데이트
            dw = self.target_w - self.current_w
            if abs(dw) > rate:
                self.current_w += rate * (1.0 if dw > 0 else -1.0)
            else:
                self.current_w = self.target_w
                
            # 2. PWM 계산 및 전송
            l_pwm, r_pwm = priority_mixing(self.current_v, self.current_w)
            self.ino.send_motion(l_pwm, r_pwm)
            
            time.sleep(SEND_INTERVAL)


    def process(self, k):
        """키 입력 처리. 인자로는 문자열 키 또는 눌린 키들의 리스트를 받을 수 있음.
        Returns False to indicate exit.
        """
        # normalize to list
        if isinstance(k, str):
            keys = [k.lower()]
        else:
            # assume iterable of keys
            keys = [str(x).lower() for x in k if x]

        # Control keys (speed/mode) — handle first
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

        if 'm' in keys:
            self.ino.send_cmd("<MODE:M>")
            self.target_v = self.target_w = 0.0
            self.current_v = self.current_w = 0.0
            self.decay_mode = 'brake'
            print("🟢 MANUAL (IDLE) Mode")
            self.last_key_time = time.time()
            return True

        if 't' in keys:
            self.ino.send_cmd("<MODE:A>")
            print("🟡 AUTO Mode")
            self.last_key_time = time.time()
            return True

        if 'p' in keys:
            self.ino.send_cmd("<CLEAR>")
            print("✅ CLEAR Triggered")
            self.last_key_time = time.time()
            return True

        if '\x1b' in keys:
            print("👋 종료")
            self.running = False
            return False

        # ===========================
        # 방향 제어
        # ===========================
        # Combine keys to compute target_v and target_w
        # Prioritize in-place rotation keys (r/f) if present alone
        if ' ' in keys:
            # explicit stop (brake)
            self.target_v = 0.0
            self.target_w = 0.0
            self.decay_mode = 'brake'
            print("🛑 정지 (감쇠 시작)")
            self.last_key_time = time.time()
            return True

        # in-place rotation (take precedence)
        if 'r' in keys and 'f' not in keys and 'w' not in keys and 's' not in keys:
            self.target_v = 0.0
            self.target_w = self.speed * 1.0
            self.decay_mode = None
            print(f"⟲ 제자리 좌회전 (Target W={self.target_w:.2f})")
            self.last_key_time = time.time()
            return True
        if 'f' in keys and 'r' not in keys and 'w' not in keys and 's' not in keys:
            self.target_v = 0.0
            self.target_w = -self.speed * 1.0
            self.decay_mode = None
            print(f"⟳ 제자리 우회전 (Target W={self.target_w:.2f})")
            self.last_key_time = time.time()
            return True

        # Determine forward/backward presence
        forward = 'w' in keys
        backward = 's' in keys
        left = 'a' in keys
        right = 'd' in keys

        # Combined movement logic:
        if forward and not backward:
            if left and not right:
                # forward-left
                self.target_v = self.speed * 0.7
                self.target_w = self.speed * 0.7
            elif right and not left:
                # forward-right
                self.target_v = self.speed * 0.7
                self.target_w = -self.speed * 0.7
            else:
                # forward
                self.target_v = self.speed
                self.target_w = 0.0
        elif backward and not forward:
            if left and not right:
                # backward-left
                self.target_v = -self.speed * 0.7
                self.target_w = self.speed * 0.7
            elif right and not left:
                # backward-right
                self.target_v = -self.speed * 0.7
                self.target_w = -self.speed * 0.7
            else:
                # backward
                self.target_v = -self.speed
                self.target_w = 0.0
        else:
            # no forward/backward primary: if only left/right pressed, do gentle turn in place
            if left and not right:
                self.target_v = 0.0
                self.target_w = self.speed * 0.7
            elif right and not left:
                self.target_v = 0.0
                self.target_w = -self.speed * 0.7
            else:
                # no movement keys
                # do not change targets here; release logic handles stopping
                self.last_key_time = time.time()
                return True

        # when any movement key(s) are processed, ensure decay mode cleared
        self.decay_mode = None
        self.last_key_time = time.time()
        # Print a concise status (avoid flooding by printing only on change)
        print(f"Target updated: V={self.target_v:.2f} W={self.target_w:.2f}")
        return True

        # control keys handled below

    def _update_target_motion(self):
        """속도(+/-) 변경 시 현재 키에 할당된 목표 속도 값에 새 self.speed를 반영"""
        # 현재 목표 v, w의 비율과 부호를 유지하며 self.speed를 반영
        
        # 선형 속도의 크기(절대값)를 기준으로 비율 계산
        if abs(self.target_v) > 0.0:
            target_v_sign = 1.0 if self.target_v > 0 else -1.0
            target_w_ratio = self.target_w / self.target_v
            
            # 새로운 목표 속도 적용
            self.target_v = target_v_sign * self.speed
            self.target_w = self.target_v * target_w_ratio
            
        elif abs(self.target_w) > 0.0: # 제자리 회전 중인 경우
            target_w_sign = 1.0 if self.target_w > 0 else -1.0
            self.target_w = target_w_sign * self.speed
            self.target_v = 0.0
            
    def _print_guide(self):
        print("=" * 60)
        print("🚢 KABOAT 키보드 원격 제어 (연속 입력 보장)")
        print("=" * 60)
        print("⚠️  AUTO 모드에서만 작동합니다!")
        print("W/S: 전진/후진 | A/D: 좌우회전")
        print("R/F: 제자리 좌/우 회전")
        print(f"Space: 정지 (감쇠) | +/-: 속도 (MAX V={self.speed:.2f})")  # ✅ f-string으로 수정
        print("T: AUTO | M: MANUAL | P: CLEAR")
        print("ESC: 종료")
        print("=" * 60)
        print(f"현재 최대 목표 속도: {self.speed:.2f}")
        print("=" * 60)

    def run(self):
        try:
            while self.running:
                # Prefer asking for all pressed keys when available
                now = time.time()
                pressed_keys = []
                try:
                    pressed_keys = self.key.get_pressed_keys()
                except Exception:
                    # fallback to single-key read
                    k = self.key.get()
                    if k:
                        pressed_keys = [k]

                if pressed_keys:
                    # Process combined keys (e.g., ['w','a'])
                    if not self.process(pressed_keys):
                        break
                    # last_key_time updated inside process()
                else:
                    # No keys currently detected
                    # If keyboard module can tell us about holds, use it
                    pressed = False
                    try:
                        pressed = self.key.is_any_movement_pressed()
                    except Exception:
                        pressed = False

                    if pressed:
                        self.last_key_time = now
                    else:
                        # If no key and timeout elapsed, consider it a release
                        if now - self.last_key_time > self.release_input_timeout:
                            if (self.target_v != 0.0) or (self.target_w != 0.0):
                                self.target_v = 0.0
                                self.target_w = 0.0
                                self.decay_mode = 'release'

                # _loop 스레드가 제어를 담당하므로, 메인 루프는 가볍게 유지
                time.sleep(0.001)
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
        print("✅ 종료")

# ===========================================================
# MAIN
# ===========================================================

def main():
    Remote().run()

if __name__ == "__main__":
    main()