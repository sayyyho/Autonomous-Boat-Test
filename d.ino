// /*
//  * Final Integration Code v3 (Power order & 3-color lamp fixed)
//  * 통합: Remote Control + ROS(AUTO) + Power/Safety + 3색 LED 램프
//  *
//  * [기능 요약]
//  * 1. Safety:
//  *    - 물리 E-STOP 버튼 (Pin 2, Active-Low)
//  *    - RC 조종기 CH6 (Pin 26, High = Kill)
//  *    - 원격 <KILL>/<CLEAR> 명령
//  *    → Kill시 릴레이 순차 OFF (Motor -> LiDAR -> NUC)
//  *
//  * 2. Power ON:
//  *    → NUC -> Motor -> LiDAR 순서로 전원 인가
//  *
//  * 3. Motor:
//  *    - MDD10A 드라이버 제어 (DIR/PWM)
//  *    - 램핑(Ramping)으로 부드러운 가감속
//  *
//  * 4. Control:
//  *    - AUTO 모드:
//  *        ROS 노드가 <L:+ddd,R:-ddd>\n 패킷 전송
//  *        & <MODE:A> 명령으로 자율 모드 요청
//  *    - MANUAL 모드:
//  *        RC 조종기 CH1/CH2로 스로틀+조향
//  *        (조종기 움직이면 자동으로 MANUAL 우선)
//  *    - EMERGENCY 모드:
//  *        Kill 조건 발생 시 진입, 전원 OFF + 모터 정지
//  *
//  * 5. 3색 램프 (D30/31/32, Active-Low 릴레이, 12V 램프):
//  *    - D30 (RED)    : EMERGENCY / KILL 상태
//  *    - D31 (YELLOW) : MANUAL 모드
//  *    - D32 (GREEN)  : AUTO 모드
//  *
//  * [주의]
//  *  - 이 스케치는 오직 Serial(USB, /dev/ttyACM0, 115200bps)만 사용.
//  *  - ROS 노드(motor_PID_test1.py)와만 연결하고,
//  *    다른 Python 스크립트(motor_remote.py 등)와 동시에 같은 포트 사용하면 안 됨.
//  */

// #include <ctype.h>
// #include <string.h>

// // ========== 디버그 설정 ==========
// #define DEBUG 0
// #if DEBUG
//   #define DBG_PRINT(x)   Serial.print(x)
//   #define DBG_PRINTLN(x) Serial.println(x)
// #else
//   #define DBG_PRINT(x)
//   #define DBG_PRINTLN(x)
// #endif

// // =========================================================
// // 1. 핀 설정 (Pin Definitions)
// // =========================================================

// // --- RC 수신기 (RC Inputs) ---
// const int CH1_PIN = 22;   // Steering
// const int CH2_PIN = 23;   // Throttle
// const int CH6_PIN = 26;   // Kill Switch (Aux)

// // --- 모터 드라이버 (MDD10A) ---
// // 왼쪽 모터
// const int LEFT_DIR_PIN = 4;
// const int LEFT_PWM_PIN = 5;
// // 오른쪽 모터
// const int RIGHT_DIR_PIN = 7;
// const int RIGHT_PWM_PIN = 6;

// // --- 전원 제어용 릴레이 (NUC / MOTOR / LIDAR) ---
// const int RELAY_NUC   = 11;
// const int RELAY_MOTOR = 12;
// const int RELAY_LIDAR = 13;

// // 전원 릴레이 모듈: Active-Low (LOW = ON, HIGH = OFF)
// const int PWR_RELAY_ON_LEVEL  = LOW;
// const int PWR_RELAY_OFF_LEVEL = HIGH;

// // --- 비상 스위치 + 내장 LED (한 핀 공유) ---
// const int ESTOP_SW_LED_PIN = 2; // Active-Low 스위치 + LED

// // --- 3색 램프용 릴레이 (외부 12V 램프) ---
// // 테스트 코드 기준 그대로
// const int RELAY_LED_RED    = 32;  // 빨강 (EMERGENCY)
// const int RELAY_LED_YELLOW = 31;  // 노랑 (AUTO)
// const int RELAY_LED_GREEN  = 30;  // 초록 (MANUAL)

// // 3색 램프 릴레이 모듈: Active-Low (LOW = ON, HIGH = OFF)
// const int LED_RELAY_ON  = LOW;
// const int LED_RELAY_OFF = HIGH;

// // =========================================================
// // 2. 설정 상수 (Configuration)
// // =========================================================

// // --- 모터 제어 파라미터 ---
// const int PWM_MAX = 255;
// const int PWM_MIN = 0;             // 필요 시 20~30 정도로 올려도 됨
// const int RAMP_MAX_STEP = 30;      // 가감속 부드러움 정도 (작을수록 부드러움)

// // 모터 방향 정의 (배선에 따라 수정 가능)
// const bool LEFT_FORWARD_HIGH  = false; // Left 전진 = LOW
// const bool RIGHT_FORWARD_HIGH = true;  // Right 전진 = HIGH

// // --- RC 설정 ---
// const int RC_MIN = 1000;
// const int RC_MAX = 2000;
// const int RC_DEADZONE_PWM = 30;       // -255~255 스케일에서 데드존
// const int RC_KILL_THRESHOLD = 1500;   // CH6가 이 값보다 크면 Kill
// const unsigned long RC_FAILSAFE_TIMEOUT = 500; // CH1/2 신호 없을 때 failsafe

// // --- 시리얼 설정 ---
// const unsigned long SERIAL_BAUD     = 115200;
// const unsigned long SERIAL_TIMEOUT  = 1000; // 1초간 시리얼 명령 없으면 AUTO 정지
// const size_t BUF_SZ = 64;

// // =========================================================
// // 3. 전역 상태 (Global State)
// // =========================================================

// // 모드 정의
// enum RunMode { MODE_MANUAL, MODE_AUTO, MODE_EMERGENCY };

// // 상태 변수
// RunMode currentMode   = MODE_MANUAL;  // 실제 동작 모드
// RunMode requestedMode = MODE_MANUAL;  // ROS에서 요청한 모드 (<MODE:A/M>)

// bool softKill   = false;  // <KILL>/<CLEAR>로 제어되는 소프트웨어 Kill
// bool isKilled   = false;  // 현재 Kill 상태 (HW + CH6 + softKill)
// bool lastKilled = false;  // 직전 Loop의 Kill 상태
// bool powerEnabled = false; // 릴레이 ON/OFF 상태

// // 모터 현재 속도 (램핑용)
// int curSpeedL = 0;
// int curSpeedR = 0;

// // RC failsafe용 타임스탬프
// unsigned long lastRcSignalTime = 0;

// // 시리얼 관련
// char serBuf[BUF_SZ];
// int  serIdx = 0;
// unsigned long lastSerialTime = 0; // 마지막 유효 모션 패킷 시각

// // 현재 목표 모터 속도 (AUTO 모드에서 사용)
// int targetL = 0;
// int targetR = 0;

// // =========================================================
// // 4. 유틸 함수 (Utility Functions)
// // =========================================================

// // --- 전원 릴레이 제어 ---
// void setPowerRelay(int pin, bool on) {
//   digitalWrite(pin, on ? PWR_RELAY_ON_LEVEL : PWR_RELAY_OFF_LEVEL);
// }

// // 순차 전원 ON: NUC -> Motor -> LiDAR
// void sequentialPowerOn() {
//   // 1) NUC ON
//   setPowerRelay(RELAY_NUC, true);
//   delay(2000);

//   // 2) Motor ON
//   setPowerRelay(RELAY_MOTOR, true);
//   delay(2000);

//   // 3) LiDAR ON
//   setPowerRelay(RELAY_LIDAR, true);
// }

// // 순차 전원 OFF: Motor -> LiDAR -> NUC
// void sequentialPowerOff() {
//   // 1) Motor OFF
//   setPowerRelay(RELAY_MOTOR, false);
//   delay(2000);

//   // 2) LiDAR OFF
//   setPowerRelay(RELAY_LIDAR, false);
//   delay(2000);

//   // 3) NUC OFF
//   setPowerRelay(RELAY_NUC, false);
// }

// // --- E-STOP 스위치 읽기 (Pin2 공유) ---
// bool readEstopPressed() {
//   pinMode(ESTOP_SW_LED_PIN, INPUT_PULLUP);
//   int v = digitalRead(ESTOP_SW_LED_PIN);
//   return (v == LOW); // Active-Low
// }

// // --- E-STOP LED 업데이트 (Pin2 공유) ---
// void updateEstopLed(bool killed) {
//   pinMode(ESTOP_SW_LED_PIN, OUTPUT);
//   // killed = true -> LED OFF (전원 차단 상태 표시)
//   // killed = false -> LED ON  (정상 전원 상태)
//   digitalWrite(ESTOP_SW_LED_PIN, killed ? LOW : HIGH);
// }

// // --- 3색 램프 릴레이 제어 ---
// void setLedRelay(int pin, bool on) {
//   digitalWrite(pin, on ? LED_RELAY_ON : LED_RELAY_OFF);
// }

// // --- 모드에 따른 3색 램프 상태 업데이트 ---
// void updateModeLeds() {
//   // 기본값: 모두 OFF
//   bool redOn    = false;
//   bool yellowOn = false;
//   bool greenOn  = false;

//   if (isKilled || currentMode == MODE_EMERGENCY || !powerEnabled) {
//     // 비상/킬/전원OFF 상태 → 빨강 ON
//     redOn = true;
//   } else {
//     // 정상 전원 상태에서 모드에 따라
//     if (currentMode == MODE_AUTO) {
//       yellowOn = true;   // AUTO → 노랑
//     } else if (currentMode == MODE_MANUAL) {
//       greenOn = true;  // MANUAL → 초록
//     }
//   }

//   setLedRelay(RELAY_LED_RED,    redOn);
//   setLedRelay(RELAY_LED_YELLOW, yellowOn);
//   setLedRelay(RELAY_LED_GREEN,  greenOn);
// }

// // --- RC 신호 읽기 ---
// int readPulse(int pin) {
//   unsigned long pw = pulseIn(pin, HIGH, 30000); // 최대 30ms 대기
//   if (pw == 0) return 0; 
//   return (int)pw;
// }

// // --- 모터 구동 (Ramping + Direction 처리) ---
// void driveMotorRaw(int pwmPin, int dirPin, int &curVal, int targetVal, bool forwardHigh) {
//   // 1. 범위 제한
//   targetVal = constrain(targetVal, -PWM_MAX, PWM_MAX);
  
//   // 2. 데드존 (아주 작은 값은 0으로)
//   if (abs(targetVal) < 10) targetVal = 0;

//   // 3. Ramping (급격한 변화 방지)
//   int step = targetVal - curVal;
//   if (step > RAMP_MAX_STEP)  step = RAMP_MAX_STEP;
//   if (step < -RAMP_MAX_STEP) step = -RAMP_MAX_STEP;
//   curVal += step;

//   // 4. 하드웨어 출력
//   bool forward = (curVal >= 0);
//   int speed = abs(curVal);

//   // 방향 핀 설정
//   if (forwardHigh) {
//     digitalWrite(dirPin, forward ? HIGH : LOW);
//   } else {
//     digitalWrite(dirPin, forward ? LOW : HIGH);
//   }
  
//   analogWrite(pwmPin, speed);
// }

// // [수정] driveMotorRaw 함수 (최소 기동 PWM 적용)
// // void driveMotorRaw(int pwmPin, int dirPin, int &curVal, int targetVal, bool forwardHigh) {
// //   targetVal = constrain(targetVal, -PWM_MAX, PWM_MAX);
  
// //   // [추가] 최소 기동 토크 보정 (Feedforward)
// //   // 모터가 예를 들어 PWM 40 이하에서는 윙~ 소리만 나고 안 돈다면,
// //   // 40 이하의 명령이 들어오면 과감하게 0으로 끄거나, 돌려야 한다면 40부터 시작하게 함.
// //   int min_pwm = 30; // 모터 특성에 따라 조절 (20~50)
  
// //   if (targetVal > 0 && targetVal < min_pwm) targetVal = min_pwm;
// //   if (targetVal < 0 && targetVal > -min_pwm) targetVal = -min_pwm;
// //   if (abs(targetVal) < 5) targetVal = 0; // 완전 정지 구간

// //   // Ramping
// //   int step = targetVal - curVal;
// //   if (step > RAMP_MAX_STEP)  step = RAMP_MAX_STEP;
// //   if (step < -RAMP_MAX_STEP) step = -RAMP_MAX_STEP;
// //   curVal += step;

// //   // 하드웨어 출력
// //   bool forward = (curVal >= 0);
// //   int speed = abs(curVal);

// //   if (forwardHigh) {
// //     digitalWrite(dirPin, forward ? HIGH : LOW);
// //   } else {
// //     digitalWrite(dirPin, forward ? LOW : HIGH);
// //   }
  
// //   analogWrite(pwmPin, speed);
// // }

// void updateMotors(int l, int r) {
//   driveMotorRaw(LEFT_PWM_PIN,  LEFT_DIR_PIN,  curSpeedL, l, LEFT_FORWARD_HIGH);
//   driveMotorRaw(RIGHT_PWM_PIN, RIGHT_DIR_PIN, curSpeedR, r, RIGHT_FORWARD_HIGH);
// }

// void stopMotorsFast() {
//   analogWrite(LEFT_PWM_PIN, 0);
//   analogWrite(RIGHT_PWM_PIN, 0);
//   curSpeedL = 0;
//   curSpeedR = 0;
// }

// // --- 시리얼 모션 패킷 파싱 <L:+100,R:-100> ---
// bool parseMotionPacket(const char* s, int& outL, int& outR) {
//   int l=0, r=0, signL=1, signR=1, p=0;

//   if (s[p++]!='<') return false;
//   if (s[p++]!='L' || s[p++]!=':') return false;

//   if (s[p]=='+') { signL= 1; p++; }
//   else if (s[p]=='-') { signL=-1; p++; }

//   if (!isdigit(s[p])) return false;
//   while (isdigit(s[p])) { l = l*10 + (s[p++]-'0'); }

//   if (s[p++]!=',') return false;
//   if (s[p++]!='R' || s[p++]!=':') return false;

//   if (s[p]=='+') { signR= 1; p++; }
//   else if (s[p]=='-') { signR=-1; p++; }

//   if (!isdigit(s[p])) return false;
//   while (isdigit(s[p])) { r = r*10 + (s[p++]-'0'); }

//   if (s[p++]!='>') return false;

//   outL = constrain(signL*l, -PWM_MAX, PWM_MAX);
//   outR = constrain(signR*r, -PWM_MAX, PWM_MAX);
//   return true;
// }

// // --- 시리얼 제어 명령 처리 (<MODE:A>, <KILL>, <CLEAR> 등) ---
// void handleCommandPacket(const char* s) {
//   // <KILL>
//   if (strcmp(s, "<KILL>") == 0) {
//     softKill = true;
//     DBG_PRINTLN("CMD: KILL");
//   }
//   // <CLEAR>
//   else if (strcmp(s, "<CLEAR>") == 0) {
//     softKill = false;
//     DBG_PRINTLN("CMD: CLEAR");
//   }
//   // <MODE:A> / <MODE:M> / <MODE:E>
//   else if (strncmp(s, "<MODE:", 6) == 0) {
//     char m = s[6];
//     if (m == 'A') {
//       requestedMode = MODE_AUTO;
//       DBG_PRINTLN("CMD: MODE=AUTO");
//     } else if (m == 'M') {
//       requestedMode = MODE_MANUAL;
//       DBG_PRINTLN("CMD: MODE=MANUAL");
//     } else if (m == 'E') {
//       // MODE:E는 사실상 KILL + EMERGENCY 의미로 사용
//       softKill      = true;
//       requestedMode = MODE_EMERGENCY;
//       DBG_PRINTLN("CMD: MODE=EMERGENCY(KILL)");
//     }
//   }
//   // 그 외는 무시
// }

// // =========================================================
// // 5. Setup
// // =========================================================
// void setup() {
//   Serial.begin(SERIAL_BAUD);

//   // 핀 모드 설정
//   pinMode(CH1_PIN, INPUT);
//   pinMode(CH2_PIN, INPUT);
//   pinMode(CH6_PIN, INPUT);
  
//   pinMode(RELAY_NUC,   OUTPUT);
//   pinMode(RELAY_MOTOR, OUTPUT);
//   pinMode(RELAY_LIDAR, OUTPUT);
  
//   pinMode(LEFT_DIR_PIN,  OUTPUT);
//   pinMode(LEFT_PWM_PIN,  OUTPUT);
//   pinMode(RIGHT_DIR_PIN, OUTPUT);
//   pinMode(RIGHT_PWM_PIN, OUTPUT);

//   // 3색 램프 릴레이 핀
//   pinMode(RELAY_LED_RED,    OUTPUT);
//   pinMode(RELAY_LED_YELLOW, OUTPUT);
//   pinMode(RELAY_LED_GREEN,  OUTPUT);

//   // 초기에는 모두 OFF
//   //setPowerRelay(RELAY_NUC,   false);
//   setPowerRelay(RELAY_MOTOR, false);
//   setPowerRelay(RELAY_LIDAR, false);
//   setLedRelay(RELAY_LED_RED,    false);
//   setLedRelay(RELAY_LED_YELLOW, false);
//   setLedRelay(RELAY_LED_GREEN,  false);

//   // 초기 안전 상태 확인
//   delay(100); // 안정화 대기
//   bool estop_pressed = readEstopPressed();
//   int  ch6           = readPulse(CH6_PIN);
//   bool rc_kill       = (ch6 > RC_KILL_THRESHOLD && ch6 < 2500);

//   softKill   = false; // 시작 시 원격 Kill은 없다고 가정
//   isKilled   = estop_pressed || rc_kill || softKill;
//   lastKilled = isKilled;

//   if (isKilled) {
//     Serial.println("INIT: KILL STATE (Power OFF)");
//     sequentialPowerOff();
//     powerEnabled = false;
//     currentMode  = MODE_EMERGENCY;
//   } else {
//     Serial.println("INIT: NORMAL STATE (Power ON)");
//     sequentialPowerOn();
//     powerEnabled = true;
//     currentMode  = MODE_MANUAL;  // 기본 시작은 RC 수동
//   }

//   // LED 초기 상태
//   updateEstopLed(isKilled);
//   updateModeLeds();

//   lastRcSignalTime = millis();
//   lastSerialTime   = millis();
// }

// // =========================================================
// // 6. Loop
// // =========================================================
// void loop() {
//   // --- 1. E-STOP & CH6 상태 읽기 ---
//   bool estop_pressed = readEstopPressed(); // Pin2를 INPUT_PULLUP으로 읽은 뒤, LED는 나중에 OUTPUT으로 설정
//   int  ch6_raw       = readPulse(CH6_PIN);
//   bool rc_kill       = (ch6_raw > RC_KILL_THRESHOLD) && (ch6_raw < 2500);

//   // Kill 상태 결정
//   bool killNow = estop_pressed || rc_kill || softKill;

//   // --- 2. Kill 상태 변화에 따른 전원 제어 ---
//   if (killNow && !lastKilled) {
//     // Normal -> Kill
//     Serial.println("SAFETY TRIGGERED: Power OFF (sequential)");
//     stopMotorsFast();
//     sequentialPowerOff();
//     powerEnabled = false;
//     currentMode  = MODE_EMERGENCY;
//   }
//   else if (!killNow && lastKilled) {
//     // Kill -> Normal
//     Serial.println("SAFETY CLEARED: Power ON (sequential)");
//     sequentialPowerOn();
//     powerEnabled = true;
//     // Kill 해제 후 기본은 MANUAL, 필요하면 요청에 따라 AUTO
//     if (requestedMode == MODE_AUTO) {
//       currentMode = MODE_AUTO;
//     } else {
//       currentMode = MODE_MANUAL;
//     }
//   }

//   isKilled   = killNow;
//   lastKilled = killNow;

//   // --- 3. RC 입력 읽기 (CH1, CH2) ---
//   int ch1 = readPulse(CH1_PIN); // Steer
//   int ch2 = readPulse(CH2_PIN); // Throttle

//   int steerInput    = 0;
//   int throttleInput = 0;
//   bool rcSignalValid = false;

//   if (ch1 > 900 && ch1 < 2100) {
//     steerInput = map(ch1, RC_MIN, RC_MAX, -255, 255);
//     rcSignalValid = true;
//   }
//   if (ch2 > 900 && ch2 < 2100) {
//     throttleInput = map(ch2, RC_MIN, RC_MAX, -255, 255);
//     rcSignalValid = true;
//   }

//   if (rcSignalValid) {
//     lastRcSignalTime = millis();
//   }

//   // 데드존 처리
//   if (abs(steerInput)    < RC_DEADZONE_PWM) steerInput    = 0;
//   if (abs(throttleInput) < RC_DEADZONE_PWM) throttleInput = 0;

//   bool rcActive = (abs(steerInput) > 0 || abs(throttleInput) > 0);

//   // --- 4. 시리얼 수신 (모션/명령) ---
//   while (Serial.available()) {
//     char c = Serial.read();
//     if (c == '\n' || c == '\r') {
//       if (serIdx > 0) {
//         serBuf[serIdx] = '\0';
//         serIdx = 0;

//         int sl, sr;
//         if (parseMotionPacket(serBuf, sl, sr)) {
//           // 모션 패킷
//           targetL = sl;
//           targetR = sr;
//           lastSerialTime = millis();
//           DBG_PRINT("MOTION: L="); DBG_PRINT(sl);
//           DBG_PRINT(" R="); DBG_PRINTLN(sr);
//         } else {
//           // 제어 명령 패킷
//           handleCommandPacket(serBuf);
//         }
//       } else {
//         serIdx = 0;
//       }
//     } else {
//       if (serIdx < (int)BUF_SZ - 1) {
//         serBuf[serIdx++] = c;
//       }
//     }
//   }

//   // --- 5. 모드 결정 ---
//   if (isKilled || !powerEnabled) {
//     currentMode = MODE_EMERGENCY;
//   } else {
//     if (rcActive) {
//       // RC 스틱이 움직이면 무조건 수동 우선
//       currentMode = MODE_MANUAL;
//     } else {
//       // RC 중립 상태 -> ROS 모드 요청에 따라
//       if (requestedMode == MODE_AUTO) {
//         currentMode = MODE_AUTO;
//       } else {
//         currentMode = MODE_MANUAL;
//       }
//     }
//   }

//   // --- 6. 모터 명령 결정 ---
//   int finalL = 0;
//   int finalR = 0;

//   if (currentMode == MODE_EMERGENCY || isKilled || !powerEnabled) {
//     finalL = finalR = 0;
//   }
//   else if (currentMode == MODE_MANUAL) {
//     // RC 수동 모드 (Arcade Drive Mixing)
//     // failsafe: 일정 시간 이상 CH1/2 신호 없으면 정지
//     if (millis() - lastRcSignalTime > RC_FAILSAFE_TIMEOUT) {
//       finalL = finalR = 0;
//     } else {
//       int leftCmd  = throttleInput + steerInput;
//       int rightCmd = throttleInput - steerInput;
//       finalL = constrain(leftCmd,  -255, 255);
//       finalR = constrain(rightCmd, -255, 255);
//     }
//   }
//   else if (currentMode == MODE_AUTO) {
//     // AUTO 모드: 최근 시리얼 명령 사용 (타임아웃 시 정지)
//     if (millis() - lastSerialTime < SERIAL_TIMEOUT) {
//       finalL = targetL;
//       finalR = targetR;
//     } else {
//       finalL = finalR = 0;
//     }
//   }

//   // --- 7. 모터 구동/정지 ---
//   if (currentMode == MODE_EMERGENCY || isKilled || !powerEnabled) {
//     stopMotorsFast();
//   } else {
//     updateMotors(finalL, finalR);
//   }

//   // --- 8. LED 업데이트 ---
//   updateEstopLed(isKilled);
//   updateModeLeds();

//   delay(20); // 약 50Hz 루프
// }




/*
 * Final Integration Code v3_NoRC (ROS Only + Safety)
 * 통합: ROS(AUTO) + Power/Safety + 3색 LED 램프 (RC 제거됨)
 *
 * [기능 요약]
 * 1. Safety:
 * - 물리 E-STOP 버튼 (Pin 2, Active-Low)
 * - 원격 <KILL>/<CLEAR> 명령
 * → Kill시 릴레이 순차 OFF (Motor -> LiDAR -> NUC)
 *
 * 2. Power ON:
 * → NUC -> Motor -> LiDAR 순서로 전원 인가
 *
 * 3. Motor:
 * - MDD10A 드라이버 제어 (DIR/PWM)
 * - 램핑(Ramping) 및 최소 기동 부하 보정
 *
 * 4. Control:
 * - AUTO 모드: ROS 노드 명령(<L:..R:..>)에 따라 주행
 * - MANUAL 모드: RC가 없으므로 "대기(정지)" 상태가 됨
 * - EMERGENCY 모드: Kill 조건 발생 시 진입
 *
 * 5. 3색 램프:
 * - RED    : EMERGENCY / KILL
 * - YELLOW : AUTO (주행 중)
 * - GREEN  : MANUAL (대기 중)
 */

#include <ctype.h>
#include <string.h>

// ========== 디버그 설정 ==========
#define DEBUG 0
#if DEBUG
  #define DBG_PRINT(x)   Serial.print(x)
  #define DBG_PRINTLN(x) Serial.println(x)
#else
  #define DBG_PRINT(x)
  #define DBG_PRINTLN(x)
#endif

// =========================================================
// 1. 핀 설정 (Pin Definitions)
// =========================================================

// --- 모터 드라이버 (MDD10A) ---
const int LEFT_DIR_PIN = 4;
const int LEFT_PWM_PIN = 5;
const int RIGHT_DIR_PIN = 7;
const int RIGHT_PWM_PIN = 6;

// --- 전원 제어용 릴레이 (NUC / MOTOR / LIDAR) ---
const int RELAY_NUC   = 11;
const int RELAY_MOTOR = 12;
const int RELAY_LIDAR = 13;

const int PWR_RELAY_ON_LEVEL  = LOW;
const int PWR_RELAY_OFF_LEVEL = HIGH;

// --- 비상 스위치 + 내장 LED (한 핀 공유) ---
const int ESTOP_SW_LED_PIN = 2; // Active-Low 스위치 + LED

// --- 3색 램프용 릴레이 (외부 12V 램프) ---
const int RELAY_LED_RED    = 32;  // 빨강 (EMERGENCY)
const int RELAY_LED_YELLOW = 31;  // 노랑 (AUTO)
const int RELAY_LED_GREEN  = 30;  // 초록 (MANUAL/IDLE)

const int LED_RELAY_ON  = LOW;
const int LED_RELAY_OFF = HIGH;

// =========================================================
// 2. 설정 상수 (Configuration)
// =========================================================

// --- 모터 제어 파라미터 ---
const int PWM_MAX = 255;
const int RAMP_MAX_STEP = 30;      // 가감속 부드러움 정도

// 모터 방향 정의
const bool LEFT_FORWARD_HIGH  = false; 
const bool RIGHT_FORWARD_HIGH = true;  

// --- 시리얼 설정 ---
const unsigned long SERIAL_BAUD     = 115200;
const unsigned long SERIAL_TIMEOUT  = 1000; // 1초간 시리얼 명령 없으면 정지
const size_t BUF_SZ = 64;

// =========================================================
// 3. 전역 상태 (Global State)
// =========================================================

enum RunMode { MODE_MANUAL, MODE_AUTO, MODE_EMERGENCY };

RunMode currentMode   = MODE_MANUAL;  // 현재 상태
RunMode requestedMode = MODE_MANUAL;  // 요청된 상태

bool softKill   = false;  // 시리얼 명령에 의한 Kill
bool isKilled   = false;  // 최종 Kill 상태
bool lastKilled = false; 
bool powerEnabled = false;

// 모터 현재 속도 (램핑용)
int curSpeedL = 0;
int curSpeedR = 0;

// 시리얼 관련
char serBuf[BUF_SZ];
int  serIdx = 0;
unsigned long lastSerialTime = 0;

// 목표 속도
int targetL = 0;
int targetR = 0;

// =========================================================
// 4. 유틸 함수
// =========================================================

void setPowerRelay(int pin, bool on) {
  digitalWrite(pin, on ? PWR_RELAY_ON_LEVEL : PWR_RELAY_OFF_LEVEL);
}

// 순차 전원 ON
void sequentialPowerOn() {
  setPowerRelay(RELAY_NUC, true);
  delay(2000);
  setPowerRelay(RELAY_MOTOR, true);
  delay(2000);
  setPowerRelay(RELAY_LIDAR, true);
}

// 순차 전원 OFF
void sequentialPowerOff() {
  setPowerRelay(RELAY_MOTOR, false);
  delay(2000);
  setPowerRelay(RELAY_LIDAR, false);
  delay(2000);
  setPowerRelay(RELAY_NUC, false);
}

// E-STOP 읽기
bool readEstopPressed() {
  pinMode(ESTOP_SW_LED_PIN, INPUT_PULLUP);
  int v = digitalRead(ESTOP_SW_LED_PIN);
  return (v == LOW); 
}

// E-STOP LED
void updateEstopLed(bool killed) {
  pinMode(ESTOP_SW_LED_PIN, OUTPUT);
  digitalWrite(ESTOP_SW_LED_PIN, killed ? LOW : HIGH);
}

void setLedRelay(int pin, bool on) {
  digitalWrite(pin, on ? LED_RELAY_ON : LED_RELAY_OFF);
}

void updateModeLeds() {
  bool redOn    = false;
  bool yellowOn = false;
  bool greenOn  = false;

  if (isKilled || currentMode == MODE_EMERGENCY || !powerEnabled) {
    redOn = true;
  } else {
    if (currentMode == MODE_AUTO) {
      yellowOn = true; 
    } else {
      greenOn = true;  // MANUAL = IDLE
    }
  }

  setLedRelay(RELAY_LED_RED,    redOn);
  setLedRelay(RELAY_LED_YELLOW, yellowOn);
  setLedRelay(RELAY_LED_GREEN,  greenOn);
}

// [수정] driveMotorRaw 함수 (최소 기동 PWM 적용)
void driveMotorRaw(int pwmPin, int dirPin, int &curVal, int targetVal, bool forwardHigh) {
  targetVal = constrain(targetVal, -PWM_MAX, PWM_MAX);
  
  // [추가] 최소 기동 토크 보정 (Feedforward)
  // 모터가 예를 들어 PWM 40 이하에서는 윙~ 소리만 나고 안 돈다면,
  // 40 이하의 명령이 들어오면 과감하게 0으로 끄거나, 돌려야 한다면 40부터 시작하게 함.
  int min_pwm = 30; // 모터 특성에 따라 조절 (20~50)
  
  if (targetVal > 0 && targetVal < min_pwm) targetVal = min_pwm;
  if (targetVal < 0 && targetVal > -min_pwm) targetVal = -min_pwm;
  if (abs(targetVal) < 5) targetVal = 0; // 완전 정지 구간

  // Ramping
  int step = targetVal - curVal;
  if (step > RAMP_MAX_STEP)  step = RAMP_MAX_STEP;
  if (step < -RAMP_MAX_STEP) step = -RAMP_MAX_STEP;
  curVal += step;

  // 하드웨어 출력
  bool forward = (curVal >= 0);
  int speed = abs(curVal);

  if (forwardHigh) {
    digitalWrite(dirPin, forward ? HIGH : LOW);
  } else {
    digitalWrite(dirPin, forward ? LOW : HIGH);
  }
  
  analogWrite(pwmPin, speed);
}

void updateMotors(int l, int r) {
  driveMotorRaw(LEFT_PWM_PIN,  LEFT_DIR_PIN,  curSpeedL, l, LEFT_FORWARD_HIGH);
  driveMotorRaw(RIGHT_PWM_PIN, RIGHT_DIR_PIN, curSpeedR, r, RIGHT_FORWARD_HIGH);
}

void stopMotorsFast() {
  analogWrite(LEFT_PWM_PIN, 0);
  analogWrite(RIGHT_PWM_PIN, 0);
  curSpeedL = 0;
  curSpeedR = 0;
}

// 시리얼 패킷 파싱
bool parseMotionPacket(const char* s, int& outL, int& outR) {
  int l=0, r=0, signL=1, signR=1, p=0;
  if (s[p++]!='<') return false;
  if (s[p++]!='L' || s[p++]!=':') return false;
  if (s[p]=='+') { signL= 1; p++; } else if (s[p]=='-') { signL=-1; p++; }
  if (!isdigit(s[p])) return false;
  while (isdigit(s[p])) { l = l*10 + (s[p++]-'0'); }
  if (s[p++]!=',') return false;
  if (s[p++]!='R' || s[p++]!=':') return false;
  if (s[p]=='+') { signR= 1; p++; } else if (s[p]=='-') { signR=-1; p++; }
  if (!isdigit(s[p])) return false;
  while (isdigit(s[p])) { r = r*10 + (s[p++]-'0'); }
  if (s[p++]!='>') return false;
  outL = constrain(signL*l, -PWM_MAX, PWM_MAX);
  outR = constrain(signR*r, -PWM_MAX, PWM_MAX);
  return true;
}

void handleCommandPacket(const char* s) {
  if (strcmp(s, "<KILL>") == 0) {
    softKill = true;
    DBG_PRINTLN("CMD: KILL");
  }
  else if (strcmp(s, "<CLEAR>") == 0) {
    softKill = false;
    DBG_PRINTLN("CMD: CLEAR");
  }
  else if (strncmp(s, "<MODE:", 6) == 0) {
    char m = s[6];
    if (m == 'A') {
      requestedMode = MODE_AUTO;
      DBG_PRINTLN("CMD: MODE=AUTO");
    } else if (m == 'M') {
      requestedMode = MODE_MANUAL;
      DBG_PRINTLN("CMD: MODE=MANUAL(IDLE)");
    } else if (m == 'E') {
      softKill = true;
      requestedMode = MODE_EMERGENCY;
    }
  }
}

// =========================================================
// 5. Setup
// =========================================================
void setup() {
  Serial.begin(SERIAL_BAUD);

  pinMode(RELAY_NUC,   OUTPUT);
  pinMode(RELAY_MOTOR, OUTPUT);
  pinMode(RELAY_LIDAR, OUTPUT);
  
  pinMode(LEFT_DIR_PIN,  OUTPUT);
  pinMode(LEFT_PWM_PIN,  OUTPUT);
  pinMode(RIGHT_DIR_PIN, OUTPUT);
  pinMode(RIGHT_PWM_PIN, OUTPUT);

  pinMode(RELAY_LED_RED,    OUTPUT);
  pinMode(RELAY_LED_YELLOW, OUTPUT);
  pinMode(RELAY_LED_GREEN,  OUTPUT);

  // 초기 OFF
  setPowerRelay(RELAY_MOTOR, false);
  setPowerRelay(RELAY_LIDAR, false);
  setLedRelay(RELAY_LED_RED,    false);
  setLedRelay(RELAY_LED_YELLOW, false);
  setLedRelay(RELAY_LED_GREEN,  false);

  delay(100); 
  bool estop_pressed = readEstopPressed();

  softKill = false;
  isKilled = estop_pressed; 
  lastKilled = isKilled;

  if (isKilled) {
    Serial.println("INIT: KILL STATE (Power OFF)");
    sequentialPowerOff();
    powerEnabled = false;
    currentMode  = MODE_EMERGENCY;
  } else {
    Serial.println("INIT: NORMAL STATE (Power ON)");
    sequentialPowerOn();
    powerEnabled = true;
    currentMode  = MODE_MANUAL; // Default: Idle
  }

  updateEstopLed(isKilled);
  updateModeLeds();
  lastSerialTime = millis();
}

// =========================================================
// 6. Loop
// =========================================================
void loop() {
  // --- 1. E-STOP 상태 읽기 ---
  bool estop_pressed = readEstopPressed();
  
  // Kill 조건 확인 (물리 버튼 OR 소프트웨어 명령)
  bool killNow = estop_pressed || softKill;

  // --- 2. Kill 상태 변화 처리 ---
  if (killNow && !lastKilled) {
    Serial.println("SAFETY TRIGGERED: Power OFF");
    stopMotorsFast();
    sequentialPowerOff();
    powerEnabled = false;
    currentMode  = MODE_EMERGENCY;
  }
  else if (!killNow && lastKilled) {
    Serial.println("SAFETY CLEARED: Power ON");
    sequentialPowerOn();
    powerEnabled = true;
    // 복구 시 요청된 모드로
    currentMode = requestedMode;
  }

  isKilled   = killNow;
  lastKilled = killNow;

  // --- 3. 시리얼 수신 ---
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serIdx > 0) {
        serBuf[serIdx] = '\0';
        serIdx = 0;
        int sl, sr;
        if (parseMotionPacket(serBuf, sl, sr)) {
          targetL = sl;
          targetR = sr;
          lastSerialTime = millis();
        } else {
          handleCommandPacket(serBuf);
        }
      } else { serIdx = 0; }
    } else {
      if (serIdx < (int)BUF_SZ - 1) serBuf[serIdx++] = c;
    }
  }

  // --- 4. 모드 결정 ---
  if (isKilled || !powerEnabled) {
    currentMode = MODE_EMERGENCY;
  } else {
    // RC가 없으므로 단순히 요청된 모드 따름
    currentMode = requestedMode;
  }

  // --- 5. 모터 명령 ---
  int finalL = 0;
  int finalR = 0;

  if (currentMode == MODE_AUTO) {
    if (millis() - lastSerialTime < SERIAL_TIMEOUT) {
      finalL = targetL;
      finalR = targetR;
    } else {
      // 타임아웃 시 정지
      finalL = 0;
      finalR = 0;
    }
  } 
  // MODE_MANUAL이나 EMERGENCY는 둘 다 0 (정지)

  // --- 6. 모터 구동 ---
  if (currentMode == MODE_EMERGENCY || !powerEnabled) {
    stopMotorsFast();
  } else {
    updateMotors(finalL, finalR);
  }

  // --- 7. LED 업데이트 ---
  updateEstopLed(isKilled);
  updateModeLeds();

  delay(20); 
}