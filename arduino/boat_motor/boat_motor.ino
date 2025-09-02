/*
 * 보트 제어 시스템 - Arduino (하이브리드 버전)
 * 좌우 모터 개별 제어 + 단일 명령 지원
 * 
 * 핀 연결:
 * ENA(좌측PWM): 5
 * ENB(우측PWM): 6  
 * IN1(좌측DIR1): 7
 * IN2(좌측DIR2): 8
 * IN3(우측DIR1): 9
 * IN4(우측DIR2): 10
 * 
 * 지원 프로토콜:
 * 1. L{speed},R{speed} - 개별 모터 제어 (수동 모드)
 * 2. F, B, L, R, S - 단일 명령 (자동 회피 모드)
 */

// 모터 핀 정의
#define LEFT_PWM_PIN 5    // ENA
#define LEFT_DIR1_PIN 7   // IN1
#define LEFT_DIR2_PIN 8   // IN2

#define RIGHT_PWM_PIN 6   // ENB
#define RIGHT_DIR1_PIN 9  // IN3
#define RIGHT_DIR2_PIN 10 // IN4

// LED 핀
#define LED_PIN 13

// 자동 모드 기본 속도
#define AUTO_FORWARD_SPEED 150
#define AUTO_TURN_SPEED 130
#define AUTO_BACKWARD_SPEED 120

// 변수
int leftSpeed = 0;   // -255 ~ 255
int rightSpeed = 0;  // -255 ~ 255
String inputString = "";
boolean stringComplete = false;
String currentMode = "MANUAL"; // MANUAL 또는 AUTO

void setup() {
  Serial.begin(115200);
  
  // 모터 핀 설정
  pinMode(LEFT_PWM_PIN, OUTPUT);
  pinMode(LEFT_DIR1_PIN, OUTPUT);
  pinMode(LEFT_DIR2_PIN, OUTPUT);
  
  pinMode(RIGHT_PWM_PIN, OUTPUT);
  pinMode(RIGHT_DIR1_PIN, OUTPUT);
  pinMode(RIGHT_DIR2_PIN, OUTPUT);
  
  pinMode(LED_PIN, OUTPUT);
  
  // 초기화
  stopAllMotors();
  
  Serial.println("🚤 Arduino 하이브리드 보트 제어 시스템 시작!");
  Serial.println("================================================");
  Serial.println("핀 연결: ENA=5, ENB=6, IN1=7, IN2=8, IN3=9, IN4=10");
  Serial.println();
  Serial.println("지원 명령:");
  Serial.println("📱 수동 모드: L{speed},R{speed} (예: L150,R-150)");
  Serial.println("🤖 자동 모드: F(전진), B(후진), L(좌회전), R(우회전), S(정지)");
  Serial.println("🛠️  유틸리티: STOP(긴급정지), STATUS(상태확인)");
  Serial.println("================================================");
  
  // LED 시작 신호
  startupBlink();
}

void loop() {
  // 시리얼 통신 처리
  if (stringComplete) {
    processCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
  
  // 상태 LED (동작 중이면 켜짐)
  if (leftSpeed != 0 || rightSpeed != 0) {
    digitalWrite(LED_PIN, HIGH);
  } else {
    digitalWrite(LED_PIN, LOW);
  }
  
  delay(10);
}

void processCommand(String command) {
  command.trim();
  command.toUpperCase(); // 대소문자 구분 없이
  
  Serial.print("📡 수신: ");
  Serial.println(command);
  
  // 1. 개별 모터 제어 (L{speed},R{speed})
  if (command.startsWith("L") && command.indexOf(",R") > 0) {
    currentMode = "MANUAL";
    processManualCommand(command);
  }
  
  // 2. 단일 문자 명령 (자동 모드)
  else if (command.length() == 1) {
    currentMode = "AUTO";
    processAutoCommand(command.charAt(0));
  }
  
  // 3. 유틸리티 명령
  else if (command == "STOP") {
    emergencyStop();
  }
  else if (command == "STATUS") {
    printStatus();
  }
  else {
    Serial.println("❌ 잘못된 명령 형식");
    Serial.println("사용법:");
    Serial.println("  수동: L{speed},R{speed} (예: L150,R-150)");
    Serial.println("  자동: F, B, L, R, S");
    Serial.println("  기타: STOP, STATUS");
  }
}

void processManualCommand(String command) {
  int commaPos = command.indexOf(",R");
  
  // 좌측 모터 속도 추출
  String leftSpeedStr = command.substring(1, commaPos);
  leftSpeed = leftSpeedStr.toInt();
  
  // 우측 모터 속도 추출
  String rightSpeedStr = command.substring(commaPos + 2);
  rightSpeed = rightSpeedStr.toInt();
  
  // 속도 범위 제한
  leftSpeed = constrain(leftSpeed, -255, 255);
  rightSpeed = constrain(rightSpeed, -255, 255);
  
  // 모터 제어 실행
  setLeftMotor(leftSpeed);
  setRightMotor(rightSpeed);
  
  // 응답 전송
  Serial.print("🕹️  수동 실행: 좌측=");
  Serial.print(leftSpeed);
  Serial.print(", 우측=");
  Serial.println(rightSpeed);
  
  printMotorDirections();
}

void processAutoCommand(char command) {
  switch(command) {
    case 'F': // 전진
      leftSpeed = AUTO_FORWARD_SPEED;
      rightSpeed = -AUTO_FORWARD_SPEED;
      Serial.println("🤖 자동: 전진");
      break;
      
    case 'B': // 후진
      leftSpeed = -AUTO_BACKWARD_SPEED;
      rightSpeed = AUTO_BACKWARD_SPEED;
      Serial.println("🤖 자동: 후진");
      break;
      
    case 'L': // 좌회전
      leftSpeed = -AUTO_TURN_SPEED;
      rightSpeed = -AUTO_TURN_SPEED;
      Serial.println("🤖 자동: 좌회전");
      break;
      
    case 'R': // 우회전
      leftSpeed = AUTO_TURN_SPEED;
      rightSpeed = AUTO_TURN_SPEED;
      Serial.println("🤖 자동: 우회전");
      break;
      
    case 'S': // 정지
      leftSpeed = 0;
      rightSpeed = 0;
      Serial.println("🤖 자동: 정지");
      break;
      
    default:
      Serial.print("❌ 알 수 없는 자동 명령: ");
      Serial.println(command);
      return;
  }
  
  // 모터 제어 실행
  setLeftMotor(leftSpeed);
  setRightMotor(rightSpeed);
  
  printMotorDirections();
}

void setLeftMotor(int speed) {
  if (speed > 0) {
    // 좌측 전진 (시계 방향)
    digitalWrite(LEFT_DIR1_PIN, HIGH);
    digitalWrite(LEFT_DIR2_PIN, LOW);
    analogWrite(LEFT_PWM_PIN, speed);
  } else if (speed < 0) {
    // 좌측 후진 (반시계 방향)
    digitalWrite(LEFT_DIR1_PIN, LOW);
    digitalWrite(LEFT_DIR2_PIN, HIGH);
    analogWrite(LEFT_PWM_PIN, -speed);
  } else {
    // 정지
    digitalWrite(LEFT_DIR1_PIN, LOW);
    digitalWrite(LEFT_DIR2_PIN, LOW);
    analogWrite(LEFT_PWM_PIN, 0);
  }
}

void setRightMotor(int speed) {
  if (speed > 0) {
    // 우측 전진 (반시계 방향)
    digitalWrite(RIGHT_DIR1_PIN, LOW);
    digitalWrite(RIGHT_DIR2_PIN, HIGH);
    analogWrite(RIGHT_PWM_PIN, speed);
  } else if (speed < 0) {
    // 우측 후진 (시계 방향)
    digitalWrite(RIGHT_DIR1_PIN, HIGH);
    digitalWrite(RIGHT_DIR2_PIN, LOW);
    analogWrite(RIGHT_PWM_PIN, -speed);
  } else {
    // 정지
    digitalWrite(RIGHT_DIR1_PIN, LOW);
    digitalWrite(RIGHT_DIR2_PIN, LOW);
    analogWrite(RIGHT_PWM_PIN, 0);
  }
}

void stopAllMotors() {
  setLeftMotor(0);
  setRightMotor(0);
  leftSpeed = 0;
  rightSpeed = 0;
  Serial.println("⏹️  모든 모터 정지");
}

void emergencyStop() {
  stopAllMotors();
  Serial.println("🚨 비상 정지!");
  
  // 비상 정지 LED 경고
  for(int i = 0; i < 10; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
}

void printStatus() {
  Serial.println("📊 현재 상태:");
  Serial.print("  모드: ");
  Serial.println(currentMode);
  Serial.print("  좌측 모터: ");
  Serial.println(leftSpeed);
  Serial.print("  우측 모터: ");
  Serial.println(rightSpeed);
  printMotorDirections();
}

void printMotorDirections() {
  String leftDir = getDirectionString(leftSpeed);
  String rightDir = getReverseDirectionString(rightSpeed);
  
  Serial.print("  방향: 좌측(");
  Serial.print(leftDir);
  Serial.print(") 우측(");
  Serial.print(rightDir);
  Serial.println(")");
}

String getDirectionString(int speed) {
  if (speed > 0) return "시계";
  else if (speed < 0) return "반시계";
  else return "정지";
}

String getReverseDirectionString(int speed) {
  // 우측 모터는 반대로 표시 (실제 회전 방향)
  if (speed > 0) return "반시계";
  else if (speed < 0) return "시계";
  else return "정지";
}

void startupBlink() {
  for(int i = 0; i < 5; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(150);
    digitalWrite(LED_PIN, LOW);
    delay(150);
  }
}

// 시리얼 이벤트 핸들러
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    
    if (inChar == '\n' || inChar == '\r') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}