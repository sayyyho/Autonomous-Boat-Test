/*
 * 보트 제어 시스템 - Arduino 
 * 좌우 모터 개별 제어
 * 
 * 핀 연결:
 * ENA(좌측PWM): 5
 * ENB(우측PWM): 6  
 * IN1(좌측DIR1): 7
 * IN2(좌측DIR2): 8
 * IN3(우측DIR1): 9
 * IN4(우측DIR2): 10
 * 
 * 전진: 좌측(시계) + 우측(반시계)
 * 후진: 좌측(반시계) + 우측(시계)
 * 좌회전: 양쪽(반시계)
 * 우회전: 양쪽(시계)
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

// 변수
int leftSpeed = 0;   // -255 ~ 255
int rightSpeed = 0;  // -255 ~ 255
String inputString = "";
boolean stringComplete = false;

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
  
  Serial.println("🚤 Arduino 보트 제어 시스템 시작!");
  Serial.println("핀 연결: ENA=5, ENB=6, IN1=7, IN2=8, IN3=9, IN4=10");
  Serial.println("전진: 좌측(시계) + 우측(반시계)");
  Serial.println("프로토콜: L{speed},R{speed}");
  Serial.println("예시: L150,R150 (안정적 전진)");
  
  // LED 시작 신호
  for(int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
}

void loop() {
  // 시리얼 통신 처리
  if (stringComplete) {
    processCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
  
  // 상태 LED
  if (leftSpeed != 0 || rightSpeed != 0) {
    digitalWrite(LED_PIN, HIGH);
  } else {
    digitalWrite(LED_PIN, LOW);
  }
  
  delay(10);
}

void processCommand(String command) {
  command.trim();
  
  Serial.print("📡 수신: ");
  Serial.println(command);
  
  // "L{speed},R{speed}" 형식 파싱
  if (command.startsWith("L") && command.indexOf(",R") > 0) {
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
    Serial.print("실행: 좌측=");
    Serial.print(leftSpeed);
    Serial.print(", 우측=");
    Serial.println(rightSpeed);
    
    // 방향 표시
    String leftDir = (leftSpeed > 0) ? "시계" : (leftSpeed < 0) ? "반시계" : "정지";
    String rightDir = (rightSpeed > 0) ? "반시계" : (rightSpeed < 0) ? "시계" : "정지";
    Serial.print("방향: 좌측(");
    Serial.print(leftDir);
    Serial.print(") 우측(");
    Serial.print(rightDir);
    Serial.println(")");
    
  } else {
    Serial.println("잘못된 명령 형식");
    Serial.println("올바른 형식: L{speed},R{speed}");
  }
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
  Serial.println("모든 모터 정지");
}

// 시리얼 이벤트 핸들러
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}

// 비상 정지 함수
void emergencyStop() {
  stopAllMotors();
  Serial.println("비상 정지!");
  
  for(int i = 0; i < 10; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
}