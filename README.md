# 🚢 자율운항보트 테스트

ROS2 Humble과 아두이노를 활용한 자율운항보트입니다.

## 📋 시스템 개요

- **메인 컴퓨터**: 라즈베리파이 5 (Ubuntu 24.04 + Docker + ROS2 Humble)
- **모터 제어**: 아두이노 (USB 시리얼 통신)
- **원격 제어**: 웹 기반 (WebSocket + ROS Bridge)
- **센서 시각화**: RVIZ2 (라이다 + 카메라)

## 🏗️ 시스템 아키텍처

```
[원격 PC] ←─ WiFi ─→ [라즈베리파이] ←─ USB ─→ [아두이노] ←─ PWM ─→ [모터]
     │                      │
  웹 조이스틱              ROS2 + RVIZ2
  WebSocket               센서 시각화
```

## 📊 데이터 흐름

1. **원격 PC** → 조이스틱 조작 (웹 또는 앱)
2. **WebSocket** → 라즈베리파이 rosbridge로 전송
3. **ROS2** → `/cmd_vel` 토픽 퍼블리시
4. **motor_controller.py** → 토픽 구독
5. **USB 시리얼** → 아두이노에 속도 데이터 전송
6. **아두이노** → PWM 신호로 모터 드라이버 제어
7. **모터 드라이버** → 실제 모터 구동
8. **라이다/카메라** → RVIZ2에서 시각화

## 🔧 설치 및 설정

### 🔴 라즈베리파이 5(메인 컴퓨터)

#### 필수 설치 패키지
```bash
# 호스트 OS (Ubuntu 24.04)
sudo apt update && sudo apt install -y docker.io

# Docker 컨테이너 내부 (Ubuntu 22.04 + ROS2 Humble)
sudo apt install -y \
  ros-humble-desktop \
  python3-serial \
  ros-humble-rosbridge-suite \
  ros-humble-rqt-robot-steering
```

#### 개발할 파일들
- `motor_controller.py` - ROS2 노드 (cmd_vel → 아두이노 시리얼 통신)
- `boat_control_node.py` - 통합 제어 노드
- `launch/boat_system.launch.py` - 전체 시스템 런치 파일

**기술 스택**: ROS2, Python, Docker, Serial

### 🔵 아두이노 (모터 제어)

#### 필수 라이브러리
```cpp
// Arduino IDE 라이브러리 매니저에서 설치
- rosserial_arduino
- geometry_msgs
- Motor Driver Library (예: L298N)
- SoftwareSerial (필요시)
```

#### 개발할 파일들
- `boat_motor_control.ino` - 메인 아두이노 스케치 (ROS 구독자 + 모터 제어)
- `motor_functions.h` - 모터 제어 함수들
- `config.h` - 핀 설정 및 상수

**기술 스택**: C/C++, Arduino IDE, ROS Serial

### 🟢 원격 제어 PC

#### 필수 설치
```bash
# 선택사항: ROS2 설치 (GUI 제어용)
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-rqt-robot-steering

# 필수: 웹 브라우저, SSH 클라이언트
```

#### 개발할 파일들
- `remote_control.html` - 웹 기반 조이스틱 제어
- `boat_controller.js` - 웹소켓 통신 + 제어 로직
- `mobile_app/` (선택) - 모바일 앱 (React Native)

**기술 스택**: HTML/JS, WebSocket, ROS Bridge

## 🚀 구현 순서 (단계별 가이드)

### 1단계: 아두이노 기본 모터 제어 구현
- 시리얼로 "1.0,0.5" 같은 명령 받기
- PWM으로 모터 드라이버 제어
- **테스트**: 시리얼 모니터로 직접 명령 전송

### 2단계: 라즈베리파이 ROS2 시리얼 노드 개발
- motor_controller.py 작성
- /cmd_vel 구독 → 시리얼 전송
- **테스트**: `ros2 topic pub`으로 직접 명령

### 3단계: 로컬 네트워크 ROS2 통신 테스트
- ROS_DOMAIN_ID 설정
- 같은 네트워크에서 토픽 통신 확인
- **테스트**: 다른 PC에서 토픽 확인

### 4단계: rosbridge 웹소켓 서버 구축
- rosbridge_suite 설치 및 실행
- 포트 9090 웹소켓 서버 구동
- **테스트**: 브라우저 콘솔에서 연결 확인

### 5단계: 웹 기반 원격 조이스틱 개발
- HTML/JavaScript 조이스틱 UI
- roslib.js로 웹소켓 통신
- **테스트**: 웹에서 모터 동작 확인

### 6단계: Docker 환경 통합 및 최적화
- 시리얼 권한 설정
- 포트 포워딩 설정
- launch 파일로 전체 시스템 구동

### 7단계: RVIZ2 연동 및 최종 테스트
- 센서 데이터와 모터 제어 동시 확인
- 네트워크 지연시간 최적화
- 안전 장치 (비상정지) 구현

## 📁 프로젝트 구조

```
autonomous_boat/
├── arduino/
│   ├── boat_motor_control.ino
│   ├── motor_functions.h
│   └── config.h
├── ros2_ws/
│   └── src/
│       └── boat_control/
│           ├── boat_control/
│           │   ├── motor_controller.py
│           │   └── boat_control_node.py
│           ├── launch/
│           │   └── boat_system.launch.py
│           └── package.xml
├── web_interface/
│   ├── remote_control.html
│   ├── boat_controller.js
│   └── assets/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── README.md
```
