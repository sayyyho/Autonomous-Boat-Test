
#include <Wire.h>
#include "MPU9250.h" // SparkFun MPU-9250 Breakout 라이브러리 사용
#include <TinyGPS++.h>
#include <SoftwareSerial.h>

// MPU9250 객체 생성
MPU9250 imu;

// GPS 객체 생성
TinyGPSPlus gps;

// GPS 모듈을 위한 SoftwareSerial 설정
// 아두이노의 4번 핀을 GPS의 TX에, 3번 핀을 GPS의 RX에 연결하세요.
SoftwareSerial gpsSerial(4, 3); // RX, TX

void printIMUData() {
    Serial.print("IMU,");
    Serial.print(imu.getAccelX_mss(), 4); Serial.print(",");
    Serial.print(imu.getAccelY_mss(), 4); Serial.print(",");
    Serial.print(imu.getAccelZ_mss(), 4); Serial.print(",");
    Serial.print(imu.getGyroX_rads(), 4); Serial.print(",");
    Serial.print(imu.getGyroY_rads(), 4); Serial.print(",");
    Serial.print(imu.getGyroZ_rads(), 4); Serial.print(",");
    Serial.print(imu.getMagX_uT(), 4);    Serial.print(",");
    Serial.print(imu.getMagY_uT(), 4);    Serial.print(",");
    Serial.print(imu.getMagZ_uT(), 4);
}

void printGPSData() {
    Serial.print(",GPS,");
    if (gps.location.isValid()) {
        Serial.print(gps.location.lat(), 6); Serial.print(",");
        Serial.print(gps.location.lng(), 6); Serial.print(",");
        Serial.print(gps.altitude.meters(), 2); Serial.print(",");
        Serial.print(gps.satellites.value());
    } else {
        Serial.print("0.0,0.0,0.0,0");
    }
}

void setup() {
  // 라즈베리파이와 통신을 위한 시리얼 시작 (전송 속도: 115200)
  Serial.begin(115200);

  // I2C 통신 시작
  Wire.begin();

  // MPU-9250 초기화
  if (imu.begin() != INV_SUCCESS) {
    while (1) {
      Serial.println("IMU connection failed. Check wiring.");
      delay(5000);
    }
  }
  
  // GPS 모듈과의 통신을 위한 시리얼 시작 (전송 속도: 9600)
  gpsSerial.begin(9600);
  
  Serial.println("Sensor node setup complete. Starting data transmission...");
}

void loop() {
  // IMU 데이터 읽기
  if (imu.dataReady()) {
    imu.updateSensors();

    // GPS 데이터 읽기
    while (gpsSerial.available() > 0) {
      gps.encode(gpsSerial.read());
    }

    // 통합된 데이터 전송
    printIMUData();
    printGPSData();
    Serial.println(); // End of line
  }
  
  // 약 100Hz로 데이터 전송 시도 (IMU의 dataReady()가 속도를 조절)
  delay(10);
}
