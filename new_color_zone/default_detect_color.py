import cv2
import numpy as np

# 웹캠 초기화
cap = cv2.VideoCapture(0)

# --- HSV 범위 정의 ---

lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

lower_red = np.array([0, 120, 100])      
upper_red = np.array([5, 255, 255])


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR -> HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- 마스크 생성 ---
    # 1. 초록색 마스크
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 2. 빨간색 마스크 (두 범위를 합침)
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    # --- 객체 검출 (초록색) ---
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_green:
        if cv2.contourArea(contour) > 1000:  # 최소 면적 설정
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(frame, "Green Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # --- 객체 검출 (빨간색) ---
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours_red:
        if cv2.contourArea(contour) > 1500:  # 최소 면적 설정
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(frame, "Red Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # 결과 화면에 표시
    cv2.imshow("Result", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()