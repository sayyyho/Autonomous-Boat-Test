import cv2
import numpy as np

def find_camera(max_index=10):
    """
    0번부터 max_index(기본 10)까지 카메라를 확인하고,
    사용 가능한 첫 번째 cap 객체를 반환합니다.
    """
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        
        # 1. isOpened()로 카메라가 열렸는지 확인
        if cap.isOpened():
            # 2. read()로 프레임을 실제로 읽을 수 있는지 확인
            ret, _ = cap.read()
            if ret:
                print(f"✅ 카메라 찾음! 인덱스 {i}번을 사용합니다.")
                return cap  # 성공한 cap 객체 반환
            else:
                print(f"❌ 인덱스 {i}번: 열렸으나 프레임 읽기 실패.")
                cap.release()
        else:
            print(f"❌ 인덱스 {i}번: 열기 실패.")
            cap.release()
            
    return None # 10번까지 모두 실패

# --- [신규] 가장 큰 객체를 찾는 헬퍼 함수 ---
def find_largest_contour(contours, min_area):
    """컨투어 리스트에서 최소 면적을 넘는 가장 큰 객체의 바운딩 박스를 반환합니다."""
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area > min_area:
        return cv2.boundingRect(largest_contour)
    
    return None

# --- 색상 범위 (아이폰 스크립트 기준) ---
lower_green = np.array([72, 120, 90])
upper_green = np.array([92, 255, 255])

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([165, 100, 100])
upper_red2 = np.array([180, 255, 255])

# --- 객체 최소 면적 ---
min_area_green = 500
min_area_red = 500

# --- 메인 코드 시작 ---
def main():
    cap = find_camera(10)

    if cap is None:
        print("오류: 0~10번 인덱스에서 사용 가능한 카메라를 찾지 못했습니다.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_center_x = frame_width // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("오류: 프레임을 읽을 수 없습니다.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- 마스크 생성 ---
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # --- [수정] 노이즈 제거 (ROS 코드와 유사하게) ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # --- [수정] 가장 큰 객체 찾기 ---
        contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        green_bb = find_largest_contour(contours_green, min_area_green)
        red_bb = find_largest_contour(contours_red, min_area_red)

        green_cx = None
        red_cx = None

        # --- [신규] 게이트 로직 ---
        
        # 1. 감지된 객체 정보 업데이트 및 그리기
        if green_bb:
            x, y, w, h = green_bb
            green_cx = x + w // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, "Green", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if red_bb:
            x, y, w, h = red_bb
            red_cx = x + w // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, "Red", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 2. 게이트 상태 판단 및 중점 계산
        status_text = ""
        if green_cx and red_cx:
            # 2a. 둘 다 감지됨 (좌=초, 우=빨 규칙 확인)
            if green_cx < red_cx:
                midpoint_x = (green_cx + red_cx) // 2
                status_text = "GATE DETECTED"
                # 중점 라인 그리기
                cv2.line(frame, (midpoint_x, 0), (midpoint_x, frame_height), (255, 255, 0), 3)
                cv2.putText(frame, "TARGET", (midpoint_x - 40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            else:
                status_text = "Invalid Pair (Red-Green)"
                
        elif green_cx:
            # 2b. 초록색만 감지됨
            if green_cx < frame_center_x:
                status_text = "Scan Right for Red"
            else:
                status_text = "Invalid Green (Position Error)"
                
        elif red_cx:
            # 2c. 빨간색만 감지됨
            if red_cx > frame_center_x:
                status_text = "Scan Left for Green"
            else:
                status_text = "Invalid Red (Position Error)"
                
        else:
            # 2d. 아무것도 감지되지 않음
            status_text = "Searching for Gate..."

        # 상태 메시지 표시
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        # 결과 화면에 표시
        cv2.imshow("Result", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()