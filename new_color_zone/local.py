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
    # [!!] 0번부터 10번까지 카메라를 체크합니다.
    cap = find_camera(10)

    if cap is None:
        print("오류: 0~10번 인덱스에서 사용 가능한 카메라를 찾지 못했습니다.")
        return # 프로그램 종료

    # 카메라 해상도 정보 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_center_x = frame_width // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("오류: 프레임을 읽을 수 없습니다.")
            break

        # BGR -> HSV 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- 마스크 생성 ---
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # --- 객체 검출 (초록색) ---
        contours_green, _ = cv2.findContours(green_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_green:
            if cv2.contourArea(contour) > min_area_green:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(frame, "Green Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # --- 객체 검출 (빨간색) ---
        contours_red, _ = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_red:
            if cv2.contourArea(contour) > min_area_red:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(frame, "Red Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # 결과 화면에 표시
        cv2.imshow("Result", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()