from typing import Dict, List, Tuple
from functools import reduce
import numpy as np
import cv2

HSV_RANGES: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
    'RED': [
        (np.array([0, 150, 120]), np.array([8, 255, 255])),
        (np.array([172, 150, 120]), np.array([180, 255, 255]))
    ],
    'GREEN': [
        (np.array([40,120,120]), np.array([105,255,255])),
    ],
    'YELLOW': [
        (np.array([22, 120, 120]), np.array([32, 255, 255]))
    ]
}

def show_mask(color: str) -> None:
    cap: cv2.VideoCapture = cv2.VideoCapture(1)     # 1 = macbook internal cam
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)       # exposure 안정성 상승

    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            break

        hsv: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        masks: List[np.ndarray] = []
        for (low, high) in HSV_RANGES[color]:
            masks.append(cv2.inRange(hsv, low, high))

        mask: np.ndarray = reduce(cv2.bitwise_or, masks)
        res: np.ndarray = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("orig", frame)
        cv2.imshow(f"{color}_mask", mask)
        cv2.imshow(f"{color}_result", res)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:    # esc
            break

    cap.release()
    cv2.destroyAllWindows()

show_mask("RED")     # 필요시 "GREEN" / "YELLOW" 로 바꾸면 됨
