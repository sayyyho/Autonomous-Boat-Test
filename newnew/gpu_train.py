"""
KABOAT 부표 검출을 위한 YOLO 커스텀 훈련 가이드

1. 데이터셋 구조:
    dataset/
    ├── images/
    │   ├── train/
    │   │   ├── img001.jpg
    │   │   └── ...
    │   └── val/
    │       ├── img050.jpg
    │       └── ...
    └── labels/
        ├── train/
        │   ├── img001.txt
        │   └── ...
        └── val/
            ├── img050.txt
            └── ...

2. Label 형식 (YOLO format):
    class_id center_x center_y width height
    (모든 값은 0~1 사이로 정규화)
    
    예시:
    0 0.5 0.3 0.1 0.15
    1 0.7 0.4 0.12 0.18
"""

import os
import shutil
import yaml
from pathlib import Path


def create_yolo_dataset_structure(base_path: str = './kaboat_dataset'):
    """YOLO 데이터셋 폴더 구조 생성"""
    
    base = Path(base_path)
    
    # 폴더 생성
    folders = [
        base / 'images' / 'train',
        base / 'images' / 'val',
        base / 'labels' / 'train',
        base / 'labels' / 'val'
    ]
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ 데이터셋 폴더 구조 생성 완료: {base_path}")
    
    return base


def create_yolo_config(dataset_path: str, num_classes: int = 1):
    """
    YOLO 훈련용 config 파일 생성
    
    Args:
        dataset_path: 데이터셋 경로
        num_classes: 클래스 수 (1: 부표 통합, 2: 빨강/초록 분리)
    """
    
    if num_classes == 1:
        class_names = ['buoy']
    else:
        class_names = ['red_buoy', 'green_buoy']
    
    config = {
        'path': os.path.abspath(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': num_classes
    }
    
    config_path = os.path.join(dataset_path, 'kaboat.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Config 파일 생성: {config_path}")
    print(f"   클래스: {class_names}")
    
    return config_path


def train_yolo_model(config_path: str, 
                     model_size: str = 'n',  # n, s, m, l, x
                     epochs: int = 100,
                     img_size: int = 640,
                     batch_size: int = 16):
    """
    YOLO 모델 훈련
    
    Args:
        config_path: 데이터셋 config 파일 경로
        model_size: 모델 크기 (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: 훈련 에포크 수
        img_size: 입력 이미지 크기
        batch_size: 배치 크기
    """
    from ultralytics import YOLO
    
    # 사전훈련 모델 로드
    model = YOLO(f'yolov8{model_size}.pt')
    
    # 훈련 실행
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='kaboat_buoy_detection',
        patience=50,  # Early stopping
        save=True,
        device=0,  # GPU 사용 (CPU는 'cpu')
        
        # Augmentation 설정 (해상 환경 고려)
        hsv_h=0.015,      # Hue augmentation
        hsv_s=0.7,        # Saturation augmentation
        hsv_v=0.4,        # Value augmentation
        degrees=10,       # 회전
        translate=0.1,    # 이동
        scale=0.5,        # 스케일
        shear=0.0,        # 전단
        perspective=0.0,  # 원근
        flipud=0.0,       # 상하반전 (해상에서는 불필요)
        fliplr=0.5,       # 좌우반전
        mosaic=1.0,       # Mosaic augmentation
        mixup=0.0,        # Mixup augmentation
    )
    
    print("✅ 훈련 완료!")
    print(f"   최고 모델: runs/detect/kaboat_buoy_detection/weights/best.pt")
    
    return results


def evaluate_model(model_path: str, test_image_path: str):
    """
    훈련된 모델 평가
    
    Args:
        model_path: 훈련된 모델 경로
        test_image_path: 테스트 이미지 경로
    """
    from ultralytics import YOLO
    import cv2
    
    model = YOLO(model_path)
    
    # 추론
    results = model(test_image_path)
    
    # 결과 시각화
    for result in results:
        img = result.plot()
        cv2.imshow('Detection Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 성능 지표 출력
    print("\n📊 모델 성능:")
    print(f"   검출된 객체 수: {len(results[0].boxes)}")
    for box in results[0].boxes:
        print(f"   Class: {int(box.cls[0])}, Confidence: {float(box.conf[0]):.3f}")


# ========================================
# 데이터 수집 가이드
# ========================================

ANNOTATION_GUIDE = """
# KABOAT 부표 데이터 수집 및 라벨링 가이드

## 1. 데이터 수집 전략

### 촬영 조건 다양화
- ✅ 시간대: 오전, 오후, 저녁 (조명 변화)
- ✅ 날씨: 맑음, 흐림, 안개 (가시성 변화)
- ✅ 파도: 잔잔함, 중간, 거친 파도 (흔들림)
- ✅ 거리: 5m ~ 50m (다양한 스케일)
- ✅ 각도: 정면, 측면, 비스듬히 (다양한 시점)

### 권장 이미지 수
- 최소: 500장 (train 400, val 100)
- 권장: 1000장 이상
- 이상적: 2000장 이상

## 2. 라벨링 도구

### Roboflow (추천)
1. https://roboflow.com 회원가입
2. 프로젝트 생성: "KABOAT Buoy Detection"
3. 이미지 업로드
4. Bounding Box 그리기
5. YOLO 형식으로 Export

### LabelImg (무료 오픈소스)
```bash
pip install labelImg
labelImg
```

### CVAT (온라인/로컬)
https://www.cvat.ai

## 3. 라벨링 주의사항

### Bounding Box 그리기 원칙
✅ 부표 전체를 포함 (물에 잠긴 부분 포함)
✅ 여백 최소화 (tight bbox)
✅ 가려진 부표도 보이는 부분만 표시
✅ 흐릿한 부표는 제외

### 클래스 전략

#### 옵션 1: 통합 클래스 (추천)
- class 0: buoy (모든 부표)
- 장점: 데이터 적어도 학습 가능, 빠른 검출
- 단점: 색상은 HSV 필터로 후처리 필요

#### 옵션 2: 색상별 분리
- class 0: red_buoy
- class 1: green_buoy
- 장점: 색상 검증 불필요
- 단점: 데이터 2배 필요, 색상 구분 실수 가능

## 4. 데이터 증강 (Augmentation)

YOLO 훈련 시 자동 적용되는 증강:
- Brightness/Contrast 조정
- Hue/Saturation 변경
- 회전, 이동, 스케일
- Mosaic (4장 합성)

추가 증강 (필요 시):
- 비 효과 시뮬레이션
- 렌즈 왜곡 보정
- 노이즈 추가

## 5. 훈련 팁

### 하이퍼파라미터 최적화
- epochs: 100~200 (early stopping 사용)
- batch_size: GPU 메모리에 맞춰 조정
- img_size: 640 (실시간), 1280 (정확도 우선)

### Transfer Learning
- 사전훈련 모델 사용 (COCO dataset)
- Fine-tuning으로 빠른 수렴

### 실전 테스트
- 실제 경기장 환경에서 검증
- FPS 측정 (목표: 20+ fps)
- 오검출/미검출 분석

## 6. 성능 개선 전략

### 낮은 정확도 시
1. 데이터 추가 수집 (특히 실패 케이스)
2. 라벨링 재검토 (일관성 확인)
3. 모델 크기 증가 (n → s → m)
4. 훈련 에포크 증가

### 낮은 FPS 시
1. 모델 경량화 (m → s → n)
2. 입력 이미지 크기 감소 (640 → 480)
3. TensorRT 최적화 (GPU)
4. ONNX 변환 (추론 가속)
"""


def main():
    """실행 예시"""
    print("=" * 60)
    print("KABOAT YOLO 훈련 파이프라인")
    print("=" * 60)
    
    # 1. 데이터셋 구조 생성
    dataset_path = create_yolo_dataset_structure('./kaboat_dataset')
    
    # 2. Config 파일 생성
    config_path = create_yolo_config(str(dataset_path), num_classes=1)
    
    print("\n" + "=" * 60)
    print("다음 단계:")
    print("=" * 60)
    print("1. 데이터 수집 및 라벨링")
    print("   - kaboat_dataset/images/train/ 에 이미지 추가")
    print("   - kaboat_dataset/labels/train/ 에 라벨 추가")
    print()
    print("2. 훈련 실행")
    print("   python yolo_training_guide.py --train")
    print()
    print("3. 모델 평가")
    print("   python yolo_training_guide.py --eval")
    print("=" * 60)
    
    # 가이드 저장
    with open('ANNOTATION_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(ANNOTATION_GUIDE)
    print("\n📖 라벨링 가이드 저장: ANNOTATION_GUIDE.md")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='모델 훈련 시작')
    parser.add_argument('--eval', action='store_true', help='모델 평가')
    parser.add_argument('--config', type=str, default='./kaboat_dataset/kaboat.yaml')
    parser.add_argument('--model', type=str, default='runs/detect/kaboat_buoy_detection/weights/best.pt')
    parser.add_argument('--test-image', type=str, default='test.jpg')
    
    args = parser.parse_args()
    
    if args.train:
        print("🚀 훈련 시작...")
        train_yolo_model(args.config)
    elif args.eval:
        print("📊 평가 시작...")
        evaluate_model(args.model, args.test_image)
    else:
        main()