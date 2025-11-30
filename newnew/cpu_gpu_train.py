"""
KABOAT YOLO 훈련 - GPU 자동 감지 버전
GPU 없으면 자동으로 CPU로 전환
"""

from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import torch


def detect_device():
    """사용 가능한 디바이스 자동 감지"""
    if torch.cuda.is_available():
        device = '0'
        device_name = f"GPU ({torch.cuda.get_device_name(0)})"
        print(f"✅ GPU 감지: {device_name}")
    else:
        device = 'cpu'
        device_name = "CPU"
        print(f"⚠️  GPU 없음. CPU 모드로 실행")
        print(f"   💡 팁: GPU가 있다면 훈련 속도가 10배 이상 빨라집니다!")
    
    return device, device_name


def check_dataset_structure(dataset_path: str):
    """데이터셋 구조 검증"""
    dataset_path = Path(dataset_path)
    
    print("=" * 60)
    print("📁 데이터셋 구조 확인")
    print("=" * 60)
    
    # 필수 파일/폴더 확인
    required_items = {
        'data.yaml': dataset_path / 'data.yaml',
        'train': dataset_path / 'train',
        'valid': dataset_path / 'valid',
    }
    
    all_exist = True
    for name, path in required_items.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n⚠️  필수 파일/폴더가 없습니다!")
        return False
    
    # 데이터 개수 확인
    try:
        train_images = list((dataset_path / 'train' / 'images').glob('*.jpg')) + \
                      list((dataset_path / 'train' / 'images').glob('*.png'))
        valid_images = list((dataset_path / 'valid' / 'images').glob('*.jpg')) + \
                      list((dataset_path / 'valid' / 'images').glob('*.png'))
        
        print(f"\n📊 데이터 개수:")
        print(f"   Train: {len(train_images)} 이미지")
        print(f"   Valid: {len(valid_images)} 이미지")
        print(f"   Total: {len(train_images) + len(valid_images)} 이미지")
        
        if len(train_images) < 10:
            print(f"\n⚠️  훈련 데이터가 너무 적습니다!")
            print(f"   현재: {len(train_images)}장")
            print(f"   권장: 최소 50장, 이상적으로는 200장 이상")
        elif len(train_images) < 50:
            print(f"\n⚠️  데이터가 부족합니다. 정확도가 낮을 수 있습니다.")
            print(f"   현재: {len(train_images)}장")
            print(f"   권장: 200장 이상")
        else:
            print(f"\n✅ 충분한 데이터!")
        
    except Exception as e:
        print(f"\n⚠️  데이터 확인 중 오류: {e}")
    
    # data.yaml 내용 확인
    try:
        with open(dataset_path / 'data.yaml', 'r') as f:
            config = yaml.safe_load(f)
            print(f"\n📋 data.yaml 내용:")
            print(f"   클래스 수: {config.get('nc', 'N/A')}")
            print(f"   클래스 이름: {config.get('names', 'N/A')}")
    except Exception as e:
        print(f"\n⚠️  data.yaml 읽기 오류: {e}")
    
    print("=" * 60)
    return True


def train_buoy_detector(
    dataset_path: str = './docking',
    model_size: str = 'n',
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,
    project_name: str = 'kaboat_red_buoy',
    device: str = 'auto'  # 'auto', '0', 'cpu'
):
    """
    빨간 부표 검출기 훈련
    """
    
    # 데이터셋 검증
    if not check_dataset_structure(dataset_path):
        return None
    
    # 디바이스 자동 감지
    if device == 'auto':
        device, device_name = detect_device()
    else:
        device_name = device
    
    # CPU 사용 시 배치 크기 자동 조정
    if device == 'cpu' and batch_size > 8:
        original_batch = batch_size
        batch_size = 8
        print(f"\n⚠️  CPU 모드: 배치 크기 자동 조정 ({original_batch} → {batch_size})")
    
    # CPU 사용 시 추천 설정
    if device == 'cpu':
        print("\n💡 CPU 최적화 팁:")
        print("   - 작은 모델 사용 (n 권장)")
        print("   - 배치 크기 8 이하")
        print("   - 이미지 크기 640 또는 480")
        print("   - 훈련 시간: GPU 대비 10-20배 느림")
        print("   - 예상 시간: 에포크당 5-10분 (39장 기준)\n")
    
    # data.yaml 경로
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    
    print("\n" + "=" * 60)
    print("🚀 YOLO 훈련 시작")
    print("=" * 60)
    print(f"모델: YOLOv8{model_size}")
    print(f"데이터셋: {dataset_path}")
    print(f"에포크: {epochs}")
    print(f"이미지 크기: {img_size}")
    print(f"배치 크기: {batch_size}")
    print(f"디바이스: {device_name}")
    print("=" * 60)
    
    # 사전훈련 모델 로드
    model = YOLO(f'yolov8{model_size}.pt')
    
    # 훈련 시작
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name=project_name,
            device=device,
            
            # 최적화 설정
            patience=50,
            save=True,
            save_period=10,
            
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            
            # 성능 설정
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            
            # 기타
            cos_lr=True,
            close_mosaic=10,
            verbose=True,
            seed=0,
            deterministic=True,
            
            # CPU 최적화
            workers=4 if device == 'cpu' else 8,
        )
        
        print("\n" + "=" * 60)
        print("✅ 훈련 완료!")
        print("=" * 60)
        print(f"📁 결과 저장: runs/detect/{project_name}/")
        print(f"🏆 최고 모델: runs/detect/{project_name}/weights/best.pt")
        print(f"📊 마지막 모델: runs/detect/{project_name}/weights/last.pt")
        print(f"📈 훈련 그래프: runs/detect/{project_name}/results.png")
        print("=" * 60)
        
        # 다음 단계 안내
        print("\n🎯 다음 단계:")
        print("1. 결과 확인:")
        print(f"   - 훈련 그래프 보기: runs/detect/{project_name}/results.png")
        print(f"   - Confusion Matrix: runs/detect/{project_name}/confusion_matrix.png")
        print()
        print("2. 모델 테스트:")
        print(f"   python {__file__} --mode test \\")
        print(f"       --weights runs/detect/{project_name}/weights/best.pt \\")
        print(f"       --test-image ./test.jpg")
        print()
        print("3. 메인 시스템에 통합:")
        print("   gate_navigation_system.py 에서 모델 경로 수정")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 훈련을 중단했습니다.")
        print("   부분 저장된 모델: runs/detect/{project_name}/weights/last.pt")
        return None
        
    except Exception as e:
        print(f"\n❌ 훈련 중 오류 발생: {e}")
        
        # 일반적인 오류 해결법
        print("\n🔧 문제 해결:")
        if "CUDA" in str(e) or "GPU" in str(e):
            print("   GPU 오류:")
            print("   → --device cpu 옵션 사용")
        elif "memory" in str(e).lower():
            print("   메모리 부족:")
            print("   → --batch 를 4 또는 2로 줄이기")
            print("   → --img-size 를 480 또는 320으로 줄이기")
        elif "data" in str(e).lower():
            print("   데이터셋 오류:")
            print("   → data.yaml 경로 확인")
            print("   → 이미지/라벨 파일 확인")
        
        return None


def validate_trained_model(model_path: str, data_yaml: str):
    """훈련된 모델 검증"""
    print("\n" + "=" * 60)
    print("📊 모델 검증")
    print("=" * 60)
    
    try:
        model = YOLO(model_path)
        results = model.val(data=data_yaml)
        
        print(f"\n📈 성능 지표:")
        print(f"   mAP50: {results.box.map50:.3f} (0.5 IoU)")
        print(f"   mAP50-95: {results.box.map:.3f} (0.5-0.95 IoU)")
        print(f"   Precision: {results.box.p:.3f}")
        print(f"   Recall: {results.box.r:.3f}")
        
        print(f"\n💡 해석:")
        map50 = results.box.map50
        if map50 >= 0.9:
            print("   🌟 훌륭함! 실전 배포 가능")
        elif map50 >= 0.7:
            print("   ✅ 양호함. 실전 테스트 필요")
        elif map50 >= 0.5:
            print("   ⚠️  개선 필요. 데이터 추가 또는 훈련 연장")
        else:
            print("   ❌ 성능 부족. 데이터셋 점검 필요")
        
        print("=" * 60)
        return results
        
    except Exception as e:
        print(f"❌ 검증 오류: {e}")
        return None


def test_on_image(model_path: str, image_path: str):
    """테스트 이미지로 추론"""
    import cv2
    
    print("\n" + "=" * 60)
    print("🖼️  테스트 이미지 추론")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    try:
        model = YOLO(model_path)
        
        # 추론
        results = model(image_path, conf=0.5)
        
        # 결과 출력
        for r in results:
            print(f"\n🎯 검출 결과:")
            print(f"   총 {len(r.boxes)} 개 부표 검출")
            
            if len(r.boxes) == 0:
                print("   ⚠️  검출된 부표가 없습니다")
                print("   💡 원인: 신뢰도 임계값(0.5)이 너무 높거나, 모델 성능 부족")
            else:
                for i, box in enumerate(r.boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = r.names[cls]
                    print(f"   {i+1}. {class_name}: {conf:.3f}")
            
            # 시각화
            img = r.plot()
            cv2.imshow('Detection Result', img)
            print("\n⌨️  아무 키나 누르면 종료...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 추론 오류: {e}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='KABOAT YOLO 훈련 (GPU 자동 감지)')
    
    # 기본 설정
    parser.add_argument('--dataset', type=str, default='./docking',
                       help='데이터셋 경로')
    parser.add_argument('--model', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='모델 크기')
    
    # 훈련 파라미터
    parser.add_argument('--epochs', type=int, default=100,
                       help='훈련 에포크 수')
    parser.add_argument('--img-size', type=int, default=640,
                       help='입력 이미지 크기')
    parser.add_argument('--batch', type=int, default=16,
                       help='배치 크기')
    parser.add_argument('--device', type=str, default='auto',
                       help='디바이스 (auto=자동감지, 0=GPU, cpu=CPU)')
    
    # 실행 모드
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'validate', 'test', 'check'],
                       help='실행 모드')
    
    # 추가 옵션
    parser.add_argument('--weights', type=str, 
                       default='runs/detect/kaboat_red_buoy/weights/best.pt',
                       help='검증/테스트용 모델 경로')
    parser.add_argument('--test-image', type=str, default='test.jpg',
                       help='테스트 이미지 경로')
    
    args = parser.parse_args()
    
    # 환경 정보 출력
    print("\n" + "=" * 60)
    print("🖥️  시스템 정보")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # 모드별 실행
    if args.mode == 'check':
        check_dataset_structure(args.dataset)
        
    elif args.mode == 'train':
        train_buoy_detector(
            dataset_path=args.dataset,
            model_size=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch,
            device=args.device
        )
        
    elif args.mode == 'validate':
        data_yaml = os.path.join(args.dataset, 'data.yaml')
        validate_trained_model(args.weights, data_yaml)
        
    elif args.mode == 'test':
        test_on_image(args.weights, args.test_image)


if __name__ == '__main__':
    main()