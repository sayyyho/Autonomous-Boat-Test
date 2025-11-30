"""
KABOAT YOLO 훈련 - data.yaml 자동 수정
valid 경로 문제 해결
"""

from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import torch
import shutil


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
    
    return device, device_name


def fix_data_yaml(dataset_path: str):
    """data.yaml 파일 자동 수정"""
    dataset_path = Path(dataset_path)
    yaml_path = dataset_path / 'data.yaml'
    
    if not yaml_path.exists():
        print(f"❌ data.yaml을 찾을 수 없습니다: {yaml_path}")
        return False
    
    # 백업
    backup_path = dataset_path / 'data.yaml.backup'
    if not backup_path.exists():
        shutil.copy(yaml_path, backup_path)
        print(f"💾 백업 생성: {backup_path}")
    
    # 읽기
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # valid 폴더 존재 여부 확인
    has_valid_folder = (dataset_path / 'valid').exists()
    
    modified = False
    
    # ★★★ 여기가 수정된 부분입니다 ★★★
    # valid 폴더가 없고, data.yaml에 'val' 키도 없으면
    if not has_valid_folder and 'val' not in config:
        print("🔧 data.yaml 수정 중...")
        # 'val' 키를 'train'과 동일하게 *추가*합니다.
        # (YOLO 로더를 통과시키기 위한 트릭)
        config['val'] = config['train'] 
        modified = True
        print(f"   After: 'val: {config['train']}' 추가 (자동 분할 예정)")

    # (참고) 만약 'test' 키가 없다면 'val'과 동일하게 추가
    if 'test' not in config:
        config['test'] = config['val']
        modified = True
        print(f"   After: 'test: {config['val']}' 추가")
    # ★★★ 여기까지 ★★★

    # 저장
    if modified:
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ data.yaml 수정 완료")
    else:
        print(f"✅ data.yaml 수정 불필요")
    
    return True

def check_dataset_structure(dataset_path: str):
    """데이터셋 구조 검증"""
    dataset_path = Path(dataset_path)
    
    print("=" * 60)
    print("📁 데이터셋 구조 확인")
    print("=" * 60)
    
    # 필수 파일/폴더
    required_items = {
        'data.yaml': dataset_path / 'data.yaml',
        'train': dataset_path / 'train',
    }
    
    # 선택 항목
    optional_items = {
        'valid': dataset_path / 'valid',
        'test': dataset_path / 'test',
    }
    
    # 필수 체크
    all_exist = True
    for name, path in required_items.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        all_exist = all_exist and exists
    
    # 선택 체크
    for name, path in optional_items.items():
        exists = path.exists()
        status = "✅" if exists else "⚪"
        print(f"{status} {name}: {path} (선택)")
    
    if not all_exist:
        print("\n⚠️  필수 파일/폴더가 없습니다!")
        return False
    
    # valid 없으면 경고
    has_valid = (dataset_path / 'valid').exists()
    if not has_valid:
        print("\n💡 valid 폴더 없음: train의 일부를 자동 분할")
    
    # 데이터 개수
    try:
        train_images = list((dataset_path / 'train' / 'images').glob('*.jpg')) + \
                      list((dataset_path / 'train' / 'images').glob('*.png'))
        
        print(f"\n📊 데이터 개수:")
        print(f"   Train: {len(train_images)} 이미지")
        
        if has_valid:
            valid_images = list((dataset_path / 'valid' / 'images').glob('*.jpg')) + \
                          list((dataset_path / 'valid' / 'images').glob('*.png'))
            print(f"   Valid: {len(valid_images)} 이미지")
            print(f"   Total: {len(train_images) + len(valid_images)} 이미지")
        else:
            expected_train = int(len(train_images) * 0.8)
            expected_valid = len(train_images) - expected_train
            print(f"   → 훈련용: 약 {expected_train} (80%)")
            print(f"   → 검증용: 약 {expected_valid} (20%)")
        
    except Exception as e:
        print(f"\n⚠️  데이터 확인 오류: {e}")
    
    # data.yaml 내용
    try:
        with open(dataset_path / 'data.yaml', 'r') as f:
            config = yaml.safe_load(f)
            print(f"\n📋 data.yaml:")
            print(f"   클래스 수: {config.get('nc', 'N/A')}")
            print(f"   클래스 이름: {config.get('names', 'N/A')}")
    except Exception as e:
        print(f"\n⚠️  data.yaml 읽기 오류: {e}")
    
    print("=" * 60)
    
    # data.yaml 자동 수정
    if not has_valid:
        fix_data_yaml(dataset_path)
    
    return True


def train_buoy_detector(
    dataset_path: str = './docking',
    model_size: str = 'n',
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,
    project_name: str = 'kaboat_docking',
    device: str = 'auto',
    val_split: float = 0.2
):
    """부표 검출기 훈련"""
    
    # 데이터셋 검증 및 data.yaml 자동 수정
    if not check_dataset_structure(dataset_path):
        return None
    
    # 디바이스 감지
    if device == 'auto':
        device, device_name = detect_device()
    else:
        device_name = device
    
    # CPU 배치 조정
    if device == 'cpu' and batch_size > 8:
        original_batch = batch_size
        batch_size = 8
        print(f"\n⚠️  CPU: 배치 {original_batch} → {batch_size}")
    
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    
    print("\n" + "=" * 60)
    print("🚀 YOLO 훈련 시작")
    print("=" * 60)
    print(f"모델: YOLOv8{model_size}")
    print(f"데이터셋: {dataset_path}")
    print(f"에포크: {epochs}")
    print(f"배치: {batch_size}")
    print(f"디바이스: {device_name}")
    print("=" * 60)
    
    model = YOLO(f'yolov8{model_size}.pt')
    
    try:
        # valid 없으면 split 활성화
        has_valid = Path(dataset_path, 'valid').exists()
        split_val = 0.0 if has_valid else val_split
        
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name=project_name,
            device=device,
            split=split_val,
            patience=50,
            save=True,
            save_period=10,
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
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            cos_lr=True,
            close_mosaic=10,
            verbose=True,
            seed=0,
            deterministic=True,
            workers=4 if device == 'cpu' else 8,
        )
        
        print("\n" + "=" * 60)
        print("✅ 훈련 완료!")
        print("=" * 60)
        print(f"📁 결과: runs/detect/{project_name}/")
        print(f"🏆 best.pt: runs/detect/{project_name}/weights/best.pt")
        print("=" * 60)
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  중단됨")
        return None
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        print("\n💡 data.yaml.backup에서 복원:")
        print(f"   cp {dataset_path}/data.yaml.backup {dataset_path}/data.yaml")
        return None


def validate_trained_model(model_path: str, data_yaml: str):
    """모델 검증"""
    print("\n" + "=" * 60)
    print("📊 모델 검증")
    print("=" * 60)
    
    try:
        model = YOLO(model_path)
        results = model.val(data=data_yaml)
        
        print(f"\n📈 성능:")
        print(f"   mAP50: {results.box.map50:.3f}")
        print(f"   mAP50-95: {results.box.map:.3f}")
        print(f"   Precision: {results.box.p:.3f}")
        print(f"   Recall: {results.box.r:.3f}")
        print("=" * 60)
        
        return results
    except Exception as e:
        print(f"❌ 오류: {e}")
        return None


def test_on_image(model_path: str, image_path: str):
    """이미지 테스트"""
    import cv2
    
    if not os.path.exists(image_path):
        print(f"❌ 이미지 없음: {image_path}")
        return
    
    try:
        model = YOLO(model_path)
        results = model(image_path, conf=0.3)
        
        for r in results:
            print(f"\n🎯 검출: {len(r.boxes)}개")
            for i, box in enumerate(r.boxes):
                cls_name = r.names[int(box.cls[0])]
                conf = float(box.conf[0])
                print(f"   {i+1}. {cls_name}: {conf:.3f}")
            
            img = r.plot()
            cv2.imwrite('result.jpg', img)
            print("\n💾 저장: result.jpg")
        
    except Exception as e:
        print(f"❌ 오류: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./docking')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'validate', 'test', 'check', 'fix'])
    parser.add_argument('--weights', type=str, default='runs/detect/kaboat_docking/weights/best.pt')
    parser.add_argument('--test-image', type=str, default='test.jpg')
    
    args = parser.parse_args()
    
    # 시스템 정보
    print("\n" + "=" * 60)
    print("🖥️  시스템")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print("=" * 60)
    
    # 실행
    if args.mode == 'check':
        check_dataset_structure(args.dataset)
        
    elif args.mode == 'fix':
        # data.yaml만 수정
        fix_data_yaml(args.dataset)
        
    elif args.mode == 'train':
        train_buoy_detector(
            dataset_path=args.dataset,
            model_size=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch,
            device=args.device,
            val_split=args.val_split
        )
        
    elif args.mode == 'validate':
        data_yaml = os.path.join(args.dataset, 'data.yaml')
        validate_trained_model(args.weights, data_yaml)
        
    elif args.mode == 'test':
        test_on_image(args.weights, args.test_image)


if __name__ == '__main__':
    main()