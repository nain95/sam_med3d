import torch
import random
import numpy as np
import os
import sys
from config import Config
from dataloader import get_sam_med3d_dataloader
from model import SAMFineTuner
from trainer import train_model

def set_seed(seed=42):
    """재현 가능한 결과를 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_data_availability(config):
    """데이터 파일 존재 확인"""
    if not os.path.exists(config.train_csv):
        print(f"❌ 훈련 데이터 파일이 없습니다: {config.train_csv}")
        return False
    
    if not os.path.exists(config.val_csv):
        print(f"❌ 검증 데이터 파일이 없습니다: {config.val_csv}")
        return False
    
    print(f"✅ 데이터 파일 확인 완료")
    print(f"   Train: {config.train_csv}")
    print(f"   Val: {config.val_csv}")
    return True

def print_system_info():
    """시스템 정보 출력"""
    print("🖥️ 시스템 정보:")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"           Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

def main():
    """메인 함수"""
    print("=" * 60)
    print("🧠 SAM-Med3D Fine-tuning for Brain CT ICH Segmentation")
    print("=" * 60)
    
    # 시스템 정보 출력
    print_system_info()
    print()
    
    # Configuration 로드
    config = Config.from_args()
    
    # 시드 설정
    set_seed(config.seed)
    print(f"🌱 시드 설정: {config.seed}")
    
    # 데이터 파일 확인
    if not check_data_availability(config):
        return
    
    # 설정 정보 출력
    print(f"\n📊 훈련 설정:")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Input size: {config.input_size}")
    print(f"   Visual tokens: {config.visual_tokens}")
    print(f"   Device: {config.device}")
    print(f"   Output dir: {config.output_dir}")
    
    if config.use_wandb:
        print(f"\n📈 WandB 설정:")
        print(f"   Project: {config.wandb_project}")
        print(f"   Entity: {config.wandb_entity}")
        print(f"   Run name: {config.wandb_run_name}")
        print(f"   Tags: {config.wandb_tags}")
        print(f"   Log images: {config.log_images}")
        print(f"   Log frequency: {config.log_freq} epochs")
    else:
        print(f"\n📈 WandB: 사용하지 않음")
    
    try:
        # 데이터로더 생성
        print(f"\n📁 데이터로더 생성 중...")
        train_loader = get_sam_med3d_dataloader(
            csv_path=config.train_csv, 
            batch_size=config.batch_size, 
            is_train=True, 
            target_size=config.input_size,
            verbose=False
        )
        
        val_loader = get_sam_med3d_dataloader(
            csv_path=config.val_csv, 
            batch_size=config.batch_size, 
            is_train=False, 
            target_size=config.input_size,
            verbose=False
        )
        
        print(f"✅ 데이터로더 생성 완료")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val samples: {len(val_loader.dataset)}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        # 샘플 배치 확인
        print(f"\n🔍 데이터 형태 확인...")
        for batch in train_loader:
            if batch is not None and batch["patient_id"][0] != "dummy":
                print(f"   Image shape: {batch['image'].shape}")
                print(f"   Mask shape: {batch['mask'].shape}")
                print(f"   Patient ID 예시: {batch['patient_id'][0]}")
                break
        
        # 모델 생성
        print(f"\n🤖 모델 생성 중...")
        model = SAMFineTuner(
            checkpoint_path=None,  # SAM-Med3D-turbo 사용
            representation_dim=config.visual_tokens,
            freeze_encoder=False
        )
        
        # 모델 파라미터 정보
        param_info = model.get_trainable_params()
        print(f"✅ 모델 생성 완료")
        print(f"   전체 파라미터: {param_info['total_params']:,}")
        print(f"   학습 가능: {param_info['trainable_params']:,}")
        print(f"   학습 가능 비율: {param_info['trainable_ratio']:.1%}")
        
        # GPU 메모리 확인
        if torch.cuda.is_available():
            model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
            print(f"   예상 모델 크기: {model_size_mb:.1f} MB")
            
            # GPU 메모리 상태
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU 메모리: {gpu_memory:.1f} GB")
        
        # 훈련 시작
        print(f"\n🚀 훈련 시작!")
        print("=" * 60)
        
        trained_model = train_model(model, train_loader, val_loader, config)
        
        # 훈련 완료
        print("\n" + "=" * 60)
        print("🎉 훈련 완료!")
        print("=" * 60)
        
        # 저장된 파일 확인
        saved_files = []
        if os.path.exists(os.path.join(config.output_dir, 'best_model.pth')):
            saved_files.append("best_model.pth")
        if os.path.exists(os.path.join(config.output_dir, 'sam_encoder.pth')):
            saved_files.append("sam_encoder.pth")
        
        print(f"💾 저장된 파일:")
        for file in saved_files:
            file_path = os.path.join(config.output_dir, file)
            file_size = os.path.getsize(file_path) / 1024 / 1024
            print(f"   {file} ({file_size:.1f} MB)")
        
        print(f"\n📁 출력 디렉토리: {config.output_dir}")
        
        if config.use_wandb:
            print(f"📈 WandB 링크에서 상세 결과를 확인하세요!")
        
        print(f"\n🔥 BLIP2 통합을 위해 다음 파일을 사용하세요:")
        print(f"   {os.path.join(config.output_dir, 'sam_encoder.pth')}")
        
        return trained_model
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자에 의해 훈련이 중단되었습니다.")
        return None
        
    except Exception as e:
        print(f"\n❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()