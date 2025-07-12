import torch
import random
import numpy as np
import os
import sys
import time
from pathlib import Path

from config import Config
from dataloader import get_sam_med3d_dataloader
from model import SAMFineTuner
from trainer import train_model
from utils import (
    set_seed, validate_data_paths, validate_system_requirements,
    log_gpu_memory, estimate_training_time, save_training_summary,
    print_model_info
)

def print_banner():
    """배너 출력"""
    print("\n" + "="*80)
    print("🧠 SAM-Med3D Fine-tuning for Brain CT ICH Segmentation")
    print("   Enhanced Version with Advanced Features")
    print("="*80)

def pre_training_checks(config):
    """훈련 전 검사"""
    print("\n🔍 훈련 전 시스템 검사...")
    
    # 1. 시스템 요구사항 검사
    if not validate_system_requirements():
        print("❌ 시스템 요구사항을 만족하지 않습니다.")
        return False
    
    # 2. 데이터 경로 검사
    if not validate_data_paths(config):
        print("❌ 데이터 경로 검증에 실패했습니다.")
        return False
    
    # 3. 설정 검증
    if not config.validate_config():
        print("❌ 설정 검증에 실패했습니다.")
        return False
    
    print("✅ 모든 사전 검사를 통과했습니다.")
    return True

def main():
    """개선된 메인 함수"""
    # 배너 출력
    print_banner()
    
    try:
        # Configuration 로드
        print("\n⚙️ 설정 로드 중...")
        config = Config.from_args()
        
        # 설정 출력
        print(config)
        
        # 설정 저장
        config.save_config()
        
        # 시드 설정
        set_seed(config.seed)
        
        # 사전 검사
        if not pre_training_checks(config):
            print("\n❌ 사전 검사 실패. 훈련을 중단합니다.")
            return None
        
        # GPU 메모리 초기 상태
        initial_memory = log_gpu_memory()
        
        # 훈련 시간 추정
        print("\n⏱️ 훈련 시간 추정 중...")
        time_estimate = estimate_training_time(config)
        
        try:
            # 데이터로더 생성
            print(f"\n📁 데이터로더 생성 중...")
            
            train_loader = get_sam_med3d_dataloader(
                csv_path=config.train_csv,
                mask_dir=config.mask_dir,
                batch_size=config.batch_size, 
                is_train=True, 
                target_size=config.input_size,
                channel_method=config.channel_method,
                window_level=config.window_level,
                window_width=config.window_width,
                mask_rotation=config.mask_rotation,
                augmentation_prob=0.3 if config.batch_size >= 2 else 0.1,  # 배치 크기에 따라 조정
                num_workers=config.num_workers,
                verbose=True
            )
            
            val_loader = get_sam_med3d_dataloader(
                csv_path=config.val_csv,
                mask_dir=config.mask_dir,
                batch_size=config.batch_size, 
                is_train=False, 
                target_size=config.input_size,
                channel_method=config.channel_method,
                window_level=config.window_level,
                window_width=config.window_width,
                mask_rotation=config.mask_rotation,
                augmentation_prob=0.0,  # 검증시 증강 없음
                num_workers=min(config.num_workers, 2),  # 검증시 워커 수 줄임
                verbose=True
            )
            
            print(f"\n✅ 데이터로더 생성 완료")
            print(f"   훈련 샘플: {len(train_loader.dataset)}")
            print(f"   검증 샘플: {len(val_loader.dataset)}")
            print(f"   훈련 배치: {len(train_loader)}")
            print(f"   검증 배치: {len(val_loader)}")
            
            # 데이터 형태 확인
            print(f"\n🔍 데이터 샘플 확인...")
            sample_found = False
            for batch_idx, batch in enumerate(train_loader):
                if batch is not None and batch["patient_id"][0] != "dummy":
                    print(f"   ✅ 샘플 발견 (배치 {batch_idx+1})")
                    print(f"      Image shape: {batch['image'].shape}")
                    print(f"      Mask shape: {batch['mask'].shape}")
                    print(f"      Patient ID: {batch['patient_id'][0]}")
                    print(f"      Positive ratio: {batch['positive_ratio'][0]:.3%}")
                    sample_found = True
                    break
                elif batch_idx >= 5:  # 최대 5개 배치까지만 확인
                    break
            
            if not sample_found:
                print("   ⚠️ 유효한 샘플을 찾지 못했습니다.")
                print("   데이터 경로와 마스크 디렉토리를 확인해주세요.")
                return None
            
        except Exception as e:
            print(f"\n❌ 데이터로더 생성 실패: {e}")
            return None
        
        try:
            # 모델 생성
            print(f"\n🤖 모델 생성 중...")
            model = SAMFineTuner(
                checkpoint_path=None,  # SAM-Med3D-turbo 사용
                representation_dim=config.visual_tokens,
                freeze_encoder=config.freeze_encoder,
                use_gradient_checkpointing=config.use_gradient_checkpointing
            )
            
            # 모델 정보 출력
            print_model_info(model, config)
            
            # 모델 로드 후 메모리 상태
            model_memory = log_gpu_memory()
            
        except Exception as e:
            print(f"\n❌ 모델 생성 실패: {e}")
            return None
        
        # 훈련 시작
        print(f"\n🚀 훈련 시작!")
        print("="*80)
        
        training_start_time = time.time()
        
        try:
            results = train_model(model, train_loader, val_loader, config)
            
            if results is None:
                print("\n❌ 훈련이 실패하거나 중단되었습니다.")
                return None
            
            training_end_time = time.time()
            total_training_time = training_end_time - training_start_time
            
            # 훈련 완료
            print("\n" + "="*80)
            print("🎉 훈련 완료!")
            print("="*80)
            
            # 결과 요약
            summary = results.get('summary', {})
            print(f"\n📊 최종 결과:")
            print(f"   최고 Dice: {summary.get('best_dice', 0):.4f}")
            print(f"   최고 IoU: {summary.get('best_iou', 0):.4f}")
            print(f"   총 훈련 시간: {total_training_time/3600:.2f}시간")
            print(f"   완료된 에포크: {summary.get('total_epochs', 0)}")
            
            # 저장된 파일 확인
            output_files = []
            for filename in ['best_model.pth', 'sam_encoder.pth', 'latest_model.pth']:
                filepath = Path(config.output_dir) / filename
                if filepath.exists():
                    file_size = filepath.stat().st_size / (1024*1024)
                    output_files.append(f"{filename} ({file_size:.1f} MB)")
            
            if output_files:
                print(f"\n💾 저장된 파일:")
                for file_info in output_files:
                    print(f"   {file_info}")
            
            print(f"\n📁 출력 디렉토리: {config.output_dir}")
            
            # 훈련 요약 저장
            try:
                summary_file = save_training_summary(config, results)
                if summary_file:
                    print(f"📋 훈련 요약 저장: {summary_file}")
            except Exception as e:
                print(f"⚠️ 훈련 요약 저장 실패: {e}")
            
            # BLIP2 통합 안내
            sam_encoder_path = Path(config.output_dir) / 'sam_encoder.pth'
            if sam_encoder_path.exists():
                print(f"\n🔗 BLIP2 통합용 SAM encoder:")
                print(f"   {sam_encoder_path}")
                print(f"   이 파일을 BLIP2 모델에서 사용하세요.")
            
            # WandB 링크
            if config.use_wandb:
                print(f"\n📈 상세 결과는 WandB에서 확인하세요!")
            
            return results
            
        except Exception as e:
            print(f"\n❌ 훈련 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자에 의해 중단되었습니다.")
        return None
        
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # 정리 작업
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n🧹 정리 완료")


def quick_test():
    """빠른 테스트 모드"""
    print("🧪 빠른 테스트 모드")
    
    # 임시 설정
    class TestConfig:
        def __init__(self):
            self.seed = 42
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.input_size = (128, 128, 128)
            self.visual_tokens = 256
            self.freeze_encoder = False
            self.use_gradient_checkpointing = True
    
    config = TestConfig()
    
    # 시드 설정
    set_seed(config.seed)
    
    # 시스템 검사
    if not validate_system_requirements():
        return False
    
    # 모델 테스트
    try:
        print("🤖 모델 테스트 중...")
        model = SAMFineTuner(
            representation_dim=config.visual_tokens,
            freeze_encoder=config.freeze_encoder,
            use_gradient_checkpointing=config.use_gradient_checkpointing
        )
        
        print_model_info(model, config)
        
        # 간단한 forward 테스트
        device = torch.device(config.device)
        model = model.to(device)
        
        dummy_input = torch.randn(1, 1, 128, 128, 128).to(device)
        dummy_mask = torch.randint(0, 2, (1, 1, 128, 128, 128)).float().to(device)
        
        model.eval()
        with torch.no_grad():
            results = model(dummy_input, dummy_mask)
        
        print("✅ 모델 테스트 성공!")
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")
        return False
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 테스트 모드
        success = quick_test()
        sys.exit(0 if success else 1)
    else:
        # 일반 훈련 모드
        result = main()
        sys.exit(0 if result is not None else 1)