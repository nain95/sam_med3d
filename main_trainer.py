"""
개선된 메인 훈련 함수
- 안정적인 훈련 루프
- 메모리 효율성
- 완전한 모니터링
"""

import torch
from torch.cuda.amp import GradScaler
import time
import os
from pathlib import Path
import wandb

from trainer_improved import (
    EarlyStopping, create_optimizer, create_scheduler,
    log_sample_predictions, log_metrics_to_wandb,
    get_memory_info, cleanup_memory, print_training_progress
)
from training_functions import (
    train_epoch_improved, validate_epoch_improved,
    save_checkpoint_improved, load_checkpoint_improved
)


def train_model_improved(model, train_loader, val_loader, config):
    """개선된 메인 훈련 함수"""
    
    print("\n" + "="*80)
    print("🚀 개선된 SAM-Med3D 훈련 시작")
    print("="*80)
    
    # 디바이스 설정
    device = torch.device(config.device)
    model = model.to(device)
    
    # WandB 초기화
    wandb_run = None
    if config.use_wandb:
        try:
            wandb_run = config.init_wandb()
            if wandb_run:
                wandb.watch(model, log_freq=100, log_graph=False)  # graph 로깅 비활성화 (메모리 절약)
        except Exception as e:
            print(f"⚠️ WandB 초기화 실패: {e}")
            config.use_wandb = False
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_mixed_precision else None
    
    # 옵티마이저 및 스케줄러 생성
    try:
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
        
        print(f"✅ 옵티마이저 생성 완료")
        print(f"   학습률: {config.learning_rate}")
        print(f"   가중치 감쇠: {config.weight_decay}")
        print(f"   Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"   Mixed precision: {config.use_mixed_precision}")
        
    except Exception as e:
        print(f"❌ 옵티마이저 생성 실패: {e}")
        return None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=0.001,
        restore_best_weights=True
    )
    
    # 훈련 상태 변수
    best_dice = 0
    best_iou = 0
    best_metrics = {}
    
    # 결과 저장용
    train_history = {'loss': [], 'dice': [], 'iou': []}
    val_history = {'loss': [], 'dice': [], 'iou': []}
    
    # 시작 시간
    training_start_time = time.time()
    
    # 모델 파라미터 정보
    param_info = model.get_trainable_params()
    print(f"\n🧠 모델 정보:")
    print(f"   전체 파라미터: {param_info['total_params']:,}")
    print(f"   학습 가능: {param_info['trainable_params']:,}")
    print(f"   학습 가능 비율: {param_info['trainable_ratio']:.1%}")
    
    # 초기 메모리 상태
    initial_memory = get_memory_info()
    if initial_memory.get("available", False):
        print(f"   초기 GPU 메모리: {initial_memory['allocated_gb']:.1f}GB")
    
    print(f"\n📊 훈련 설정:")
    print(f"   에포크: {config.epochs}")
    print(f"   배치 크기: {config.batch_size}")
    print(f"   훈련 샘플: {len(train_loader.dataset)}")
    print(f"   검증 샘플: {len(val_loader.dataset)}")
    print(f"   출력 디렉토리: {config.output_dir}")
    
    # 메인 훈련 루프
    try:
        for epoch in range(config.epochs):
            epoch_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"📅 Epoch {epoch+1}/{config.epochs}")
            print(f"{'='*60}")
            
            # 현재 학습률 출력
            current_lrs = [group['lr'] for group in optimizer.param_groups]
            print(f"📈 학습률: {current_lrs[0]:.2e}")
            if len(current_lrs) > 1:
                print(f"   (representation head: {current_lrs[0]:.2e}, others: {current_lrs[1]:.2e})")
            
            # 훈련
            print("\n🚂 훈련 중...")
            train_loss, train_dice, train_iou, train_loss_components = train_epoch_improved(
                model, train_loader, optimizer, scaler, device, config, epoch
            )
            
            # 검증
            print("\n🔍 검증 중...")
            val_loss, val_dice, val_iou, val_loss_components = validate_epoch_improved(
                model, val_loader, device, config, epoch
            )
            
            # 스케줄러 업데이트
            if scheduler is not None:
                if config.scheduler_type == "reduce_on_plateau":
                    scheduler.step(val_dice)
                else:
                    scheduler.step()
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            # 히스토리 저장
            train_history['loss'].append(train_loss)
            train_history['dice'].append(train_dice)
            train_history['iou'].append(train_iou)
            
            val_history['loss'].append(val_loss)
            val_history['dice'].append(val_dice)
            val_history['iou'].append(val_iou)
            
            # 현재 에포크 메트릭
            current_metrics = {
                'train': {
                    'loss': train_loss,
                    'dice': train_dice,
                    'iou': train_iou,
                    **{f"loss_{k}": v for k, v in train_loss_components.items()}
                },
                'val': {
                    'loss': val_loss,
                    'dice': val_dice,
                    'iou': val_iou,
                    **{f"loss_{k}": v for k, v in val_loss_components.items()}
                }
            }
            
            # WandB 로깅
            if config.use_wandb:
                try:
                    # 기본 메트릭
                    log_metrics_to_wandb(current_metrics['train'], epoch, "train")
                    log_metrics_to_wandb(current_metrics['val'], epoch, "val")
                    
                    # 추가 메트릭
                    wandb.log({
                        'epoch': epoch,
                        'learning_rate': current_lrs[0],
                        'epoch_duration': epoch_duration,
                        'best_dice': best_dice,
                        'best_iou': best_iou
                    }, step=epoch)
                    
                    # GPU 메모리 로깅
                    memory_info = get_memory_info()
                    if memory_info.get("available", False):
                        wandb.log({
                            'gpu_memory_allocated_gb': memory_info['allocated_gb'],
                            'gpu_memory_cached_gb': memory_info['cached_gb']
                        }, step=epoch)
                        
                except Exception as e:
                    print(f"⚠️ WandB 로깅 실패: {e}")
            
            # 진행 상황 출력
            memory_info = get_memory_info()
            print_training_progress(epoch, config.epochs, current_metrics, epoch_duration, memory_info)
            
            # 최고 성능 체크
            is_best_dice = val_dice > best_dice
            is_best_iou = val_iou > best_iou
            is_best = is_best_dice or is_best_iou
            
            if is_best:
                if is_best_dice:
                    best_dice = val_dice
                    print(f"🎯 새로운 최고 Dice: {best_dice:.4f}")
                
                if is_best_iou:
                    best_iou = val_iou
                    print(f"🎯 새로운 최고 IoU: {best_iou:.4f}")
                
                best_metrics = current_metrics
                
                # 최고 모델 저장
                best_checkpoint_path = save_checkpoint_improved(
                    model, optimizer, scheduler, scaler, epoch, 
                    current_metrics, config, "best"
                )
                
                if best_checkpoint_path:
                    print(f"💾 최고 모델 저장: {best_checkpoint_path}")
            
            # 최신 모델 저장 (주기적)
            if (epoch + 1) % 10 == 0:
                latest_checkpoint_path = save_checkpoint_improved(
                    model, optimizer, scheduler, scaler, epoch, 
                    current_metrics, config, "latest"
                )
            
            # 이미지 로깅 (주기적)
            if config.use_wandb and config.log_images and (epoch + 1) % config.log_freq == 0:
                print("📸 샘플 예측 로깅 중...")
                log_sample_predictions(model, val_loader, device, config, epoch, "val")
            
            # Early stopping 체크
            if early_stopping(val_dice, model):
                print(f"\n🛑 Early stopping 발동 (epoch {epoch+1})")
                print(f"   최고 Dice: {best_dice:.4f}")
                print(f"   최고 IoU: {best_iou:.4f}")
                break
            
            # 메모리 정리
            cleanup_memory()
            
            # 훈련 시간 추정
            elapsed_time = time.time() - training_start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = config.epochs - epoch - 1
            estimated_remaining = avg_epoch_time * remaining_epochs
            
            print(f"⏱️ 경과 시간: {elapsed_time/3600:.1f}시간, "
                  f"예상 남은 시간: {estimated_remaining/3600:.1f}시간")
        
        # 훈련 완료
        total_training_time = time.time() - training_start_time
        
        print(f"\n{'='*80}")
        print("🎉 훈련 완료!")
        print(f"{'='*80}")
        print(f"   총 훈련 시간: {total_training_time/3600:.2f}시간")
        print(f"   최고 Dice: {best_dice:.4f}")
        print(f"   최고 IoU: {best_iou:.4f}")
        print(f"   저장 위치: {config.output_dir}")
        
        # 최종 결과 요약
        summary = {
            'best_dice': best_dice,
            'best_iou': best_iou,
            'total_epochs': epoch + 1,
            'training_time_hours': total_training_time / 3600,
            'final_train_loss': train_history['loss'][-1] if train_history['loss'] else 0,
            'final_val_loss': val_history['loss'][-1] if val_history['loss'] else 0
        }
        
        # WandB 최종 요약
        if config.use_wandb:
            try:
                wandb.log({
                    'training_summary/best_dice': best_dice,
                    'training_summary/best_iou': best_iou,
                    'training_summary/total_epochs': epoch + 1,
                    'training_summary/training_time_hours': total_training_time / 3600
                })
                
                # 훈련 곡선 저장
                wandb.log({
                    'training_curves/train_loss': train_history['loss'],
                    'training_curves/val_loss': val_history['loss'],
                    'training_curves/train_dice': train_history['dice'],
                    'training_curves/val_dice': val_history['dice']
                })
                
                print(f"📈 WandB 링크: {wandb.run.url}")
                
            except Exception as e:
                print(f"⚠️ WandB 최종 로깅 실패: {e}")
        
        return {
            'model': model,
            'best_metrics': best_metrics,
            'summary': summary,
            'train_history': train_history,
            'val_history': val_history
        }
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자에 의해 훈련이 중단되었습니다 (Epoch {epoch+1})")
        
        # 중단 시점의 모델 저장
        interrupt_checkpoint_path = save_checkpoint_improved(
            model, optimizer, scheduler, scaler, epoch, 
            current_metrics, config, f"interrupted_epoch_{epoch}"
        )
        
        if interrupt_checkpoint_path:
            print(f"💾 중단 시점 모델 저장: {interrupt_checkpoint_path}")
        
        return None
        
    except Exception as e:
        print(f"\n❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        return None
        
    finally:
        # 정리 작업
        cleanup_memory()
        
        if config.use_wandb and wandb.run:
            try:
                wandb.finish()
                print("📝 WandB 세션 종료")
            except:
                pass


def resume_training_improved(model, checkpoint_path, train_loader, val_loader, config):
    """개선된 훈련 재개"""
    print(f"🔄 훈련 재개: {checkpoint_path}")
    
    device = torch.device(config.device)
    model = model.to(device)
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_mixed_precision else None
    
    # 옵티마이저 및 스케줄러 생성
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # 체크포인트 로드
    metadata = load_checkpoint_improved(
        checkpoint_path, model, optimizer, scheduler, scaler, device
    )
    
    if metadata is None:
        print("❌ 체크포인트 로드 실패")
        return None
    
    start_epoch = metadata['epoch'] + 1
    print(f"   시작 에포크: {start_epoch}")
    
    # 남은 에포크로 설정 업데이트
    remaining_epochs = max(0, config.epochs - start_epoch)
    if remaining_epochs == 0:
        print("⚠️ 이미 훈련이 완료된 모델입니다.")
        return model
    
    print(f"   남은 에포크: {remaining_epochs}")
    
    # 기존 설정 일부 복원
    if 'config' in metadata and metadata['config']:
        # 중요한 설정들은 유지하되, 경로 관련 설정은 현재 config 사용
        old_config = metadata['config']
        for key in ['dice_weight', 'focal_weight', 'iou_weight', 'iou_pred_weight', 'repr_weight']:
            if key in old_config:
                setattr(config, key, old_config[key])
    
    # 설정 조정
    config.epochs = start_epoch + remaining_epochs
    
    return train_model_improved(model, train_loader, val_loader, config)


if __name__ == "__main__":
    print("개선된 SAM-Med3D 훈련 모듈")
    print("이 파일은 직접 실행되지 않습니다. train.py에서 import하여 사용하세요.")