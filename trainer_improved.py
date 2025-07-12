"""
개선된 SAM-Med3D 훈련 모듈
- Mixed precision training 지원
- Gradient accumulation
- 향상된 메모리 관리
- 안정적인 Loss 계산
- 실시간 모니터링
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import wandb
from pathlib import Path
import gc


def calculate_dice_safe(pred, target, smooth=1e-6):
    """안전한 Dice score 계산"""
    try:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
        return dice.mean().item()
    except Exception as e:
        print(f"⚠️ Dice 계산 오류: {e}")
        return 0.0


def calculate_iou_safe(pred, target, smooth=1e-6):
    """안전한 IoU score 계산"""
    try:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean().item()
    except Exception as e:
        print(f"⚠️ IoU 계산 오류: {e}")
        return 0.0


def safe_tensor_operations(pred_masks, target_masks):
    """텐서 연산을 위한 안전한 전처리"""
    if not pred_masks.is_contiguous():
        pred_masks = pred_masks.contiguous()
    if not target_masks.is_contiguous():
        target_masks = target_masks.contiguous()
    
    return pred_masks, target_masks


def create_visualization(image, target_mask, pred_mask, patient_id, slice_idx=None):
    """분할 결과 시각화 생성 - 메모리 효율적 버전"""
    try:
        # 입력 데이터 처리
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.detach().cpu().numpy()
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = torch.sigmoid(pred_mask).detach().cpu().numpy()
        
        # 3D 데이터에서 중심 슬라이스 선택
        if len(image.shape) == 4:  # (C, D, H, W)
            image = image[0]  # 첫 번째 채널
            target_mask = target_mask[0]
            pred_mask = pred_mask[0]
        
        if slice_idx is None:
            slice_idx = image.shape[0] // 2  # 중심 슬라이스
        
        img_slice = image[slice_idx]
        target_slice = target_mask[slice_idx]
        pred_slice = pred_mask[slice_idx]
        
        # 예측을 이진화
        pred_binary = (pred_slice > 0.5).astype(np.float32)
        
        # 시각화 생성
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # 1. 원본 이미지
        axes[0].imshow(img_slice, cmap='gray', aspect='equal')
        axes[0].set_title(f'Original\n{patient_id}')
        axes[0].axis('off')
        
        # 2. Ground Truth
        axes[1].imshow(img_slice, cmap='gray', aspect='equal')
        if target_slice.max() > 0:
            axes[1].imshow(target_slice, cmap='Reds', alpha=0.6, aspect='equal')
            axes[1].contour(target_slice, levels=[0.5], colors='red', linewidths=2)
        axes[1].set_title(f'Ground Truth\n({target_slice.sum():.0f} pixels)')
        axes[1].axis('off')
        
        # 3. Prediction
        axes[2].imshow(img_slice, cmap='gray', aspect='equal')
        if pred_binary.max() > 0:
            axes[2].imshow(pred_binary, cmap='Blues', alpha=0.6, aspect='equal')
            axes[2].contour(pred_binary, levels=[0.5], colors='blue', linewidths=2)
        axes[2].set_title(f'Prediction\n({pred_binary.sum():.0f} pixels)')
        axes[2].axis('off')
        
        # 4. Overlay Comparison
        axes[3].imshow(img_slice, cmap='gray', aspect='equal')
        if target_slice.max() > 0:
            axes[3].contour(target_slice, levels=[0.5], colors='red', linewidths=2, label='GT')
        if pred_binary.max() > 0:
            axes[3].contour(pred_binary, levels=[0.5], colors='blue', linewidths=2, label='Pred')
        
        # 메트릭 계산
        dice = calculate_dice_safe(torch.tensor(pred_slice), torch.tensor(target_slice))
        iou = calculate_iou_safe(torch.tensor(pred_slice), torch.tensor(target_slice))
        
        axes[3].set_title(f'Comparison\nDice: {dice:.3f}, IoU: {iou:.3f}')
        axes[3].axis('off')
        
        # 범례 추가
        red_patch = mpatches.Patch(color='red', label='Ground Truth')
        blue_patch = mpatches.Patch(color='blue', label='Prediction')
        axes[3].legend(handles=[red_patch, blue_patch], loc='upper right')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"⚠️ 시각화 생성 실패: {e}")
        return None


def log_sample_predictions(model, dataloader, device, config, epoch, phase="val", max_samples=2):
    """샘플 예측 결과를 WandB에 로깅 - 메모리 효율적 버전"""
    if not config.use_wandb or not config.log_images:
        return
    
    model.eval()
    logged_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None or batch["patient_id"][0] == "dummy":
                continue
            
            if logged_samples >= max_samples:
                break
            
            try:
                image = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                patient_id = batch["patient_id"][0]
                
                # 메모리 효율을 위해 첫 번째 샘플만 처리
                if image.size(0) > 1:
                    image = image[:1]
                    mask = mask[:1]
                    patient_id = patient_id if isinstance(patient_id, str) else patient_id[0]
                
                # 텐서 연속성 보장
                image = image.contiguous()
                mask = mask.contiguous()
                
                # 예측
                with autocast(enabled=config.use_mixed_precision):
                    results = model(image, mask)
                    pred_masks = results['pred_masks']
                
                # 시각화 생성
                fig = create_visualization(
                    image[0], mask[0], pred_masks[0], patient_id
                )
                
                if fig is not None:
                    # WandB에 로깅
                    wandb.log({
                        f"{phase}_predictions/sample_{logged_samples+1}": wandb.Image(
                            fig, 
                            caption=f"Epoch {epoch} - {patient_id}"
                        )
                    }, step=epoch)
                    
                    plt.close(fig)
                    logged_samples += 1
                
                # 메모리 정리
                del image, mask, results
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ 이미지 로깅 실패 (배치 {batch_idx}): {e}")
                continue
            
            # 메모리 보호를 위한 조기 종료
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1e9
                if memory_used > 10:  # 10GB 이상 사용시 중단
                    print(f"⚠️ 메모리 사용량 초과 ({memory_used:.1f}GB), 로깅 중단")
                    break


def log_metrics_to_wandb(metrics_dict, epoch, phase="train"):
    """메트릭을 WandB에 로깅"""
    if wandb.run is None:
        return
    
    try:
        log_dict = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                log_dict[f"{phase}_{key}"] = value
        
        if log_dict:  # 유효한 메트릭이 있을 때만 로깅
            wandb.log(log_dict, step=epoch)
    except Exception as e:
        print(f"⚠️ WandB 메트릭 로깅 실패: {e}")


def get_memory_info():
    """GPU 메모리 사용량 정보 반환"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "cached_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
    }


def cleanup_memory():
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class EarlyStopping:
    """개선된 Early Stopping 클래스"""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


def create_optimizer(model, config):
    """옵티마이저 생성"""
    # 파라미터 그룹 분리 (다른 학습률 적용 가능)
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'repr_head' in n and p.requires_grad],
            'lr': config.learning_rate * 2,  # representation head는 더 높은 학습률
            'weight_decay': config.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'repr_head' not in n and p.requires_grad],
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay
        }
    ]
    
    # 빈 그룹 제거
    param_groups = [group for group in param_groups if len(group['params']) > 0]
    
    if not param_groups:
        raise ValueError("학습 가능한 파라미터가 없습니다.")
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    return optimizer


def create_scheduler(optimizer, config):
    """학습률 스케줄러 생성"""
    if config.scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.min_lr,
            verbose=True
        )
    elif config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.min_lr
        )
    elif config.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.epochs // 4,
            gamma=config.scheduler_factor
        )
    else:
        scheduler = None
    
    return scheduler


def print_training_progress(epoch, total_epochs, metrics, elapsed_time, memory_info):
    """훈련 진행 상황 출력"""
    print(f"\n📊 Epoch {epoch+1}/{total_epochs} 완료")
    print(f"⏱️ 소요 시간: {elapsed_time:.1f}초")
    
    # 메트릭 출력
    if 'train' in metrics:
        train_metrics = metrics['train']
        print(f"🚂 Train - Loss: {train_metrics.get('loss', 0):.4f}, "
              f"Dice: {train_metrics.get('dice', 0):.4f}, "
              f"IoU: {train_metrics.get('iou', 0):.4f}")
    
    if 'val' in metrics:
        val_metrics = metrics['val']
        print(f"🔍 Val   - Loss: {val_metrics.get('loss', 0):.4f}, "
              f"Dice: {val_metrics.get('dice', 0):.4f}, "
              f"IoU: {val_metrics.get('iou', 0):.4f}")
    
    # 메모리 정보
    if memory_info.get("available", False):
        print(f"🖥️ GPU 메모리: {memory_info['allocated_gb']:.1f}GB allocated, "
              f"{memory_info['cached_gb']:.1f}GB cached")