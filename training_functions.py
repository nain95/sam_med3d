"""
개선된 훈련 함수들
- Mixed precision training
- Gradient accumulation
- 메모리 효율적 처리
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time
import numpy as np
from trainer_improved import (
    calculate_dice_safe, calculate_iou_safe, safe_tensor_operations,
    log_sample_predictions, log_metrics_to_wandb, get_memory_info, cleanup_memory
)


def train_epoch_improved(model, dataloader, optimizer, scaler, device, config, epoch):
    """개선된 훈련 에포크 - Mixed precision 및 Gradient accumulation 지원"""
    model.train()
    
    total_loss = 0
    total_dice = 0
    total_iou = 0
    processed_batches = 0
    accumulated_batches = 0
    
    # Loss 컴포넌트별 추적
    loss_components = {
        'dice_loss': 0,
        'focal_loss': 0,
        'iou_loss': 0,
        'iou_pred_loss': 0,
        'representation_loss': 0
    }
    
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    start_time = time.time()
    
    # Gradient accumulation을 위한 초기화
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        if batch is None or batch["patient_id"][0] == "dummy":
            continue
            
        try:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            
            # 텐서 연속성 보장
            image = image.contiguous()
            mask = mask.contiguous()
            
            if image.shape[0] == 0 or mask.shape[0] == 0:
                continue
            
            # Mixed precision forward pass
            with autocast(enabled=config.use_mixed_precision):
                # Forward pass
                results = model(image, mask)
                pred_masks = results['pred_masks']
                
                if not pred_masks.is_contiguous():
                    pred_masks = pred_masks.contiguous()
                
                # Loss 계산
                loss_weights = {
                    'dice': config.dice_weight,
                    'focal': config.focal_weight,
                    'iou': config.iou_weight,
                    'iou_pred': config.iou_pred_weight,
                    'representation': config.repr_weight
                }
                
                losses = model.calculate_loss(
                    pred_masks, 
                    mask,
                    results.get('iou_predictions'),
                    results.get('representation_features'),
                    loss_weights=loss_weights
                )
                
                # Gradient accumulation을 위한 loss scaling
                total_loss_batch = losses['total_loss'] / config.gradient_accumulation_steps
            
            # Backward pass with mixed precision
            if config.use_mixed_precision:
                scaler.scale(total_loss_batch).backward()
            else:
                total_loss_batch.backward()
            
            accumulated_batches += 1
            
            # Gradient accumulation 체크
            if accumulated_batches >= config.gradient_accumulation_steps:
                # Gradient clipping and optimization step
                if config.use_mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                accumulated_batches = 0
            
            # Metrics 계산 (GPU에서 직접)
            with torch.no_grad():
                pred_masks_detached = pred_masks.detach()
                mask_detached = mask.detach()
                
                pred_masks_detached, mask_detached = safe_tensor_operations(
                    pred_masks_detached, mask_detached
                )
                
                dice = calculate_dice_safe(pred_masks_detached, mask_detached)
                iou = calculate_iou_safe(pred_masks_detached, mask_detached)
            
            # 누적
            total_loss += losses['total_loss'].item()
            total_dice += dice
            total_iou += iou
            processed_batches += 1
            
            # Loss 컴포넌트 누적
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()
            
            # Progress bar 업데이트
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{total_loss / processed_batches:.3f}',
                'Dice': f'{total_dice / processed_batches:.3f}',
                'IoU': f'{total_iou / processed_batches:.3f}',
                'LR': f'{current_lr:.2e}',
                'Mem': f'{torch.cuda.memory_allocated() / 1e9:.1f}GB' if torch.cuda.is_available() else 'N/A'
            })
            
            # 주기적 메모리 정리
            if batch_idx % 50 == 0:
                cleanup_memory()
            
            # 프린트 주기
            if processed_batches % config.print_freq == 0:
                elapsed = time.time() - start_time
                print(f"\n  배치 {processed_batches}/{len(dataloader)} - "
                      f"Loss: {total_loss / processed_batches:.4f}, "
                      f"시간: {elapsed:.1f}s")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ GPU 메모리 부족 (배치 {batch_idx}): {e}")
                print(f"💡 현재 배치 크기: {image.shape[0]}, 이미지 크기: {image.shape}")
                cleanup_memory()
                
                # 메모리 부족시 해당 배치 스킵
                if 'image' in locals():
                    del image
                if 'mask' in locals():
                    del mask
                if 'results' in locals():
                    del results
                continue
            else:
                print(f"❌ 배치 {batch_idx} 처리 실패: {e}")
                continue
        except Exception as e:
            print(f"❌ 배치 {batch_idx} 처리 실패: {e}")
            continue
    
    # 남은 gradient가 있다면 업데이트
    if accumulated_batches > 0:
        if config.use_mixed_precision:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        optimizer.zero_grad()
    
    if processed_batches == 0:
        print("⚠️ 처리된 배치가 없습니다!")
        return 0.0, 0.0, 0.0, {}
    
    # 평균 계산
    avg_loss = total_loss / processed_batches
    avg_dice = total_dice / processed_batches
    avg_iou = total_iou / processed_batches
    
    # Loss 컴포넌트 평균
    avg_loss_components = {
        key: value / processed_batches 
        for key, value in loss_components.items()
    }
    
    return avg_loss, avg_dice, avg_iou, avg_loss_components


def validate_epoch_improved(model, dataloader, device, config, epoch):
    """개선된 검증 에포크 - 메모리 효율적 처리"""
    model.eval()
    
    total_loss = 0
    total_dice = 0
    total_iou = 0
    processed_batches = 0
    
    # Loss 컴포넌트별 추적
    loss_components = {
        'dice_loss': 0,
        'focal_loss': 0,
        'iou_loss': 0,
        'iou_pred_loss': 0,
        'representation_loss': 0
    }
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch+1}")
        start_time = time.time()
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None or batch["patient_id"][0] == "dummy":
                continue
                
            try:
                image = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                
                # 텐서 연속성 보장
                image = image.contiguous()
                mask = mask.contiguous()
                
                if image.shape[0] == 0 or mask.shape[0] == 0:
                    continue
                
                # Mixed precision forward pass
                with autocast(enabled=config.use_mixed_precision):
                    # Forward pass
                    results = model(image, mask)
                    pred_masks = results['pred_masks']
                    
                    if not pred_masks.is_contiguous():
                        pred_masks = pred_masks.contiguous()
                    
                    # Loss 계산
                    loss_weights = {
                        'dice': config.dice_weight,
                        'focal': config.focal_weight,
                        'iou': config.iou_weight,
                        'iou_pred': config.iou_pred_weight,
                        'representation': config.repr_weight
                    }
                    
                    losses = model.calculate_loss(
                        pred_masks,
                        mask,
                        results.get('iou_predictions'),
                        results.get('representation_features'),
                        loss_weights=loss_weights
                    )
                
                # Metrics 계산
                pred_masks, mask = safe_tensor_operations(pred_masks, mask)
                dice = calculate_dice_safe(pred_masks, mask)
                iou = calculate_iou_safe(pred_masks, mask)
                
                # 누적
                total_loss += losses['total_loss'].item()
                total_dice += dice
                total_iou += iou
                processed_batches += 1
                
                # Loss 컴포넌트 누적
                for key in loss_components:
                    if key in losses:
                        loss_components[key] += losses[key].item()
                
                # Progress bar 업데이트
                pbar.set_postfix({
                    'Loss': f'{total_loss / processed_batches:.3f}',
                    'Dice': f'{total_dice / processed_batches:.3f}',
                    'IoU': f'{total_iou / processed_batches:.3f}',
                    'Mem': f'{torch.cuda.memory_allocated() / 1e9:.1f}GB' if torch.cuda.is_available() else 'N/A'
                })
                
                # 주기적 메모리 정리
                if batch_idx % 20 == 0:
                    cleanup_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n❌ 검증 중 GPU 메모리 부족 (배치 {batch_idx}): {e}")
                    cleanup_memory()
                    continue
                else:
                    print(f"❌ 검증 배치 {batch_idx} 실패: {e}")
                    continue
            except Exception as e:
                print(f"❌ 검증 배치 {batch_idx} 실패: {e}")
                continue
    
    if processed_batches == 0:
        print("⚠️ 검증에서 처리된 배치가 없습니다!")
        return 0.0, 0.0, 0.0, {}
    
    # 평균 계산
    avg_loss = total_loss / processed_batches
    avg_dice = total_dice / processed_batches
    avg_iou = total_iou / processed_batches
    
    # Loss 컴포넌트 평균
    avg_loss_components = {
        key: value / processed_batches 
        for key, value in loss_components.items()
    }
    
    return avg_loss, avg_dice, avg_iou, avg_loss_components


def save_checkpoint_improved(model, optimizer, scheduler, scaler, epoch, metrics, config, 
                            checkpoint_type="regular"):
    """개선된 체크포인트 저장"""
    try:
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'metrics': metrics,
            'config_dict': config.to_dict(),
            'checkpoint_type': checkpoint_type,
            'pytorch_version': torch.__version__
        }
        
        # 파일명 설정
        if checkpoint_type == "best":
            filename = "best_model.pth"
        elif checkpoint_type == "latest":
            filename = "latest_model.pth"
        elif checkpoint_type == "sam_encoder":
            filename = "sam_encoder.pth"
        else:
            filename = f"checkpoint_epoch_{epoch:03d}.pth"
        
        filepath = os.path.join(config.output_dir, filename)
        
        # 디렉토리 생성
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 저장
        torch.save(checkpoint_data, filepath)
        
        # SAM encoder만 별도 저장 (BLIP2 호환)
        if checkpoint_type == "best":
            try:
                sam_encoder_state = {
                    'image_encoder_state_dict': model.sam.image_encoder.state_dict(),
                    'metrics': metrics,
                    'config_dict': config.to_dict(),
                    'epoch': epoch
                }
                sam_encoder_path = os.path.join(config.output_dir, 'sam_encoder.pth')
                torch.save(sam_encoder_state, sam_encoder_path)
                print(f"💾 SAM encoder 저장: {sam_encoder_path}")
            except Exception as e:
                print(f"⚠️ SAM encoder 저장 실패: {e}")
        
        print(f"💾 체크포인트 저장: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ 체크포인트 저장 실패: {e}")
        return None


def load_checkpoint_improved(checkpoint_path, model, optimizer=None, scheduler=None, 
                           scaler=None, device='cuda'):
    """개선된 체크포인트 로드"""
    try:
        print(f"📂 체크포인트 로드 중: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 모델 상태 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 옵티마이저 상태 로드
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 스케줄러 상태 로드
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            if checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Scaler 상태 로드
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            if checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 메타데이터 반환
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config_dict', {})
        }
        
        print(f"✅ 체크포인트 로드 완료 (Epoch: {metadata['epoch']})")
        
        return metadata
        
    except Exception as e:
        print(f"❌ 체크포인트 로드 실패: {e}")
        return None