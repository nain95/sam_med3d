import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import wandb

def calculate_dice(pred, target):
    """Dice score 계산 - 안전한 reshape 사용"""
    pred = torch.sigmoid(pred)
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    return dice.item()

def calculate_iou(pred, target):
    """IoU score 계산 - 안전한 reshape 사용"""
    pred = torch.sigmoid(pred)
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def safe_tensor_operations(pred_masks, target_masks):
    """텐서 연산을 위한 안전한 전처리"""
    if not pred_masks.is_contiguous():
        pred_masks = pred_masks.contiguous()
    if not target_masks.is_contiguous():
        target_masks = target_masks.contiguous()
    
    return pred_masks, target_masks

def create_segmentation_visualization(image, target_mask, pred_mask, patient_id, slice_idx=None):
    """분할 결과 시각화 생성"""
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
    dice = calculate_dice(torch.tensor(pred_slice), torch.tensor(target_slice))
    iou = calculate_iou(torch.tensor(pred_slice), torch.tensor(target_slice))
    
    axes[3].set_title(f'Comparison\nDice: {dice:.3f}, IoU: {iou:.3f}')
    axes[3].axis('off')
    
    # 범례 추가
    red_patch = mpatches.Patch(color='red', label='Ground Truth')
    blue_patch = mpatches.Patch(color='blue', label='Prediction')
    axes[3].legend(handles=[red_patch, blue_patch], loc='upper right')
    
    plt.tight_layout()
    return fig

def log_sample_predictions(model, dataloader, device, config, epoch, phase="val"):
    """샘플 예측 결과를 WandB에 로깅"""
    if not config.use_wandb or not config.log_images:
        return
    
    model.eval()
    logged_samples = 0
    max_samples = 3  # 로깅할 최대 샘플 수
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None or batch["patient_id"][0] == "dummy":
                continue
            
            if logged_samples >= max_samples:
                break
            
            try:
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                patient_id = batch["patient_id"][0]
                
                # 텐서 연속성 보장
                image = image.contiguous()
                mask = mask.contiguous()
                
                # 예측
                results = model(image, mask)
                pred_masks = results['pred_masks']
                
                # 첫 번째 샘플만 시각화
                img_sample = image[0]  # (1, D, H, W)
                mask_sample = mask[0]  # (1, D, H, W)
                pred_sample = pred_masks[0]  # (1, D, H, W)
                
                # 시각화 생성
                fig = create_segmentation_visualization(
                    img_sample, mask_sample, pred_sample, patient_id
                )
                
                # WandB에 로깅
                wandb.log({
                    f"{phase}_predictions/sample_{logged_samples+1}": wandb.Image(
                        fig, 
                        caption=f"Epoch {epoch} - {patient_id}"
                    )
                }, step=epoch)
                
                plt.close(fig)
                logged_samples += 1
                
            except Exception as e:
                print(f"⚠️ 이미지 로깅 실패: {e}")
                continue

def log_metrics_to_wandb(metrics_dict, epoch, phase="train"):
    """메트릭을 WandB에 로깅"""
    if wandb.run is None:
        return
    
    log_dict = {}
    for key, value in metrics_dict.items():
        log_dict[f"{phase}_{key}"] = value
    
    wandb.log(log_dict, step=epoch)

def save_model_to_wandb(model, checkpoint_path, epoch, best_dice, best_iou, config):
    """모델을 WandB artifact로 저장"""
    if not config.use_wandb:
        return
    
    try:
        # Artifact 생성
        artifact = wandb.Artifact(
            name=f"sam_med3d_epoch_{epoch}",
            type="model",
            description=f"SAM-Med3D checkpoint at epoch {epoch}",
            metadata={
                "epoch": epoch,
                "best_dice": best_dice,
                "best_iou": best_iou,
                "architecture": "SAM-Med3D",
                "task": "Brain CT ICH Segmentation"
            }
        )
        
        # 파일 추가
        artifact.add_file(checkpoint_path)
        
        # Artifact 로그
        wandb.log_artifact(artifact)
        print(f"📦 모델 artifact 저장 완료: sam_med3d_epoch_{epoch}")
        
    except Exception as e:
        print(f"⚠️ WandB artifact 저장 실패: {e}")

def train_epoch(model, dataloader, optimizer, device, config):
    """훈련 에포크 - WandB 로깅 포함"""
    model.train()
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
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
            
        try:
            if batch["patient_id"][0] == "dummy":
                continue
            
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            
            # 텐서 연속성 보장
            image = image.contiguous()
            mask = mask.contiguous()
            
            if image.shape[0] == 0 or mask.shape[0] == 0:
                continue
            
            optimizer.zero_grad()
            
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
            
            # Backward pass
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Metrics 계산
            pred_masks, mask = safe_tensor_operations(pred_masks, mask)
            dice = calculate_dice(pred_masks, mask)
            iou = calculate_iou(pred_masks, mask)
            
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
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
        except Exception as e:
            print(f"❌ 배치 {batch_idx} 처리 실패: {e}")
            continue
    
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

def validate(model, dataloader, device, config):
    """검증 - WandB 로깅 포함"""
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
        pbar = tqdm(dataloader, desc="Validation")
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
                
            try:
                if batch["patient_id"][0] == "dummy":
                    continue
                
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                
                # 텐서 연속성 보장
                image = image.contiguous()
                mask = mask.contiguous()
                
                if image.shape[0] == 0 or mask.shape[0] == 0:
                    continue
                
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
                dice = calculate_dice(pred_masks, mask)
                iou = calculate_iou(pred_masks, mask)
                
                # 누적
                total_loss += losses['total_loss'].item()
                total_dice += dice
                total_iou += iou
                processed_batches += 1
                
                # Loss 컴포넌트 누적
                for key in loss_components:
                    if key in losses:
                        loss_components[key] += losses[key].item()
                
                pbar.set_postfix({
                    'Loss': f'{total_loss / processed_batches:.3f}',
                    'Dice': f'{total_dice / processed_batches:.3f}',
                    'IoU': f'{total_iou / processed_batches:.3f}'
                })
                
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

def train_model(model, train_loader, val_loader, config):
    """메인 훈련 함수 - WandB 로깅 포함"""
    device = torch.device(config.device)
    model = model.to(device)
    
    # WandB 초기화
    wandb_run = None
    if config.use_wandb:
        wandb_run = config.init_wandb()
        
        # 모델 아키텍처 로깅
        wandb.watch(model, log_freq=100, log_graph=True)
    
    # Optimizer 및 Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=getattr(config, 'weight_decay', 1e-4)
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5,
        patience=3, 
    )
    
    # 훈련 상태 변수
    best_dice = 0
    best_iou = 0
    patience_counter = 0
    
    # 결과 저장용
    train_history = {'loss': [], 'dice': [], 'iou': []}
    val_history = {'loss': [], 'dice': [], 'iou': []}
    
    print(f"🚀 훈련 시작")
    print(f"   Device: {device}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Output dir: {config.output_dir}")
    if config.use_wandb:
        print(f"   WandB: {wandb.run.url}")
    
    # 출력 디렉토리 생성
    os.makedirs(config.output_dir, exist_ok=True)
    
    for epoch in range(config.epochs):
        print(f"\n📊 Epoch {epoch+1}/{config.epochs}")
        print("-" * 50)
        
        # 훈련
        train_loss, train_dice, train_iou, train_loss_components = train_epoch(
            model, train_loader, optimizer, device, config
        )
        
        # 검증
        val_loss, val_dice, val_iou, val_loss_components = validate(
            model, val_loader, device, config
        )
        
        # Scheduler 업데이트
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 히스토리 저장
        train_history['loss'].append(train_loss)
        train_history['dice'].append(train_dice)
        train_history['iou'].append(train_iou)
        
        val_history['loss'].append(val_loss)
        val_history['dice'].append(val_dice)
        val_history['iou'].append(val_iou)
        
        # WandB 로깅
        if config.use_wandb:
            # 기본 메트릭 로깅
            metrics_dict = {
                'loss': train_loss,
                'dice': train_dice,
                'iou': train_iou,
                'learning_rate': current_lr,
                **{f"loss_{k}": v for k, v in train_loss_components.items()}
            }
            log_metrics_to_wandb(metrics_dict, epoch, "train")
            
            val_metrics_dict = {
                'loss': val_loss,
                'dice': val_dice,
                'iou': val_iou,
                **{f"loss_{k}": v for k, v in val_loss_components.items()}
            }
            log_metrics_to_wandb(val_metrics_dict, epoch, "val")
            
            # 추가 메트릭
            wandb.log({
                'epoch': epoch,
                'patience_counter': patience_counter,
                'best_dice': best_dice,
                'best_iou': best_iou
            }, step=epoch)
        
        # 결과 출력
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        print(f"LR    - {current_lr:.2e}")
        
        # 이미지 로깅 (주기적으로)
        if config.use_wandb and config.log_images and (epoch + 1) % config.log_freq == 0:
            log_sample_predictions(model, val_loader, device, config, epoch, "val")
        
        # 모델 저장 조건
        is_best_dice = val_dice > best_dice
        is_best_iou = val_iou > best_iou
        
        if is_best_dice or is_best_iou:
            if is_best_dice:
                best_dice = val_dice
                print(f"🎯 새로운 최고 Dice: {best_dice:.4f}")
            
            if is_best_iou:
                best_iou = val_iou
                print(f"🎯 새로운 최고 IoU: {best_iou:.4f}")
            
            patience_counter = 0
            
            # 최고 모델 저장
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_dice': best_dice,
                'best_iou': best_iou,
                'train_history': train_history,
                'val_history': val_history,
                'config': config.to_dict()
            }
            
            checkpoint_path = os.path.join(config.output_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            
            # WandB에 모델 저장 (주기적으로)
            if config.use_wandb and (epoch + 1) % config.save_model_freq == 0:
                save_model_to_wandb(model, checkpoint_path, epoch, best_dice, best_iou, config)
            
            # SAM encoder만 별도 저장
            try:
                sam_encoder_state = {
                    'image_encoder': model.sam.image_encoder.state_dict(),
                    'dice_score': best_dice,
                    'iou_score': best_iou,
                    'config': config.to_dict()
                }
                torch.save(sam_encoder_state, os.path.join(config.output_dir, 'sam_encoder.pth'))
            except Exception as e:
                print(f"⚠️ SAM encoder 저장 실패: {e}")
            
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{config.patience}")
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
    
    # 훈련 완료
    print(f"\n🎉 훈련 완료!")
    print(f"   최고 Dice: {best_dice:.4f}")
    print(f"   최고 IoU: {best_iou:.4f}")
    print(f"   저장 위치: {config.output_dir}")
    
    # 최종 모델을 WandB artifact로 저장
    if config.use_wandb:
        final_checkpoint_path = os.path.join(config.output_dir, 'best_model.pth')
        save_model_to_wandb(model, final_checkpoint_path, "final", best_dice, best_iou, config)
        
        # WandB 종료
        wandb.finish()
    
    return model

def load_sam_encoder(checkpoint_path, model_class=None):
    """Fine-tuned SAM encoder 로드"""
    try:
        if checkpoint_path.endswith('sam_encoder.pth'):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"✅ SAM encoder 로드 완료")
            print(f"   Dice: {checkpoint.get('dice_score', 'N/A'):.4f}")
            print(f"   IoU: {checkpoint.get('iou_score', 'N/A'):.4f}")
            return checkpoint
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if model_class is not None:
                model = model_class()
                model.load_state_dict(checkpoint['model_state_dict'])
                return model.sam.image_encoder
            else:
                return checkpoint
                
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None

def resume_training(model, checkpoint_path, train_loader, val_loader, config):
    """훈련 재개"""
    print(f"🔄 훈련 재개: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 모델 상태 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optimizer 재설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 히스토리 로드
    train_history = checkpoint.get('train_history', {'loss': [], 'dice': [], 'iou': []})
    val_history = checkpoint.get('val_history', {'loss': [], 'dice': [], 'iou': []})
    
    start_epoch = checkpoint['epoch'] + 1
    best_dice = checkpoint['best_dice']
    
    print(f"   시작 에포크: {start_epoch}")
    print(f"   최고 Dice: {best_dice:.4f}")
    
    # 설정 업데이트
    config.start_epoch = start_epoch
    
    return train_model(model, train_loader, val_loader, config)