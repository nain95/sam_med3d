import os
import random
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import pickle
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pathlib import Path


def set_seed(seed: int = 42):
    """재현 가능한 결과를 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Seed set to {seed}")


def setup_wandb(config, project_name: str = "sam-med3d-finetuning"):
    """WandB 초기화"""
    try:
        wandb.init(
            project=project_name,
            name=config.get_experiment_name(),
            config={
                # Training config
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'weight_decay': config.optimizer_config['weight_decay'],
                
                # Model config
                'input_size': config.input_size,
                'representation_dim': config.representation_dim,
                'freeze_prompt_encoder': config.freeze_prompt_encoder,
                'use_gradient_checkpointing': config.use_gradient_checkpointing,
                
                # Loss weights
                **{f'loss_weight_{k}': v for k, v in config.loss_weights.items()},
                
                # Other settings
                'patience': config.patience,
                'gradient_accumulation_steps': config.gradient_accumulation_steps,
            },
            tags=['sam-med3d', 'brain-ct', 'segmentation', 'fine-tuning']
        )
        
        print(f"🔗 WandB initialized: {wandb.run.url}")
        
    except Exception as e:
        print(f"❌ WandB initialization failed: {e}")
        print("Continuing without WandB logging...")


def save_model_checkpoint(model, optimizer, scheduler, epoch, metrics, config, 
                         checkpoint_type: str = "regular"):
    """모델 체크포인트 저장"""
    
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
        'checkpoint_type': checkpoint_type
    }
    
    # 파일명 설정
    if checkpoint_type == "best":
        filename = "best_model.pth"
    elif checkpoint_type == "latest":
        filename = "latest_model.pth"
    elif checkpoint_type == "blip2_encoder":
        filename = "blip2_compatible_encoder.pth"
    else:
        filename = f"checkpoint_epoch_{epoch}.pth"
    
    filepath = os.path.join(config.output_dir, filename)
    
    # 디렉토리 생성
    os.makedirs(config.output_dir, exist_ok=True)
    
    try:
        torch.save(checkpoint_data, filepath)
        print(f"💾 Checkpoint saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")
        return None


def load_model_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, 
                         load_optimizer: bool = True):
    """모델 체크포인트 로드"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 모델 weights 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Optimizer 로드 (선택적)
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Scheduler 로드 (선택적)
        if load_optimizer and scheduler is not None and 'scheduler_state_dict' in checkpoint:
            if checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 메타데이터 반환
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', None)
        }
        
        print(f"✅ Checkpoint loaded successfully")
        if 'epoch' in checkpoint:
            print(f"   📊 Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"   📊 {key}: {value:.4f}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        raise


def visualize_predictions(images, masks, predictions, save_path: str = None, 
                         max_samples: int = 4, slice_idx: int = None):
    """예측 결과 시각화"""
    
    batch_size = min(images.shape[0], max_samples)
    
    # 텐서를 numpy로 변환
    if torch.is_tensor(images):
        images = images.cpu().numpy()
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = torch.sigmoid(predictions).cpu().numpy()
    
    # 3D 데이터의 경우 중간 slice 선택
    if slice_idx is None:
        slice_idx = images.shape[2] // 2  # 중간 slice
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # 이미지
        img_slice = images[i, 0, slice_idx]  # [H, W]
        axes[i, 0].imshow(img_slice, cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}: Original')
        axes[i, 0].axis('off')
        
        # Ground truth 마스크
        mask_slice = masks[i, 0, slice_idx]
        axes[i, 1].imshow(img_slice, cmap='gray', alpha=0.7)
        axes[i, 1].imshow(mask_slice, cmap='Reds', alpha=0.5)
        axes[i, 1].set_title(f'Sample {i+1}: Ground Truth')
        axes[i, 1].axis('off')
        
        # 예측 마스크
        pred_slice = predictions[i, 0, slice_idx]
        axes[i, 2].imshow(img_slice, cmap='gray', alpha=0.7)
        axes[i, 2].imshow(pred_slice, cmap='Blues', alpha=0.5)
        axes[i, 2].set_title(f'Sample {i+1}: Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved: {save_path}")
    
    plt.show()
    return fig


def calculate_dataset_stats(csv_path: str, image_col: str = "path", 
                           mask_col: str = "mask", num_samples: int = 100):
    """데이터셋 통계 계산"""
    
    print(f"📊 Calculating dataset statistics: {csv_path}")
    
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    
    # 샘플링
    sample_size = min(num_samples, total_samples)
    sample_indices = np.random.choice(total_samples, sample_size, replace=False)
    
    image_intensities = []
    mask_ratios = []
    image_shapes = []
    
    valid_samples = 0
    
    for idx in sample_indices:
        try:
            row = df.iloc[idx]
            
            # 이미지 로드
            image_path = row[image_col]
            mask_path = row[mask_col]
            
            with open(image_path, 'rb') as f:
                image = pickle.load(f)
            with open(mask_path, 'rb') as f:
                mask = pickle.load(f)
            
            # numpy 변환
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask)
            
            # 통계 계산
            image_intensities.extend(image.flatten())
            mask_ratios.append(np.mean(mask > 0))
            image_shapes.append(image.shape)
            valid_samples += 1
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    if valid_samples == 0:
        print("❌ No valid samples found")
        return None
    
    # 결과 정리
    stats = {
        'total_samples': total_samples,
        'analyzed_samples': valid_samples,
        'image_intensity': {
            'mean': np.mean(image_intensities),
            'std': np.std(image_intensities),
            'min': np.min(image_intensities),
            'max': np.max(image_intensities),
            'percentiles': {
                '1%': np.percentile(image_intensities, 1),
                '5%': np.percentile(image_intensities, 5),
                '95%': np.percentile(image_intensities, 95),
                '99%': np.percentile(image_intensities, 99)
            }
        },
        'mask_statistics': {
            'mean_positive_ratio': np.mean(mask_ratios),
            'std_positive_ratio': np.std(mask_ratios),
            'min_positive_ratio': np.min(mask_ratios),
            'max_positive_ratio': np.max(mask_ratios)
        },
        'image_shapes': {
            'unique_shapes': list(set(image_shapes)),
            'most_common_shape': max(set(image_shapes), key=image_shapes.count)
        }
    }
    
    # 통계 출력
    print(f"📊 Dataset Statistics (based on {valid_samples} samples):")
    print(f"   📈 Image Intensity:")
    print(f"      Mean: {stats['image_intensity']['mean']:.3f} ± {stats['image_intensity']['std']:.3f}")
    print(f"      Range: [{stats['image_intensity']['min']:.3f}, {stats['image_intensity']['max']:.3f}]")
    
    print(f"   🎯 Mask Statistics:")
    print(f"      Positive ratio: {stats['mask_statistics']['mean_positive_ratio']:.3%} ± {stats['mask_statistics']['std_positive_ratio']:.3%}")
    print(f"      Range: [{stats['mask_statistics']['min_positive_ratio']:.3%}, {stats['mask_statistics']['max_positive_ratio']:.3%}]")
    
    print(f"   📏 Image Shapes:")
    print(f"      Most common: {stats['image_shapes']['most_common_shape']}")
    print(f"      Unique shapes: {len(stats['image_shapes']['unique_shapes'])}")
    
    return stats


def create_output_directories(config):
    """출력 디렉토리 생성"""
    dirs_to_create = [
        config.output_dir,
        os.path.join(config.output_dir, 'checkpoints'),
        os.path.join(config.output_dir, 'visualizations'),
        os.path.join(config.output_dir, 'logs')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Directory created: {dir_path}")


def log_gpu_memory():
    """GPU 메모리 사용량 로깅"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"🖥️ GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {total:.1f}GB total")
        
        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'total_gb': total,
            'utilization': allocated / total
        }
    else:
        print("🖥️ GPU not available")
        return None


def validate_data_paths(config):
    """데이터 경로 유효성 검사"""
    errors = []
    warnings = []
    
    # CSV 파일 확인
    for name, path in [("Training", config.train_csv_path), ("Validation", config.val_csv_path)]:
        if not path:
            errors.append(f"{name} CSV path is empty")
            continue
            
        if not os.path.exists(path):
            errors.append(f"{name} CSV file not found: {path}")
            continue
        
        try:
            df = pd.read_csv(path)
            
            # 필수 컬럼 확인
            required_cols = ["path", "mask"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                errors.append(f"{name} CSV missing columns: {missing_cols}")
            
            # 샘플 수 확인
            if len(df) == 0:
                errors.append(f"{name} CSV is empty")
            elif len(df) < 10:
                warnings.append(f"{name} CSV has only {len(df)} samples")
            
            print(f"✅ {name} CSV: {len(df)} samples")
            
        except Exception as e:
            errors.append(f"Error reading {name} CSV: {e}")
    
    # 출력 디렉토리 확인
    try:
        os.makedirs(config.output_dir, exist_ok=True)
        test_file = os.path.join(config.output_dir, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ Output directory writable: {config.output_dir}")
    except Exception as e:
        errors.append(f"Output directory not writable: {e}")
    
    # 결과 반환
    if errors:
        print("❌ Data validation errors:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    if warnings:
        print("⚠️ Data validation warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    
    print("✅ Data paths validation passed")
    return True


def format_time(seconds: float) -> str:
    """시간을 가독성 좋은 형태로 포맷"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_model_size(model) -> Dict[str, Any]:
    """모델 크기 정보 계산"""
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 메모리 사용량 추정 (bytes)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    total_size_mb = (param_size + buffer_size) / 1e6
    
    return {
        'total_params': param_count,
        'trainable_params': trainable_param_count,
        'frozen_params': param_count - trainable_param_count,
        'size_mb': total_size_mb,
        'trainable_ratio': trainable_param_count / param_count if param_count > 0 else 0
    }


def print_model_info(model, config):
    """모델 정보 출력"""
    model_info = get_model_size(model)
    
    print("🧠 Model Information:")
    print(f"   📊 Total parameters: {model_info['total_params']:,}")
    print(f"   🔧 Trainable parameters: {model_info['trainable_params']:,}")
    print(f"   ❄️ Frozen parameters: {model_info['frozen_params']:,}")
    print(f"   📏 Model size: {model_info['size_mb']:.1f} MB")
    print(f"   🔄 Trainable ratio: {model_info['trainable_ratio']:.1%}")
    
    # 모듈별 파라미터 분석
    print("\n🔍 Module-wise parameter count:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        if module_params > 0:
            print(f"   {name}: {module_params:,} ({trainable_params:,} trainable)")


if __name__ == "__main__":
    print("Testing SAM utilities...")
    
    # 시드 설정 테스트
    set_seed(42)
    
    # GPU 메모리 확인
    memory_info = log_gpu_memory()
    
    # 더미 모델로 모델 정보 테스트
    import torch.nn as nn
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(1, 64, 3, padding=1)
            self.fc = nn.Linear(64, 10)
            # 일부 파라미터 freeze
            for param in self.conv.parameters():
                param.requires_grad = False
    
    dummy_model = DummyModel()
    print_model_info(dummy_model, None)
    
    # 시간 포맷 테스트
    print(f"\nTime formatting test:")
    for seconds in [30, 90, 3660, 7200]:
        print(f"  {seconds}s -> {format_time(seconds)}")
    
    print("\nSAM utilities tests completed!")