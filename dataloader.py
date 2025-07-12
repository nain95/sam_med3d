import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import os
from scipy import ndimage
from monai.transforms import (
    Compose, ToTensord, Resized, RandRotated, 
    RandAdjustContrastd, RandGaussianNoised, RandFlipd,
    EnsureChannelFirstd, RandScaleIntensityd, RandShiftIntensityd
)

def transpose_hwdc_to_dhwc(data_dict):
    """축 순서 변경: (C,H,W,D) -> (C,D,H,W)"""
    for key in ["image", "mask"]:
        if key in data_dict:
            tensor = data_dict[key]
            if len(tensor.shape) == 4:
                data_dict[key] = tensor.permute(0, 3, 1, 2)
    return data_dict
    """축 순서 변경: (C,H,W,D) -> (C,D,H,W)"""
    for key in ["image", "mask"]:
        if key in data_dict:
            tensor = data_dict[key]
            if len(tensor.shape) == 4:
                data_dict[key] = tensor.permute(0, 3, 1, 2)
    return data_dict


def apply_ct_windowing(image, window_level=40.0, window_width=80.0, auto_window=False):
    """CT 윈도잉 적용"""
    if auto_window:
        p2, p98 = np.percentile(image, [2, 98])
        if p98 > p2:
            window_level = (p2 + p98) / 2
            window_width = (p98 - p2) * 1.2
        else:
            window_level, window_width = 40.0, 80.0
    
    window_min = window_level - window_width / 2
    window_max = window_level + window_width / 2
    
    windowed = np.clip(image, window_min, window_max)
    
    if window_max > window_min:
        windowed = (windowed - window_min) / (window_max - window_min)
    else:
        windowed = np.zeros_like(windowed)
    
    return windowed, window_min, window_max


class SAMMed3DDataset(Dataset):
    def __init__(self, csv_path, is_train=True, target_size=(128, 128, 128), 
                 channel_method="best_channel", window_level=40.0, window_width=80.0,
                 mask_rotation='no_rotation', verbose=False):
        """
        SAM-Med3D용 뇌출혈 CT 데이터셋 (패딩 기반 - 의학적으로 정확한 방식)
        
        원본 256×256×64를 128×128×128로 변환:
        - H, W: 256→128 리사이즈 (공간 해상도 감소)
        - D: 64→128 패딩 (원본 슬라이스 보존, 인위적 슬라이스 생성 방지)
        
        Args:
            csv_path: CSV 파일 경로
            is_train: 훈련 모드 여부
            target_size: 타겟 크기 (D, H, W) = (128, 128, 128)
            channel_method: 채널 선택 방법 ('first_channel', 'middle_channel', 'best_channel')
            window_level: CT 윈도우 레벨
            window_width: CT 윈도우 폭
            mask_rotation: 마스크 회전 옵션 ('no_rotation', 'rot_90_cw', 'rot_90_ccw', 'rot_180')
            verbose: 디버깅 출력 여부
        """
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['ich'] == 1].reset_index(drop=True)
        self.is_train = is_train
        self.target_size = target_size  # (D, H, W)
        self.channel_method = channel_method
        self.window_level = window_level
        self.window_width = window_width
        self.mask_rotation = mask_rotation
        self.verbose = verbose
        self.mask_dir = "/storage01/data/image/Brain_CT/ICH/CC/MASK"
        
        if verbose:
            print(f"SAM-Med3D Dataset (패딩 기반): {len(self.df)} ICH 샘플")
            print(f"변환: 256×256×64 → {target_size} (H,W 리사이즈 + D 패딩)")
            print(f"윈도우: L{window_level}/W{window_width}")
        
        self._setup_transforms()

    def _setup_transforms(self):
        """MONAI 변환 설정 - 패딩 기반 (MONAI 기본 transform 사용)"""
        from monai.transforms import Resized, SpatialPadd, CenterSpatialCropd
        
        # target_size는 (D, H, W) 순서, MONAI spatial_size는 (H, W, D) 순서
        target_h, target_w, target_d = self.target_size[1], self.target_size[2], self.target_size[0]
        
        transforms = [
            EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
            ToTensord(keys=["image", "mask"]),
            # 1단계: H, W를 128로 리사이즈, D는 원본 유지 (-1 사용)
            Resized(
                keys=["image", "mask"], 
                spatial_size=(target_h, target_w, -1),  # (H, W, D) - D는 원본 유지
                mode=["trilinear", "nearest"]
            ),
            # 2단계: D 차원을 패딩으로 128에 맞춤 (중앙 정렬)
            SpatialPadd(
                keys=["image", "mask"],
                spatial_size=(target_h, target_w, target_d),  # (H, W, D)
                mode="constant"
            ),
            # 3단계: 축 순서 변경: (C,H,W,D) -> (C,D,H,W)
            transpose_hwdc_to_dhwc,
        ]
        
        if self.is_train:
            transforms.extend([
                RandFlipd(keys=["image", "mask"], prob=0.3, spatial_axis=[1, 2]),  # 좌우만
                RandRotated(keys=["image", "mask"], prob=0.2, 
                        range_x=5, range_y=5, range_z=0),
                RandScaleIntensityd(keys=["image"], prob=0.3, factors=0.1),
                RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.05),
                RandAdjustContrastd(keys=["image"], prob=0.4, gamma=(0.8, 1.2)),
                RandGaussianNoised(keys=["image"], prob=0.2, std=0.03)
            ])
        
        self.transform = Compose(transforms)

    def _select_best_channel(self, image):
        """최적 채널 자동 선택"""
        if len(image.shape) == 4 and image.shape[0] == 3:
            channel_vars = [image[i].var() for i in range(3)]
            best_idx = np.argmax(channel_vars)
            return image[best_idx], best_idx
        return image, 0

    def process_image_channels(self, raw_image):
        """이미지 채널 처리 및 윈도잉"""
        if isinstance(raw_image, (list, tuple)):
            raw_image = raw_image[0] if len(raw_image) > 0 else raw_image
        
        image = np.array(raw_image, dtype=np.float32)
        
        # 채널 선택
        if len(image.shape) == 4 and image.shape[0] == 3:
            if self.channel_method == "first_channel":
                image = image[0]
            elif self.channel_method == "middle_channel":
                image = image[1]
            elif self.channel_method == "best_channel":
                image, _ = self._select_best_channel(image)
            else:
                image = image[0]
        
        if len(image.shape) != 3:
            raise ValueError(f"예상치 못한 이미지 차원: {image.shape}")
        
        # CT 윈도잉 적용
        original_min, original_max = image.min(), image.max()
        
        if original_max > original_min:
            if original_min < -500 and original_max > 500:  # 전형적인 CT HU 범위
                windowed_image, _, _ = apply_ct_windowing(
                    image, self.window_level, self.window_width
                )
            else:
                windowed_image, _, _ = apply_ct_windowing(image, auto_window=True)
        else:
            windowed_image = image
        
        return windowed_image

    def load_and_process_mask(self, image_path, target_shape):
        """마스크 로드 및 처리"""
        filename = os.path.basename(image_path)
        file_number = os.path.splitext(filename)[0]
        mask_path = os.path.join(self.mask_dir, f"{file_number}.nii")
        
        if not os.path.exists(mask_path):
            return None
        
        try:
            nii_img = nib.load(mask_path)
            mask = nii_img.get_fdata().astype(np.float32)
            
            # 회전 적용
            rotation_map = {
                'no_rotation': mask,
                'rot_90_cw': np.rot90(mask, k=1, axes=(0, 1)),
                'rot_90_ccw': np.rot90(mask, k=-1, axes=(0, 1)),
                'rot_180': np.rot90(mask, k=2, axes=(0, 1)),
            }
            
            mask_rotated = rotation_map.get(self.mask_rotation, mask)
            
            # 리사이즈
            if mask_rotated.shape != target_shape:
                zoom_factors = [t/m for t, m in zip(target_shape, mask_rotated.shape)]
                mask_resized = ndimage.zoom(mask_rotated, zoom_factors, order=0, prefilter=False)
            else:
                mask_resized = mask_rotated
            
            return (mask_resized > 0).astype(np.float32)
            
        except Exception as e:
            if self.verbose:
                print(f"마스크 처리 실패: {e}")
            return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        max_retries = 3
        
        for retry in range(max_retries):
            current_idx = (idx + retry) % len(self.df)
            
            try:
                row = self.df.iloc[current_idx]
                image_path = row["path"]
                
                # 이미지 로드
                with open(image_path, "rb") as f:
                    raw_image = pickle.load(f)
                
                image = self.process_image_channels(raw_image)
                
                # 마스크 로드
                mask = self.load_and_process_mask(image_path, image.shape)
                
                if mask is None:
                    continue
                
                if image.shape != mask.shape:
                    continue
                
                positive_ratio = np.mean(mask > 0)
                
                data = {
                    "image": image,
                    "mask": mask,
                    "patient_id": row.get("patient_id", f"patient_{current_idx}"),
                    "positive_ratio": positive_ratio
                }
                
                if self.transform:
                    data = self.transform(data)
                
                return data
                
            except Exception as e:
                if self.verbose:
                    print(f"샘플 {current_idx} 처리 실패: {e}")
                continue
        
        # 더미 샘플 반환
        return self._get_dummy_sample()

    def _get_dummy_sample(self):
        """더미 샘플 생성"""
        dummy_shape = self.target_size
        
        data = {
            "image": np.zeros((dummy_shape[1], dummy_shape[2], dummy_shape[0]), dtype=np.float32),
            "mask": np.zeros((dummy_shape[1], dummy_shape[2], dummy_shape[0]), dtype=np.float32),
            "patient_id": "dummy",
            "positive_ratio": 0.0
        }
        
        if self.transform:
            data = self.transform(data)
        
        return data


def robust_collate_fn(batch):
    """SAM-Med3D용 collate function"""
    valid_batch = [item for item in batch if item is not None and item.get("patient_id") != "dummy"]
    
    if len(valid_batch) == 0:
        return {
            "image": torch.zeros(1, 1, 128, 128, 128),
            "mask": torch.zeros(1, 1, 128, 128, 128),
            "patient_id": ["dummy"],
            "positive_ratio": [0.0]
        }
    
    try:
        collated = {}
        
        for key in valid_batch[0].keys():
            if key in ["image", "mask"]:
                tensors = [item[key] for item in valid_batch]
                collated[key] = torch.stack(tensors, dim=0)
            else:
                collated[key] = [item[key] for item in valid_batch]
        
        return collated
        
    except Exception as e:
        print(f"Collate 에러: {e}")
        return {
            "image": torch.zeros(1, 1, 128, 128, 128),
            "mask": torch.zeros(1, 1, 128, 128, 128),
            "patient_id": ["dummy"],
            "positive_ratio": [0.0]
        }


def get_sam_med3d_dataloader(csv_path, batch_size=2, is_train=True, 
                            target_size=(128, 128, 128), channel_method="best_channel",
                            window_level=40.0, window_width=80.0, mask_rotation='no_rotation',
                            num_workers=2, verbose=False):
    """
    SAM-Med3D용 데이터로더 생성 (패딩 기반 - 의학적으로 정확한 방식)
    
    원본 256×256×64 → 128×128×128 변환:
    - H, W: 리사이즈 (공간 해상도 감소, 허용됨)
    - D: 패딩 (원본 64 슬라이스 보존 + 32 패딩 앞뒤)
    
    Args:
        csv_path: CSV 파일 경로
        batch_size: 배치 크기
        is_train: 훈련 모드 여부
        target_size: 타겟 크기 (D, H, W) = (128, 128, 128)
        channel_method: 채널 선택 방법
        window_level: CT 윈도우 레벨 (기본: 40 HU)
        window_width: CT 윈도우 폭 (기본: 80 HU)
        mask_rotation: 마스크 회전 옵션
        num_workers: 워커 수
        verbose: 디버깅 출력 여부
        
    Returns:
        DataLoader: SAM-Med3D 호환 데이터로더 [batch, 1, 128, 128, 128]
    """
    dataset = SAMMed3DDataset(
        csv_path=csv_path,
        is_train=is_train,
        target_size=target_size,
        channel_method=channel_method,
        window_level=window_level,
        window_width=window_width,
        mask_rotation=mask_rotation,
        verbose=verbose
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=robust_collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )


def visualize_samples(csv_path, output_dir="./visualization", num_samples=5, 
                     mask_rotation='no_rotation', save_plots=True):
    """데이터로더 샘플 시각화"""
    import matplotlib.pyplot as plt
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    dataloader = get_sam_med3d_dataloader(
        csv_path=csv_path,
        batch_size=1,
        is_train=False,
        mask_rotation=mask_rotation,
        verbose=False
    )
    
    sample_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch is None or batch["patient_id"][0] == "dummy":
            continue
            
        if sample_count >= num_samples:
            break
            
        image = batch["image"][0, 0].numpy()  # (D, H, W)
        mask = batch["mask"][0, 0].numpy()
        patient_id = batch["patient_id"][0]
        positive_ratio = batch["positive_ratio"][0]
        
        print(f"샘플 {sample_count + 1}: {patient_id} | 마스크: {positive_ratio:.3%}")
        
        # 중심 슬라이스들 선택
        d_center = image.shape[0] // 2
        start_slice = max(0, d_center - 4)
        end_slice = min(image.shape[0], d_center + 5)
        slice_indices = list(range(start_slice, end_slice))
        
        if save_plots:
            fig, axes = plt.subplots(2, len(slice_indices), figsize=(18, 6))
            fig.suptitle(f'{patient_id} | Shape: {image.shape} (D,H,W) | Mask: {positive_ratio:.3%}', 
                        fontsize=12, fontweight='bold')
            
            for i, slice_idx in enumerate(slice_indices):
                img_slice = image[slice_idx, :, :]
                mask_slice = mask[slice_idx, :, :]
                
                # 원본 이미지
                axes[0, i].imshow(img_slice, cmap='gray', aspect='equal', origin='upper', vmin=0, vmax=1)
                axes[0, i].set_title(f'Axial {slice_idx}')
                axes[0, i].axis('off')
                
                # 마스크 오버레이
                axes[1, i].imshow(img_slice, cmap='gray', aspect='equal', origin='upper', vmin=0, vmax=1)
                
                mask_ratio_slice = (mask_slice > 0).mean()
                if mask_ratio_slice > 0:
                    axes[1, i].imshow(mask_slice, cmap='Reds', alpha=0.6, aspect='equal', origin='upper')
                    try:
                        axes[1, i].contour(mask_slice, levels=[0.5], colors='yellow', linewidths=1.5)
                    except:
                        pass
                
                axes[1, i].set_title(f'Overlay ({mask_ratio_slice:.2%})')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f'sample_{sample_count + 1:02d}_{patient_id}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        sample_count += 1
    
    if save_plots:
        print(f"시각화 완료: {output_dir}/ 에 {sample_count}개 샘플 저장")
    
    return True


def test_mask_rotations(csv_path, output_dir="./mask_test", sample_idx=0):
    """마스크 회전 옵션 비교 테스트"""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    rotation_options = ['no_rotation', 'rot_90_cw', 'rot_90_ccw', 'rot_180']
    
    fig, axes = plt.subplots(2, len(rotation_options), figsize=(16, 8))
    fig.suptitle(f'Mask Rotation Comparison (Sample {sample_idx})', fontsize=14, fontweight='bold')
    
    for i, rotation in enumerate(rotation_options):
        dataloader = get_sam_med3d_dataloader(
            csv_path=csv_path,
            batch_size=1,
            is_train=False,
            mask_rotation=rotation,
            verbose=False
        )
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx == sample_idx and batch["patient_id"][0] != "dummy":
                image = batch["image"][0, 0].numpy()
                mask = batch["mask"][0, 0].numpy()
                positive_ratio = batch["positive_ratio"][0]
                
                d_center = image.shape[0] // 2
                img_slice = image[d_center, :, :]
                mask_slice = mask[d_center, :, :]
                
                # 원본
                axes[0, i].imshow(img_slice, cmap='gray', aspect='equal', origin='upper', vmin=0, vmax=1)
                axes[0, i].set_title(f'{rotation}\nImage')
                axes[0, i].axis('off')
                
                # 오버레이
                axes[1, i].imshow(img_slice, cmap='gray', aspect='equal', origin='upper', vmin=0, vmax=1)
                
                mask_ratio = (mask_slice > 0).mean()
                if mask_ratio > 0:
                    axes[1, i].imshow(mask_slice, cmap='Reds', alpha=0.6, aspect='equal', origin='upper')
                    try:
                        axes[1, i].contour(mask_slice, levels=[0.5], colors='yellow', linewidths=1.5)
                    except:
                        pass
                
                axes[1, i].set_title(f'Overlay\n({positive_ratio:.3%})')
                axes[1, i].axis('off')
                
                print(f"{rotation:15s}: {positive_ratio:.3%}")
                break
        break  # 첫 번째 배치만
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'mask_rotation_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"마스크 회전 비교 저장: {save_path}")
    return True


def validate_dataloader(csv_path, batch_size=2):
    """데이터로더 기본 검증"""
    print("데이터로더 검증 중...")
    
    dataloader = get_sam_med3d_dataloader(
        csv_path=csv_path,
        batch_size=batch_size,
        is_train=False,
        verbose=False
    )
    
    for i, batch in enumerate(dataloader):
        if batch is None:
            continue
            
        image = batch["image"]
        mask = batch["mask"]
        
        print(f"배치 {i+1}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Patient IDs: {batch['patient_id']}")
        
        if i >= 2:  # 3개 배치만 확인
            break
    
    print("✅ 데이터로더 검증 완료")
    return True


if __name__ == "__main__":
    csv_path = "/storage01/user/IY/2_cerebral_hemorrhage/0_data/test_dataset_250701_chuncheon.csv"
    
    if os.path.exists(csv_path):
        print("=" * 60)
        print("SAM-Med3D 데이터로더 테스트 (패딩 기반)")
        print("=" * 60)
        print("🎯 개선사항:")
        print("   - 원본: 256×256×64 (H, W, D)")
        print("   - 타겟: 128×128×128 (D, H, W)")
        print("   - H, W: 리사이즈 (256→128, 공간 해상도 감소)")
        print("   - D: 패딩 (64→128, 원본 슬라이스 보존)")
        print("   - 장점: 인위적 슬라이스 생성 방지, 의학적 정확성")
        print("=" * 60)
        
        # 1. 기본 검증
        validate_dataloader(csv_path)
        
        # 2. 마스크 회전 비교
        test_mask_rotations(csv_path)
        
        # 3. 샘플 시각화
        visualize_samples(csv_path, num_samples=3)
        
        print("\n🎉 모든 테스트 완료!")
        print("📊 생성된 파일:")
        print("  - ./mask_test/mask_rotation_comparison.png")
        print("  - ./visualization/sample_*.png")
        print("\n💡 패딩 방식의 장점:")
        print("  ✅ 원본 64개 슬라이스 완전 보존")
        print("  ✅ 인위적 슬라이스 생성 방지")
        print("  ✅ 의학적으로 정확한 데이터 처리")
        print("  ✅ SAM-Med3D 논문 방식과 일치")
        
    else:
        print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}")