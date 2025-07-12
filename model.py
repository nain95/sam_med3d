import torch
import torch.nn as nn
import torch.nn.functional as F
import medim
import medim.models.sam_med3d as model_utils
from medim.models._pretrain import check_and_download_weights_from_hf_url
import numpy as np

def overriding():
    """기존 BLIP2 코드와 동일한 overriding 함수"""
    original_load_pretrained_weights = model_utils.load_pretrained_weights

    def optimized_load_pretrained_weights(model, checkpoint_path, mode="nnunet", state_dict_key=None):
        ckpt_local_path = checkpoint_path
        if checkpoint_path.startswith("https://huggingface.co"):
            ckpt_local_path = check_and_download_weights_from_hf_url(checkpoint_path)
        if mode == 'nnunet':
            load_nnunet_pretrained_weights(model, ckpt_local_path)
        elif mode == 'torch':
            with open(ckpt_local_path, "rb") as f:
                state_dict = torch.load(f, map_location='cpu', weights_only=False)
            if state_dict_key:
                state_dict = state_dict[state_dict_key]
            model.load_state_dict(state_dict)
        else:
            raise NotImplementedError(f"mode {mode} for weight loading is not implemented")

    model_utils.load_pretrained_weights = optimized_load_pretrained_weights


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(pred.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth)
        return 1 - dice.mean()


class IoULoss(nn.Module):
    """개선된 IoU Loss - 텐서 연속성 보장"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        # reshape() 사용으로 연속성 문제 해결
        pred_flat = pred.reshape(pred.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()


class FocalLoss(nn.Module):
    """개선된 Focal Loss"""
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()


class SAMFineTuner(nn.Module):
    """개선된 SAM-Med3D Fine-tuning 모델 - 메모리 효율성 및 안정성 향상"""
    
    def __init__(self, checkpoint_path=None, representation_dim=256, freeze_encoder=False, 
                 use_gradient_checkpointing=True):
        super().__init__()
        overriding()
        
        self.representation_dim = representation_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # SAM-Med3D 로드 (올바른 URL)
        if checkpoint_path is None:
            ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"
        else:
            ckpt_path = checkpoint_path
            
        print(f"📥 SAM-Med3D 로딩 중... ({ckpt_path})")
        try:
            self.sam = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
            print(f"✅ SAM-Med3D 로드 완료")
            
            # Gradient checkpointing 활성화 (메모리 절약)
            if self.use_gradient_checkpointing and hasattr(self.sam, 'image_encoder'):
                if hasattr(self.sam.image_encoder, 'gradient_checkpointing_enable'):
                    self.sam.image_encoder.gradient_checkpointing_enable()
                    print("🔄 Gradient checkpointing 활성화")
                
        except Exception as e:
            print(f"❌ SAM-Med3D 로드 실패: {e}")
            raise e
        
        # Encoder freeze 옵션
        if freeze_encoder:
            self.freeze_sam_encoder()
        
        # SAM 모델의 실제 출력 차원 확인을 위한 더미 forward
        self._initialize_repr_head(representation_dim)
        
        # Loss functions with improved stability
        self.dice_loss = DiceLoss(smooth=1e-6)
        self.iou_loss = IoULoss(smooth=1e-6)
        self.focal_loss = FocalLoss(alpha=0.8, gamma=2.0)
        
        # 메모리 효율을 위한 설정
        self._setup_memory_optimization()
        
        print("🧠 개선된 SAM Fine-tuner 초기화 완료")
        print(f"   Representation dim: {representation_dim}")
        print(f"   Gradient checkpointing: {use_gradient_checkpointing}")
        print(f"   Encoder frozen: {freeze_encoder}")
    
    def _setup_memory_optimization(self):
        """메모리 최적화 설정"""
        # Mixed precision을 위한 설정
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                # 가중치 초기화 개선
                if hasattr(module, 'weight') and module.weight is not None:
                    if len(module.weight.shape) > 2:  # Conv layer
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    else:  # Linear layer
                        nn.init.xavier_normal_(module.weight)
                
                # Bias 초기화
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        print("🎯 메모리 최적화 설정 완료")

    def _initialize_repr_head(self, representation_dim):
        """SAM encoder 출력 차원에 맞춰 representation head 초기화 - 128³ 대응"""
        
        # 수정된 더미 입력 (128³)
        dummy_input = torch.randn(1, 1, 128, 128, 128)
        
        print("🔍 SAM encoder 출력 차원 확인 중...")
        
        with torch.no_grad():
            try:
                dummy_output = self.sam.image_encoder(dummy_input)
                sam_output_shape = dummy_output.shape
                print(f"📐 SAM image encoder 출력 shape: {sam_output_shape}")
                
                if len(sam_output_shape) == 5:  # [B, C, D, H, W]
                    sam_feature_dim = sam_output_shape[1]  # 256 또는 다른 값
                    
                    # Adaptive pooling으로 유연하게 처리
                    self.repr_head = nn.Sequential(
                        nn.AdaptiveAvgPool3d(1),  # [B, C, D, H, W] -> [B, C, 1, 1, 1]
                        nn.Flatten(),             # [B, C, 1, 1, 1] -> [B, C]
                        nn.Linear(sam_feature_dim, representation_dim),
                        nn.LayerNorm(representation_dim),
                        nn.GELU(),  # ReLU 대신 GELU 사용
                        nn.Dropout(0.1),
                        nn.Linear(representation_dim, representation_dim)
                    )
                    
                    print(f"✅ Representation head 초기화: {sam_feature_dim} -> {representation_dim}")
                    
                elif len(sam_output_shape) == 2:  # 이미 [B, C] 형태
                    sam_feature_dim = sam_output_shape[1]
                    
                    self.repr_head = nn.Sequential(
                        nn.Linear(sam_feature_dim, representation_dim),
                        nn.LayerNorm(representation_dim),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(representation_dim, representation_dim)
                    )
                    
                    print(f"✅ Representation head 초기화 (2D): {sam_feature_dim} -> {representation_dim}")
                    
                else:
                    # 예상치 못한 형태의 경우 adaptive 처리
                    total_features = torch.prod(torch.tensor(sam_output_shape[1:]))
                    
                    self.repr_head = nn.Sequential(
                        nn.AdaptiveAvgPool3d(1) if len(sam_output_shape) == 5 else nn.Identity(),
                        nn.Flatten(),
                        nn.Linear(int(total_features), representation_dim),
                        nn.LayerNorm(representation_dim),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(representation_dim, representation_dim)
                    )
                    
                    print(f"⚠️ 예상치 못한 형태, adaptive 처리: {total_features} -> {representation_dim}")
                    
            except Exception as e:
                print(f"❌ SAM 출력 차원 확인 실패: {e}")
                # 기본값으로 설정 (더 유연하게)
                self.repr_head = nn.Sequential(
                    nn.AdaptiveAvgPool3d(1),
                    nn.Flatten(),
                    nn.Linear(256, representation_dim),  # 기본값
                    nn.LayerNorm(representation_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(representation_dim, representation_dim)
                )
                print(f"🔧 기본 representation head 사용: 256 -> {representation_dim}")

    def forward(self, image, mask=None, return_features=False):
        """개선된 Forward pass - 메모리 효율성 및 안정성 향상"""
        batch_size = image.shape[0]
        device = image.device
        
        # 입력 검증
        if image.dim() != 5 or image.shape[1] != 1:
            raise ValueError(f"입력 이미지 형태가 잘못됨: {image.shape}, 예상: [B, 1, D, H, W]")
        
        # 입력 텐서 연속성 보장
        if not image.is_contiguous():
            image = image.contiguous()
        
        # Image encoding with error handling and memory optimization
        try:
            if self.use_gradient_checkpointing and self.training:
                # Gradient checkpointing 사용시
                image_embeddings = torch.utils.checkpoint.checkpoint(
                    self.sam.image_encoder, image, use_reentrant=False
                )
            else:
                image_embeddings = self.sam.image_encoder(image)
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ GPU 메모리 부족: {e}")
                print("💡 배치 크기를 줄이거나 gradient_accumulation_steps을 늘려보세요.")
            raise e
        except Exception as e:
            print(f"❌ Image encoder 에러: {e}")
            raise e
        
        # Representation features
        try:
            repr_features = self.repr_head(image_embeddings)
        except Exception as e:
            print(f"❌ Representation head 에러: {e}")
            print(f"   Image embeddings shape: {image_embeddings.shape}")
            raise e
        
        results = {
            'image_embeddings': image_embeddings,
            'representation_features': repr_features
        }
        
        # Features만 리턴하는 경우 (downstream tasks용)
        if return_features:
            return results
        
        # Segmentation head (마스크가 있을 때만)
        if mask is not None:
            # 마스크 입력 검증
            if mask.shape != image.shape:
                raise ValueError(f"마스크와 이미지 크기 불일치: mask {mask.shape} vs image {image.shape}")
            
            # 마스크 텐서 연속성 보장
            if not mask.is_contiguous():
                mask = mask.contiguous()
                
            try:
                # Prompt encoding (no prompts)
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=None, boxes=None, masks=None
                )
                
                # Mask prediction with memory optimization
                if self.use_gradient_checkpointing and self.training:
                    # Gradient checkpointing for mask decoder
                    def mask_decoder_forward(img_emb, img_pe, sparse_emb, dense_emb):
                        return self.sam.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=img_pe,
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb,
                            multimask_output=False,
                        )
                    
                    low_res_masks, iou_predictions = torch.utils.checkpoint.checkpoint(
                        mask_decoder_forward,
                        image_embeddings,
                        self.sam.prompt_encoder.get_dense_pe(),
                        sparse_embeddings,
                        dense_embeddings,
                        use_reentrant=False
                    )
                else:
                    low_res_masks, iou_predictions = self.sam.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=self.sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                
                # Upsampling to original input size
                pred_masks = F.interpolate(
                    low_res_masks,
                    size=image.shape[2:],  # (128, 128, 128)
                    mode='trilinear',
                    align_corners=False
                )
                
                results.update({
                    'pred_masks': pred_masks,
                    'iou_predictions': iou_predictions,
                    'low_res_masks': low_res_masks
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ Mask decoder GPU 메모리 부족: {e}")
                raise e
            except Exception as e:
                print(f"❌ Mask decoder 에러: {e}")
                raise e
        
        return results

    def calculate_loss(self, pred_masks, target_masks, iou_predictions=None, 
                      representation_features=None, loss_weights=None):
        """개선된 Loss 계산 - 텐서 연속성 보장"""
        if loss_weights is None:
            loss_weights = {
                'dice': 3.0,
                'focal': 1.0,
                'iou': 1.0,
                'iou_pred': 0.02,
                'representation': 0.01
            }
        
        # 텐서 연속성 보장
        if not pred_masks.is_contiguous():
            pred_masks = pred_masks.contiguous()
        if not target_masks.is_contiguous():
            target_masks = target_masks.contiguous()
        
        losses = {}
        
        # Segmentation losses
        dice_loss = self.dice_loss(pred_masks, target_masks)
        focal_loss = self.focal_loss(pred_masks, target_masks)
        iou_loss = self.iou_loss(pred_masks, target_masks)
        
        losses['dice_loss'] = dice_loss
        losses['focal_loss'] = focal_loss
        losses['iou_loss'] = iou_loss
        
        # IoU prediction loss
        if iou_predictions is not None:
            with torch.no_grad():
                pred_sigmoid = torch.sigmoid(pred_masks)
                # reshape() 사용으로 연속성 문제 해결
                target_flat = target_masks.reshape(target_masks.shape[0], -1)
                pred_flat = pred_sigmoid.reshape(pred_sigmoid.shape[0], -1)
                
                intersection = (pred_flat * target_flat).sum(dim=1)
                union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
                true_iou = intersection / (union + 1e-6)
            
            iou_pred_loss = F.mse_loss(iou_predictions.squeeze(), true_iou)
            losses['iou_pred_loss'] = iou_pred_loss
        else:
            losses['iou_pred_loss'] = torch.tensor(0.0, device=pred_masks.device)
        
        # Representation learning loss (L2 regularization)
        if representation_features is not None:
            repr_loss = torch.mean(torch.norm(representation_features, dim=1))
            losses['representation_loss'] = repr_loss
        else:
            losses['representation_loss'] = torch.tensor(0.0, device=pred_masks.device)
        
        # Total loss
        total_loss = (
            loss_weights['dice'] * dice_loss +
            loss_weights['focal'] * focal_loss +
            loss_weights['iou'] * iou_loss +
            loss_weights['iou_pred'] * losses['iou_pred_loss'] +
            loss_weights['representation'] * losses['representation_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses

    def extract_sam_encoder(self):
        """SAM encoder 추출 (다른 모델에서 사용하기 위해)"""
        return self.sam.image_encoder
    
    def freeze_sam_encoder(self):
        """SAM encoder freeze"""
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        print("🧊 SAM encoder frozen")
    
    def unfreeze_sam_encoder(self):
        """SAM encoder unfreeze"""
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = True
        print("🔥 SAM encoder unfrozen")
        
    def freeze_decoder(self):
        """SAM decoder freeze (encoder만 fine-tuning할 때)"""
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = False
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        print("🧊 SAM decoder & prompt encoder frozen")
    
    def unfreeze_decoder(self):
        """SAM decoder unfreeze"""
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = True
        print("🔥 SAM decoder & prompt encoder unfrozen")

    def get_trainable_params(self):
        """학습 가능한 파라미터 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'trainable_ratio': trainable_params / total_params
        }


def calculate_metrics(pred_masks, target_masks, threshold=0.5):
    """개선된 segmentation metrics 계산 - 텐서 연속성 보장"""
    pred_binary = (torch.sigmoid(pred_masks) > threshold).float()
    
    # 텐서 연속성 보장
    if not pred_binary.is_contiguous():
        pred_binary = pred_binary.contiguous()
    if not target_masks.is_contiguous():
        target_masks = target_masks.contiguous()
    
    # Batch-wise 계산
    batch_size = pred_binary.shape[0]
    metrics = {
        'dice': [],
        'iou': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': []
    }
    
    for i in range(batch_size):
        # reshape() 사용으로 연속성 문제 해결
        pred_flat = pred_binary[i].reshape(-1)
        target_flat = target_masks[i].reshape(-1)
        
        # Basic metrics
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        # Dice score
        dice = (2 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
        
        # IoU
        iou = intersection / (union + 1e-6)
        
        # Accuracy
        correct = (pred_flat == target_flat).sum()
        accuracy = correct / pred_flat.numel()
        
        # Confusion matrix elements
        tp = intersection  # True Positive
        fp = pred_flat.sum() - tp  # False Positive
        fn = target_flat.sum() - tp  # False Negative
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()  # True Negative
        
        # Sensitivity (Recall)
        sensitivity = tp / (tp + fn + 1e-6)
        
        # Specificity
        specificity = tn / (tn + fp + 1e-6)
        
        # Precision
        precision = tp / (tp + fp + 1e-6)
        
        metrics['dice'].append(dice.item())
        metrics['iou'].append(iou.item())
        metrics['accuracy'].append(accuracy.item())
        metrics['sensitivity'].append(sensitivity.item())
        metrics['specificity'].append(specificity.item())
        metrics['precision'].append(precision.item())
    
    # 평균 계산
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    
    return avg_metrics


def test_improved_model():
    """개선된 모델 차원 테스트 - 메모리 효율성 포함"""
    print("🧪 개선된 모델 차원 테스트...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   테스트 디바이스: {device}")
    
    # 초기 메모리 상태
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1e9
        print(f"   초기 GPU 메모리: {initial_memory:.2f} GB")
    
    try:
        # 모델 생성
        print("🤖 모델 생성 중...")
        model = SAMFineTuner(
            representation_dim=256,
            freeze_encoder=False, 
            use_gradient_checkpointing=True
        ).to(device)
        
        if torch.cuda.is_available():
            model_memory = torch.cuda.memory_allocated() / 1e9
            print(f"   모델 로드 후 GPU 메모리: {model_memory:.2f} GB")
        
        # 파라미터 정보
        param_info = model.get_trainable_params()
        print(f"📊 모델 파라미터 정보:")
        print(f"   전체: {param_info['total_params']:,}")
        print(f"   학습가능: {param_info['trainable_params']:,}")
        print(f"   고정: {param_info['frozen_params']:,}")
        print(f"   학습가능 비율: {param_info['trainable_ratio']:.1%}")
        
        # 테스트 데이터 (올바른 128³ 크기)
        batch_size = 1  # 메모리 절약을 위해 배치 크기 줄임
        image = torch.randn(batch_size, 1, 128, 128, 128, device=device)
        mask = torch.randint(0, 2, (batch_size, 1, 128, 128, 128), device=device).float()
        
        print(f"🔍 입력 테스트 - Image: {image.shape}, Mask: {mask.shape}")
        
        if torch.cuda.is_available():
            data_memory = torch.cuda.memory_allocated() / 1e9
            print(f"   데이터 로드 후 GPU 메모리: {data_memory:.2f} GB")
        
        # Forward pass
        print("⏩ Forward pass 테스트...")
        model.eval()  # 평가 모드로 설정
        
        with torch.no_grad():  # 그래디언트 계산 비활성화
            results = model(image, mask)
        
        print("✅ Forward pass 성공!")
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
        
        if torch.cuda.is_available():
            forward_memory = torch.cuda.memory_allocated() / 1e9
            print(f"   Forward 후 GPU 메모리: {forward_memory:.2f} GB")
        
        # Loss 계산 (훈련 모드로 전환)
        print("💰 Loss 계산 테스트...")
        model.train()
        
        # 작은 배치로 Loss 계산
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            results_train = model(image, mask)
            losses = model.calculate_loss(
                results_train['pred_masks'],
                mask,
                results_train.get('iou_predictions'),
                results_train['representation_features']
            )
        
        print("✅ Loss 계산 성공!")
        for key, value in losses.items():
            if torch.is_tensor(value):
                print(f"   {key}: {value.item():.4f}")
        
        # Metrics 계산
        print("📊 Metrics 계산 테스트...")
        with torch.no_grad():
            metrics = calculate_metrics(results_train['pred_masks'], mask)
        print("✅ Metrics 계산 성공!")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
        
        # Feature extraction 테스트
        print("🔍 Feature extraction 테스트...")
        model.eval()
        with torch.no_grad():
            features = model(image, return_features=True)
        print(f"   Image embeddings: {features['image_embeddings'].shape}")
        print(f"   Representation features: {features['representation_features'].shape}")
        
        # 최종 메모리 상태
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1e9
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"   최종 GPU 메모리: {final_memory:.2f} GB")
            print(f"   최대 GPU 메모리 사용량: {peak_memory:.2f} GB")
        
        print("\n🎉 모든 개선된 모델 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        
        if torch.cuda.is_available():
            error_memory = torch.cuda.memory_allocated() / 1e9
            print(f"   오류 시점 GPU 메모리: {error_memory:.2f} GB")
        
        return False
    
    finally:
        # 메모리 정리
        if 'model' in locals():
            del model
        if 'image' in locals():
            del image
        if 'mask' in locals():
            del mask
        if 'results' in locals():
            del results
        if 'losses' in locals():
            del losses
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            cleaned_memory = torch.cuda.memory_allocated() / 1e9
            print(f"   정리 후 GPU 메모리: {cleaned_memory:.2f} GB")


if __name__ == "__main__":
    test_improved_model()