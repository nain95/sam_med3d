import argparse
import torch
import os
import wandb
from datetime import datetime
from pathlib import Path

class Config:
    def __init__(self):
        # 기본 설정
        self.seed = 42
        self.epochs = 200
        self.batch_size = 2  # 3D 의료 데이터에 맞게 조정
        self.learning_rate = 1e-4  # SAM fine-tuning에 적합한 학습률
        self.weight_decay = 1e-4
        self.patience = 15
        
        # 데이터 경로 (상대 경로 사용)
        self.train_csv = ""
        self.val_csv = ""
        self.mask_dir = ""  # 마스크 디렉토리 경로 추가
        self.output_dir = "./outputs"
        
        # 모델 설정
        self.input_size = (128, 128, 128)  # (D, H, W)
        self.visual_tokens = 256  # 더 풍부한 representation
        self.freeze_encoder = False
        self.use_gradient_checkpointing = True  # 메모리 효율성
        
        # 데이터 처리 설정
        self.channel_method = "best_channel"
        self.window_level = 40.0
        self.window_width = 80.0
        self.mask_rotation = 'no_rotation'
        
        # Loss weights (적응적 조정 가능)
        self.dice_weight = 3.0
        self.focal_weight = 1.0
        self.iou_weight = 1.5
        self.iou_pred_weight = 0.05
        self.repr_weight = 0.01
        
        # 훈련 최적화 설정
        self.gradient_accumulation_steps = 4  # 효과적인 배치 크기 증가
        self.max_grad_norm = 1.0
        self.use_mixed_precision = True
        self.num_workers = 2
        
        # 스케줄러 설정
        self.scheduler_type = "reduce_on_plateau"
        self.scheduler_factor = 0.5
        self.scheduler_patience = 5
        self.min_lr = 1e-6
        
        # 기타
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # WandB 설정
        self.use_wandb = False
        self.wandb_project = "sam-med3d-brain-ct"
        self.wandb_entity = None
        self.wandb_run_name = None
        self.wandb_tags = ["sam-med3d", "brain-ct", "ich", "segmentation"]
        self.wandb_notes = "SAM-Med3D fine-tuning for Brain CT ICH segmentation"
        
        # Logging 설정
        self.log_images = True
        self.log_freq = 10  # 더 자주 로깅
        self.save_model_freq = 20
        self.print_freq = 10
        
    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="SAM-Med3D Fine-tuning Configuration")
        
        # 데이터 경로
        parser.add_argument("--train_csv", required=True, help="Path to training CSV file")
        parser.add_argument("--val_csv", required=True, help="Path to validation CSV file")
        parser.add_argument("--mask_dir", required=True, help="Directory containing mask files")
        parser.add_argument("--output_dir", default="./outputs", help="Output directory")
        
        # 훈련 설정
        parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
        parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
        parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        
        # 모델 설정
        parser.add_argument("--visual_tokens", type=int, default=256, help="Visual token dimension")
        parser.add_argument("--freeze_encoder", action="store_true", help="Freeze SAM encoder")
        parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True, 
                          help="Use gradient checkpointing")
        
        # 데이터 처리 설정
        parser.add_argument("--channel_method", default="best_channel", 
                          choices=["first_channel", "middle_channel", "best_channel"],
                          help="Channel selection method")
        parser.add_argument("--window_level", type=float, default=40.0, help="CT window level")
        parser.add_argument("--window_width", type=float, default=80.0, help="CT window width")
        parser.add_argument("--mask_rotation", default="no_rotation",
                          choices=["no_rotation", "rot_90_cw", "rot_90_ccw", "rot_180"],
                          help="Mask rotation option")
        
        # 최적화 설정
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                          help="Gradient accumulation steps")
        parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
        parser.add_argument("--use_mixed_precision", action="store_true", default=True,
                          help="Use mixed precision training")
        parser.add_argument("--num_workers", type=int, default=2, help="Number of data workers")
        
        # 스케줄러 설정
        parser.add_argument("--scheduler_type", default="reduce_on_plateau",
                          choices=["reduce_on_plateau", "cosine", "step"],
                          help="Learning rate scheduler type")
        parser.add_argument("--scheduler_factor", type=float, default=0.5,
                          help="Scheduler reduction factor")
        parser.add_argument("--scheduler_patience", type=int, default=5,
                          help="Scheduler patience")
        parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
        
        # Loss weights
        parser.add_argument("--dice_weight", type=float, default=3.0, help="Dice loss weight")
        parser.add_argument("--focal_weight", type=float, default=1.0, help="Focal loss weight")
        parser.add_argument("--iou_weight", type=float, default=1.5, help="IoU loss weight")
        parser.add_argument("--iou_pred_weight", type=float, default=0.05, help="IoU prediction loss weight")
        parser.add_argument("--repr_weight", type=float, default=0.01, help="Representation loss weight")
        
        # WandB 설정
        parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
        parser.add_argument("--wandb_project", default="sam-med3d-brain-ct", help="WandB project name")
        parser.add_argument("--wandb_entity", default=None, help="WandB entity (team/username)")
        parser.add_argument("--wandb_run_name", default=None, help="WandB run name")
        parser.add_argument("--wandb_tags", nargs='+', default=["sam-med3d", "brain-ct", "ich"], 
                          help="WandB tags")
        parser.add_argument("--wandb_notes", default="SAM-Med3D fine-tuning for Brain CT ICH segmentation", 
                          help="WandB run notes")
        
        # Logging 설정
        parser.add_argument("--log_images", action="store_true", default=True, 
                          help="Log sample images to wandb")
        parser.add_argument("--log_freq", type=int, default=10, 
                          help="Image logging frequency (epochs)")
        parser.add_argument("--save_model_freq", type=int, default=20, 
                          help="Model saving frequency to wandb (epochs)")
        parser.add_argument("--print_freq", type=int, default=10, 
                          help="Print frequency during training")
        
        args = parser.parse_args()
        
        config = cls()
        
        # 경로 설정
        config.train_csv = args.train_csv
        config.val_csv = args.val_csv
        config.mask_dir = args.mask_dir
        config.output_dir = args.output_dir
        
        # 훈련 설정
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        config.weight_decay = args.weight_decay
        config.patience = args.patience
        config.seed = args.seed
        
        # 모델 설정
        config.visual_tokens = args.visual_tokens
        config.freeze_encoder = args.freeze_encoder
        config.use_gradient_checkpointing = args.use_gradient_checkpointing
        
        # 데이터 처리 설정
        config.channel_method = args.channel_method
        config.window_level = args.window_level
        config.window_width = args.window_width
        config.mask_rotation = args.mask_rotation
        
        # 최적화 설정
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
        config.max_grad_norm = args.max_grad_norm
        config.use_mixed_precision = args.use_mixed_precision
        config.num_workers = args.num_workers
        
        # 스케줄러 설정
        config.scheduler_type = args.scheduler_type
        config.scheduler_factor = args.scheduler_factor
        config.scheduler_patience = args.scheduler_patience
        config.min_lr = args.min_lr
        
        # Loss weights
        config.dice_weight = args.dice_weight
        config.focal_weight = args.focal_weight
        config.iou_weight = args.iou_weight
        config.iou_pred_weight = args.iou_pred_weight
        config.repr_weight = args.repr_weight
        
        # WandB 설정
        config.use_wandb = args.use_wandb
        config.wandb_project = args.wandb_project
        config.wandb_entity = args.wandb_entity
        config.wandb_run_name = args.wandb_run_name
        config.wandb_tags = args.wandb_tags
        config.wandb_notes = args.wandb_notes
        
        # Logging 설정
        config.log_images = args.log_images
        config.log_freq = args.log_freq
        config.save_model_freq = args.save_model_freq
        config.print_freq = args.print_freq
        
        # 출력 디렉토리 생성
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        return config
    
    def get_experiment_name(self):
        """실험 이름 생성"""
        if self.wandb_run_name:
            return self.wandb_run_name
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"sam_med3d_ich_{timestamp}"
    
    def init_wandb(self):
        """WandB 초기화"""
        if not self.use_wandb:
            return None
            
        try:
            # WandB 설정
            wandb_config = {
                # 훈련 하이퍼파라미터
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "patience": self.patience,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_grad_norm": self.max_grad_norm,
                "use_mixed_precision": self.use_mixed_precision,
                
                # 모델 설정
                "input_size": self.input_size,
                "visual_tokens": self.visual_tokens,
                "freeze_encoder": self.freeze_encoder,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                
                # 데이터 처리 설정
                "channel_method": self.channel_method,
                "window_level": self.window_level,
                "window_width": self.window_width,
                "mask_rotation": self.mask_rotation,
                
                # 스케줄러 설정
                "scheduler_type": self.scheduler_type,
                "scheduler_factor": self.scheduler_factor,
                "scheduler_patience": self.scheduler_patience,
                "min_lr": self.min_lr,
                
                # Loss weights
                "dice_weight": self.dice_weight,
                "focal_weight": self.focal_weight,
                "iou_weight": self.iou_weight,
                "iou_pred_weight": self.iou_pred_weight,
                "repr_weight": self.repr_weight,
                
                # 기타
                "device": self.device,
                "seed": self.seed,
                "num_workers": self.num_workers,
                
                # 데이터 파일 이름만 저장 (보안상)
                "train_csv_name": Path(self.train_csv).name if self.train_csv else None,
                "val_csv_name": Path(self.val_csv).name if self.val_csv else None,
            }
            
            run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=self.get_experiment_name(),
                config=wandb_config,
                tags=self.wandb_tags,
                notes=self.wandb_notes,
                save_code=True,
                resume="allow"  # 중단된 실험 재개 허용
            )
            
            print(f"✅ WandB initialized:")
            print(f"   Project: {self.wandb_project}")
            print(f"   Run: {wandb.run.name}")
            print(f"   URL: {wandb.run.url}")
            
            return run
            
        except Exception as e:
            print(f"❌ WandB 초기화 실패: {e}")
            print("WandB 없이 훈련을 계속합니다.")
            self.use_wandb = False
            return None
    
    def validate_config(self):
        """설정 유효성 검사"""
        errors = []
        warnings = []
        
        # 필수 경로 확인
        if not self.train_csv or not Path(self.train_csv).exists():
            errors.append(f"Training CSV file not found: {self.train_csv}")
        
        if not self.val_csv or not Path(self.val_csv).exists():
            errors.append(f"Validation CSV file not found: {self.val_csv}")
            
        if not self.mask_dir or not Path(self.mask_dir).exists():
            errors.append(f"Mask directory not found: {self.mask_dir}")
        
        # 하이퍼파라미터 검사
        if self.batch_size < 1:
            errors.append("Batch size must be >= 1")
        
        if self.learning_rate <= 0:
            errors.append("Learning rate must be > 0")
            
        if self.epochs <= 0:
            errors.append("Epochs must be > 0")
        
        # 경고 사항
        if self.batch_size > 4:
            warnings.append("Batch size > 4 may cause GPU memory issues with 3D data")
            
        if self.learning_rate > 1e-3:
            warnings.append("Learning rate > 1e-3 may be too high for SAM fine-tuning")
        
        # 결과 출력
        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        if warnings:
            print("⚠️ Configuration warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        
        print("✅ Configuration validation passed")
        return True
    
    def to_dict(self):
        """Config를 딕셔너리로 변환"""
        return {
            key: value for key, value in self.__dict__.items() 
            if not key.startswith('_') and not callable(value)
        }
    
    def save_config(self, filepath=None):
        """설정을 JSON 파일로 저장"""
        import json
        
        if filepath is None:
            filepath = Path(self.output_dir) / "config.json"
        
        config_dict = self.to_dict()
        
        # Path 객체를 문자열로 변환
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            print(f"📝 Configuration saved: {filepath}")
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
    
    def __str__(self):
        """설정 정보를 읽기 쉽게 출력"""
        lines = ["=" * 60, "🔧 SAM-Med3D Configuration", "=" * 60]
        
        sections = {
            "📊 Training Settings": [
                f"Epochs: {self.epochs}",
                f"Batch Size: {self.batch_size}",
                f"Learning Rate: {self.learning_rate}",
                f"Weight Decay: {self.weight_decay}",
                f"Patience: {self.patience}",
                f"Seed: {self.seed}"
            ],
            "🧠 Model Settings": [
                f"Input Size: {self.input_size}",
                f"Visual Tokens: {self.visual_tokens}",
                f"Freeze Encoder: {self.freeze_encoder}",
                f"Gradient Checkpointing: {self.use_gradient_checkpointing}"
            ],
            "📁 Data Settings": [
                f"Train CSV: {Path(self.train_csv).name if self.train_csv else 'Not set'}",
                f"Val CSV: {Path(self.val_csv).name if self.val_csv else 'Not set'}",
                f"Mask Dir: {Path(self.mask_dir).name if self.mask_dir else 'Not set'}",
                f"Channel Method: {self.channel_method}",
                f"Window L/W: {self.window_level}/{self.window_width}"
            ],
            "⚙️ Optimization": [
                f"Gradient Accumulation: {self.gradient_accumulation_steps}",
                f"Max Grad Norm: {self.max_grad_norm}",
                f"Mixed Precision: {self.use_mixed_precision}",
                f"Scheduler: {self.scheduler_type}"
            ],
            "📈 Loss Weights": [
                f"Dice: {self.dice_weight}",
                f"Focal: {self.focal_weight}",
                f"IoU: {self.iou_weight}",
                f"IoU Pred: {self.iou_pred_weight}",
                f"Representation: {self.repr_weight}"
            ]
        }
        
        for section_name, section_items in sections.items():
            lines.append(f"\n{section_name}:")
            for item in section_items:
                lines.append(f"  {item}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
        