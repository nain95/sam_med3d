import argparse
import torch
import os
import wandb

class Config:
    def __init__(self):
        # 기본 설정
        self.seed = 42
        self.epochs = 100
        self.batch_size = 4
        self.learning_rate = 3e-4
        self.patience = 10
        
        # 데이터 경로
        self.train_csv = ""
        self.val_csv = ""
        self.output_dir = "./outputs2"
        
        # 모델 설정
        self.input_size = (128, 128, 128)
        self.visual_tokens = 32
        
        # Loss weights
        self.dice_weight = 2.0
        self.focal_weight = 1.0
        self.iou_weight = 1.5
        self.iou_pred_weight = 0.05
        self.repr_weight = 0.02
        
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
        self.log_freq = 2000
        self.save_model_freq = 20
        
    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        
        # 기본 파라미터
        parser.add_argument("--train_csv", default="/storage01/user/IY/2_cerebral_hemorrhage/0_data/train_dataset_250701_chuncheon.csv")
        parser.add_argument("--val_csv", default="/storage01/user/IY/2_cerebral_hemorrhage/0_data/val_dataset_250701_chuncheon.csv")
        parser.add_argument("--epochs", type=int, default=200)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--output_dir", default="./outputs")
        
        # WandB 파라미터
        parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
        parser.add_argument("--wandb_project", default="sam-med3d-brain-ct", help="WandB project name")
        parser.add_argument("--wandb_entity", default=None, help="WandB entity (team/username)")
        parser.add_argument("--wandb_run_name", default=None, help="WandB run name")
        parser.add_argument("--wandb_tags", nargs='+', default=["sam-med3d", "brain-ct", "ich"], help="WandB tags")
        parser.add_argument("--wandb_notes", default="SAM-Med3D fine-tuning for Brain CT ICH segmentation", help="WandB run notes")
        
        # Logging 파라미터
        parser.add_argument("--log_images", action="store_true", default=True, help="Log sample images to wandb")
        parser.add_argument("--log_freq", type=int, default=10, help="Image logging frequency (epochs)")
        parser.add_argument("--save_model_freq", type=int, default=20, help="Model saving frequency to wandb (epochs)")
        
        # Loss weights
        parser.add_argument("--dice_weight", type=float, default=2.0)
        parser.add_argument("--focal_weight", type=float, default=1.0)
        parser.add_argument("--iou_weight", type=float, default=1.5)
        parser.add_argument("--iou_pred_weight", type=float, default=0.05)
        parser.add_argument("--repr_weight", type=float, default=0.02)
        
        args = parser.parse_args()
        
        config = cls()
        config.train_csv = args.train_csv
        config.val_csv = args.val_csv
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        config.output_dir = args.output_dir
        
        # WandB 설정
        config.use_wandb = args.use_wandb
        config.wandb_project = args.wandb_project
        config.wandb_entity = args.wandb_entity
        config.wandb_run_name = args.wandb_run_name
        config.wandb_tags = args.wandb_tags
        config.wandb_notes = args.wandb_notes
        config.log_images = args.log_images
        config.log_freq = args.log_freq
        config.save_model_freq = args.save_model_freq
        
        # Loss weights
        config.dice_weight = args.dice_weight
        config.focal_weight = args.focal_weight
        config.iou_weight = args.iou_weight
        config.iou_pred_weight = args.iou_pred_weight
        config.repr_weight = args.repr_weight
        
        os.makedirs(config.output_dir, exist_ok=True)
        return config
    
    def init_wandb(self):
        """WandB 초기화"""
        if not self.use_wandb:
            return None
            
        # WandB 설정
        wandb_config = {
            # 훈련 하이퍼파라미터
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "patience": self.patience,
            
            # 모델 설정
            "input_size": self.input_size,
            "visual_tokens": self.visual_tokens,
            
            # Loss weights
            "dice_weight": self.dice_weight,
            "focal_weight": self.focal_weight,
            "iou_weight": self.iou_weight,
            "iou_pred_weight": self.iou_pred_weight,
            "repr_weight": self.repr_weight,
            
            "device": self.device,
            "seed": self.seed,
            
            "train_csv": os.path.basename(self.train_csv),
            "val_csv": os.path.basename(self.val_csv),
        }
        
        run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=self.wandb_run_name,
            config=wandb_config,
            tags=self.wandb_tags,
            notes=self.wandb_notes,
            save_code=True  # 코드 저장
        )
        
        print(f"   Project: {self.wandb_project}")
        print(f"   Run: {wandb.run.name}")
        print(f"   URL: {wandb.run.url}")
        
        return run
    
    def to_dict(self):
        """Config를 딕셔너리로 변환"""
        return {
            key: value for key, value in self.__dict__.items() 
            if not key.startswith('_')
        }
        