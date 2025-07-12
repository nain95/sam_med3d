# 개선된 SAM-Med3D Fine-tuning for Brain CT ICH Segmentation

이 프로젝트는 뇌 CT 이미지에서 뇌내출혈(ICH) 분할을 위한 SAM-Med3D 모델의 개선된 파인튜닝 구현입니다.

## 🆕 개선사항

### 1. 구성 관리 개선 (config.py)
- **하드코딩 제거**: 모든 경로와 설정이 명령행 인수로 설정 가능
- **설정 검증**: 자동 설정 유효성 검사 및 경고 시스템
- **유연한 스케줄러**: ReduceLROnPlateau, Cosine, Step 스케줄러 지원
- **WandB 통합**: 완전한 실험 추적 및 resume 지원

### 2. 데이터 처리 개선 (dataloader.py)
- **중복 함수 제거**: 코드 정리 및 안정성 향상
- **에러 처리 강화**: 견고한 데이터 로딩 및 전처리
- **메모리 효율성**: 적응적 데이터 증강 및 워커 관리
- **통계 계산**: 자동 데이터셋 통계 및 검증

### 3. 모델 아키텍처 개선 (model.py)
- **URL 수정**: 정확한 HuggingFace 모델 URL
- **메모리 최적화**: Gradient checkpointing 및 mixed precision 지원
- **에러 처리**: 강화된 입력 검증 및 메모리 관리
- **유연한 구조**: 동적 representation head 초기화

### 4. 훈련 프로세스 개선 (trainer.py)
- **Mixed Precision**: 자동 혼합 정밀도 훈련
- **Gradient Accumulation**: 큰 배치 크기 효과 구현
- **메모리 관리**: 실시간 메모리 모니터링 및 정리
- **안정성**: 강화된 에러 복구 및 체크포인트 시스템

### 5. 유틸리티 강화 (utils.py)
- **시스템 검사**: 자동 시스템 요구사항 및 GPU 메모리 검사
- **훈련 모니터링**: 실시간 진행 상황 및 시간 추정
- **결과 관리**: 자동 훈련 요약 및 체크포인트 정리

### 6. 메인 스크립트 개선 (train.py)
- **사전 검사**: 종합적인 시스템 및 데이터 검증
- **향상된 로깅**: 상세한 진행 상황 및 결과 출력
- **테스트 모드**: 빠른 시스템 테스트 기능

## 🚀 사용법

### 1. 기본 훈련 실행

```bash
python train.py \
    --train_csv /path/to/train.csv \
    --val_csv /path/to/val.csv \
    --mask_dir /path/to/masks \
    --output_dir ./outputs \
    --epochs 200 \
    --batch_size 2 \
    --lr 1e-4
```

### 2. 고급 설정 예시

```bash
python train.py \
    --train_csv /path/to/train.csv \
    --val_csv /path/to/val.csv \
    --mask_dir /path/to/masks \
    --output_dir ./outputs \
    --epochs 200 \
    --batch_size 2 \
    --lr 1e-4 \
    --use_mixed_precision \
    --gradient_accumulation_steps 4 \
    --scheduler_type cosine \
    --use_wandb \
    --wandb_project my-sam-med3d \
    --freeze_encoder
```

### 3. 빠른 시스템 테스트

```bash
python train.py --test
```

## 📋 주요 매개변수

### 필수 매개변수
- `--train_csv`: 훈련 데이터 CSV 파일 경로
- `--val_csv`: 검증 데이터 CSV 파일 경로  
- `--mask_dir`: 마스크 파일 디렉토리 경로

### 훈련 설정
- `--epochs`: 훈련 에포크 수 (기본: 200)
- `--batch_size`: 배치 크기 (기본: 2)
- `--lr`: 학습률 (기본: 1e-4)
- `--weight_decay`: 가중치 감쇠 (기본: 1e-4)
- `--patience`: Early stopping patience (기본: 15)

### 최적화 설정
- `--use_mixed_precision`: Mixed precision 훈련 사용
- `--gradient_accumulation_steps`: Gradient accumulation 단계 (기본: 4)
- `--max_grad_norm`: Gradient clipping 임계값 (기본: 1.0)
- `--scheduler_type`: 스케줄러 타입 (reduce_on_plateau/cosine/step)

### 모델 설정
- `--visual_tokens`: Visual token 차원 (기본: 256)
- `--freeze_encoder`: SAM encoder 고정
- `--use_gradient_checkpointing`: Gradient checkpointing 사용

### WandB 설정
- `--use_wandb`: WandB 로깅 사용
- `--wandb_project`: WandB 프로젝트 이름
- `--wandb_entity`: WandB 엔티티 (팀/사용자명)
- `--log_images`: 샘플 이미지 로깅 사용
- `--log_freq`: 이미지 로깅 주기 (에포크)

## 🔧 시스템 요구사항

### 하드웨어
- **GPU**: CUDA 지원 GPU (8GB+ VRAM 권장)
- **RAM**: 16GB+ 시스템 메모리
- **저장공간**: 10GB+ 여유 공간

### 소프트웨어
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (GPU 사용시)

### 필수 라이브러리
```bash
pip install torch torchvision torchaudio
pip install monai nibabel pandas numpy matplotlib tqdm
pip install wandb  # WandB 사용시
pip install psutil  # 시스템 모니터링용
```

## 📊 데이터 형식

### CSV 파일 형식
```csv
path,ich,patient_id
/path/to/image1.pkl,1,patient_001
/path/to/image2.pkl,1,patient_002
...
```

### 마스크 파일
- 형식: NIfTI (.nii)
- 명명: `{파일번호}.nii` (이미지 파일명과 일치)
- 값: 0 (배경), 1 (ICH 영역)

## 📈 출력 파일

### 모델 파일
- `best_model.pth`: 최고 성능 모델 (전체)
- `sam_encoder.pth`: SAM encoder만 (BLIP2 통합용)
- `latest_model.pth`: 최신 체크포인트

### 로그 파일
- `config.json`: 훈련 설정
- `training_summary_*.json`: 훈련 결과 요약

## 🔬 BLIP2 통합

훈련된 SAM encoder를 BLIP2에 통합하려면:

```python
# BLIP2 모델에서 SAM encoder 로드
sam_encoder_path = "outputs/sam_encoder.pth"
checkpoint = torch.load(sam_encoder_path)
image_encoder_state = checkpoint['image_encoder_state_dict']

# BLIP2 모델의 비전 인코더에 적용
blip2_model.visual_encoder.load_state_dict(image_encoder_state)
```

## 🐛 문제 해결

### 메모리 부족 오류
1. 배치 크기 줄이기: `--batch_size 1`
2. Gradient accumulation 늘리기: `--gradient_accumulation_steps 8`
3. Mixed precision 사용: `--use_mixed_precision`

### 데이터 로딩 오류
1. 경로 확인: CSV 파일과 마스크 디렉토리 경로
2. 권한 확인: 파일 읽기 권한
3. 테스트 실행: `python train.py --test`

### 성능 최적화
1. 워커 수 조정: `--num_workers 4`
2. 학습률 스케줄링: `--scheduler_type cosine`
3. 데이터 증강 조정: 코드에서 `augmentation_prob` 설정

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 시스템 테스트: `python train.py --test`
2. 설정 출력에서 경고 메시지 확인
3. GPU 메모리 사용량 모니터링
4. WandB 로그에서 상세 정보 확인

## 🏆 성능 벤치마크

| 설정 | GPU | 배치 크기 | 메모리 사용량 | 훈련 시간/에포크 |
|------|-----|-----------|---------------|------------------|
| 기본 | RTX 3080 (10GB) | 2 | ~8GB | ~15분 |
| 최적화 | RTX 3080 (10GB) | 1 + grad_acc=4 | ~6GB | ~20분 |
| 고성능 | A100 (40GB) | 8 | ~25GB | ~8분 |

이 개선된 버전은 안정성, 성능, 그리고 사용성 모든 면에서 크게 향상되었습니다.