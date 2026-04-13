# Taxonomy-Aware ESM2: 분류학 기반 단백질 기능 예측

ESM2 단백질 언어 모델에 NCBI 분류학 계통 정보를 교차 어텐션(cross-attention)으로 결합해 Gene Ontology(GO) 용어를 다중 레이블 분류 방식으로 예측합니다. CAFA(Critical Assessment of Functional Annotation) 챌린지를 타겟으로 설계되었습니다.

---

## 모델 구조

```
단백질 서열 (FASTA)
      │
      ▼
ESM2 백본 (650M)  ──LoRA──▶  서열 임베딩  (B, L, 1280)
                                    │
                          [Cross-Attention]  ◀──  분류학 임베딩  (B, 7, 1280)
                                    │
                           LayerNorm + Residual
                                    │
                           Masked Mean Pooling
                                    │
                          Linear Classifier (→ 40,122 GO 용어)
```

### 구성 요소

| 모듈 | 설명 |
|---|---|
| **ESM2 백본** | `facebook/esm2_t33_650M_UR50D` (6억 5천만 파라미터), LoRA로 파인튜닝 |
| **분류학 인코더** | 7단계 분류 계층(문·강·목·과·속·종·아종)마다 별도 임베딩 레이어(dim=128), 선형 투영으로 1280차원으로 확장 |
| **교차 어텐션 융합** | 서열을 Query, 분류 임베딩을 Key/Value로 사용하는 8-head MHA. 각 서열 위치에 분류학 컨텍스트를 주입 |
| **분류 헤드** | Masked mean pooling → Linear(1280 → 40,122), 다중 레이블 출력 |

---

## 사용된 머신러닝 기법

### 1. LoRA (Low-Rank Adaptation)
ESM2의 모든 파라미터를 갱신하는 대신 어텐션 레이어의 Query·Value 행렬에만 저순위(rank=8) 어댑터를 삽입합니다. 훈련 가능 파라미터를 650M → 약 800K 수준으로 줄이면서 사전 학습 지식을 유지합니다.

```
W_new = W_frozen + α/r · (A · B)   (A: r×d, B: d×r, lora_alpha=32)
```

### 2. Cross-Attention 기반 분류학 융합
단순 concatenation이나 FiLM 방식 대신, 서열 임베딩이 7개 분류 계층 임베딩 전체에서 관련 정보를 선택적으로 가져오도록 교차 어텐션을 설계했습니다. 같은 속(genus)이라도 종마다 기능이 다를 수 있으므로, 어텐션이 어느 계층을 얼마나 참조할지를 학습합니다.

### 3. Asymmetric Focal Loss + IC 가중치
GO 용어는 레이블 불균형이 매우 극심합니다(일부 용어는 수만 단백질, 대부분은 소수). 두 가지 기법을 조합해 대응합니다.

- **비대칭 포컬 손실**: 양성 샘플과 음성 샘플에 다른 집중 계수(γ) 적용. 음성 오분류에 더 높은 페널티(`γ_neg=4, γ_pos=0`)를 줘서 희귀 용어 학습을 촉진.
- **Asymmetric Clipping**: 예측 확률이 낮은 음성 샘플의 기여도를 잘라내(`clip=0.05`) 불필요한 그래디언트를 제거.
- **IC(Information Content) 가중치**: `IA.tsv`에서 각 GO 용어의 정보량을 로드해 손실에 곱함. 상위 일반 용어(낮은 IC)보다 하위 특이적 용어(높은 IC)를 더 중요하게 학습. 가중치는 비-제로 항목의 평균이 1이 되도록 정규화.

### 4. 혼합 정밀도 훈련 (AMP)
`torch.cuda.amp`의 `autocast` + `GradScaler`를 사용해 FP16으로 순전파를 수행하고 FP32로 파라미터를 갱신합니다. VRAM 사용량을 줄이면서 속도를 높입니다.

### 5. Weighted F-max (CAFA 평가 지표)
임계값을 0.01~1.0 구간에서 스캔하며 IC-가중 정밀도·재현율을 계산하고, F1이 최대가 되는 임계값에서의 점수를 보고합니다. 검증 손실과 별도로 이 지표로 최고 모델을 선택합니다.

---

## 데이터 전처리

### 입력 데이터 형식

| 파일 | 형식 | 설명 |
|---|---|---|
| `*.fasta` | `>sp\|Q12345\|PROT_HUMAN OX=9606 ...` | 단백질 서열, 헤더에 TaxID 포함 |
| `term.tsv` | `EntryID \t GO:xxxxxxx \t aspect \t evidence` | 다중 레이블 GO 주석 |
| `species_vectors.tsv` | `TaxID \t [i0, i1, ..., i6]` | TaxID → 7단계 분류 인덱스 벡터 |
| `IA.tsv` | `GO:xxxxxxx \t IC값` | GO 용어별 정보량 |

### 전처리 파이프라인

#### 1단계: 분류학 어휘 구축 (`src/build_taxonomy_vocab.py`)
NCBI taxonomy dump를 파싱해 각 분류 계층(문~아종)마다 `{이름: 인덱스}` 형태의 JSON 어휘 파일을 생성합니다. 미등록 분류군은 인덱스 0(unknown)으로 처리합니다.

```
Phylum: 51개 | Class: 146개 | Order: 285개 | Family: 873개
Genus: 2,638개 | Species: 11,644개 | Subspecies: 2,638개
```

#### 2단계: TaxID → 벡터 변환 (`src/vectorize_species.py`)
각 TaxID를 7단계 분류 계층 인덱스의 정수 배열 `[phylum_idx, ..., subspecies_idx]`로 변환해 `species_vectors.tsv`에 저장합니다. 훈련 시 O(1) 조회를 위한 룩업 테이블로 사용됩니다.

#### 3단계: GO 어휘 구축 (`src/build_go_vocab.py`)
OBO 형식의 GO 온톨로지 파일을 파싱해 유효한 GO 용어 40,122개를 `{GO:xxxxxxx: 인덱스}` JSON으로 저장합니다.

#### 4단계: 서열 토크나이징
ESM2 토크나이저로 최대 1,024 토큰으로 잘라냄. 짧은 서열은 `max_length`에 맞게 패딩. `attention_mask`로 패딩 위치를 마스킹해 mean pooling 시 무시합니다.

#### 5단계: GO 레이블 전파 (옵션)
CSR 형식의 조상 행렬(`go_ancestors.npz`)을 사용해 주석된 GO 용어의 모든 상위(부모·조상) 용어로 레이블을 확장합니다. GO의 true path rule을 반영: 특정 기능에 주석된 단백질은 그 상위 일반 기능도 자동으로 가집니다.

```
GO:0006096 (해당 단백질에 직접 주석)
    └─▶ GO:0006090 (조상)
          └─▶ GO:0006091 (조상)  → 세 용어 모두 양성 레이블로 확장
```

#### 6단계: 다중 레이블 인코딩
각 단백질에 대해 크기 40,122의 0/1 이진 벡터를 생성합니다. 양성 GO 용어 위치에만 1.0을 할당합니다.

---

## 실행 방법

### 환경 설치
```bash
pip install -r requirements.txt
```

### 로컬 학습 (소형 ESM 모델, 빠른 테스트)
```bash
python local_train.py
```

### 전체 학습
```bash
python src/train.py \
  --data_path dataset/ \
  --esm_model_name facebook/esm2_t33_650M_UR50D \
  --epochs 20 --batch_size 64 --lr 1e-4 \
  --use_lora True --lora_rank 8 \
  --gamma_neg 4 --gamma_pos 0 --clip 0.05 \
  --output_dir outputs --mlflow_dir mlruns
```

### 실험 추적
```bash
mlflow ui --backend-store-uri sqlite:///src/mlflow.db
```

### 출력 결과물

| 파일 | 내용 |
|---|---|
| `outputs/best_model_loss.pth` | 검증 손실 기준 최고 모델 |
| `outputs/best_model_fmax.pth` | Weighted F-max 기준 최고 모델 |
| `outputs/gpu_preds_epoch_N.tsv` | 에폭별 예측 결과 |
| `outputs/evaluation_metrics_epoch_N.tsv` | 임계값별 F1 점수 |

---

## 프로젝트 구조

```
src/
  model.py                  # TaxonomyAwareESM, AsymmetricLoss
  dataset.py                # ProteinTaxonomyDataset
  train.py                  # 학습 루프, 평가, MLflow 연동
  asymmetric_loss.py        # 손실 함수, IC 가중치 로더
  build_taxonomy_vocab.py   # 분류 어휘 구축
  vectorize_species.py      # TaxID → 정수 벡터
  build_go_vocab.py         # GO 어휘 구축
  cafa_evaluator_driver.py  # CAFA 평가 래퍼
  model_analysis.py         # 교차 어텐션 가중치 시각화
  CAFA-evaluator-PK/        # CAFA 공식 평가 툴킷 (서브모듈)
dataset/
  learning_superset/        # 훈련 FASTA (Git LFS) + term TSV
  validation_superset/      # 검증 FASTA + term TSV
  taxon_embedding/          # species_vectors.tsv, vocab/*.json
  go_info/                  # OBO 온톨로지, 조상 행렬
  IA.tsv                    # GO 용어별 정보량 (40,122개)
local_train.py              # 로컬 실행 진입점
azure_ml_taxonomy_aware_submit.ipynb  # Azure ML 클라우드 학습 제출
```
