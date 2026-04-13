# Taxonomy-Aware ESM2

단백질 기능 예측 모델입니다. ESM2에 NCBI 분류학 계통 정보를 교차 어텐션으로 결합해 Gene Ontology(GO) 용어를 예측합니다. CAFA 챌린지를 목표로 작업했습니다.

---

## 배경

단백질 기능 예측에서 진화적 계통 정보는 꽤 유의미한 힌트가 됩니다. 같은 서열이라도 어떤 분류군에 속하느냐에 따라 발현되는 기능이 달라질 수 있기 때문입니다. ESM2가 서열 자체의 구조적 특징을 잡아낸다면, 분류학 정보는 그 서열이 어떤 생물학적 맥락에 있는지를 알려주는 역할을 합니다. 이 둘을 어떻게 자연스럽게 합치느냐가 이 프로젝트의 핵심 문제였습니다.

## 내 기여

GO 데이터 수집 및 유효 데이터 정제를 담당했습니다. UniProt에서 GO 주석 데이터를 수집하고, obsolete 용어 제거와 증거 코드 필터링을 통해 학습에 실제로 쓸 수 있는 데이터셋을 만들었습니다. GO 온톨로지 OBO 파일을 파싱해 40,122개의 유효 용어를 추출하고, true path rule에 따른 레이블 전파 파이프라인도 직접 구성했습니다.

---

## 모델 구조

```
단백질 서열 (FASTA)
      │
      ▼
ESM2 백본 (650M, LoRA)  →  서열 임베딩 (B, L, 1280)
                                    │
                          Cross-Attention  ←  분류학 임베딩 (B, 7, 1280)
                                    │
                           LayerNorm + Residual
                                    │
                           Masked Mean Pooling
                                    │
                          Linear (→ 40,122 GO 용어)
```

ESM2(`esm2_t33_650M_UR50D`)는 LoRA로 파인튜닝합니다. Query·Value 행렬에만 rank=8 어댑터를 붙여서 훈련 파라미터를 650M에서 약 800K로 줄였습니다.

분류학 인코더는 문·강·목·과·속·종·아종 7단계 계층마다 별도의 임베딩 레이어(dim=128)를 둡니다. 교차 어텐션에서 서열이 Query, 분류 임베딩이 Key/Value가 되어 각 서열 위치가 어느 분류 계층을 얼마나 참조할지를 학습합니다. 단순 concatenation보다 계층별 기여를 유연하게 조정할 수 있다는 게 장점입니다.

## 학습

**손실 함수**: Asymmetric Focal Loss + IC 가중치 조합을 씁니다. GO 레이블은 불균형이 심해서 (흔한 용어는 수만 개, 희귀 용어는 수십 개) 일반 BCE로는 흔한 용어에만 수렴하는 경향이 있습니다. 음성 샘플에 더 높은 집중 계수(`γ_neg=4`)를 주고, IA.tsv의 정보량(IC) 값으로 용어별 가중치를 다르게 줘서 희귀하지만 의미 있는 용어의 학습을 강제합니다.

**평가**: CAFA 기준인 Weighted F-max를 씁니다. 임계값을 0.01~1.0 구간에서 스캔하면서 IC-가중 정밀도·재현율을 계산하고 F1이 최대인 지점을 보고합니다. 검증 손실과 별개로 이 지표로 최고 모델을 따로 저장합니다.

**혼합 정밀도**: AMP(`autocast` + `GradScaler`)로 FP16 순전파, FP32 파라미터 갱신을 사용합니다.

## 데이터 전처리

데이터는 크게 세 가지 소스를 전처리해서 만들었습니다.

**GO 데이터** (`src/build_go_vocab.py`): OBO 파일에서 obsolete 용어를 제거하고 유효한 GO 용어 40,122개를 추출했습니다. UniProt 주석에서는 증거 코드 기준으로 실험적으로 검증된 항목 위주로 정제했습니다. 주석된 GO 용어의 모든 조상 용어로 레이블을 확장하는 전파(propagation)도 적용했는데, CSR 희소 행렬로 구현해서 40,122 × 40,122 크기에도 처리 속도가 나옵니다.

**분류학 데이터** (`src/build_taxonomy_vocab.py`, `src/vectorize_species.py`): NCBI taxonomy dump를 파싱해서 각 계층(문~아종)별 어휘를 만들고, TaxID를 7개 정수 인덱스 배열로 변환해 룩업 테이블로 저장합니다.

**서열 데이터**: FASTA 헤더의 `OX=` 필드에서 TaxID를 파싱하고, ESM2 토크나이저로 최대 1,024 토큰으로 자릅니다. 어노테이션이 없거나 분류학 벡터가 없는 항목은 학습에서 제외합니다.

---

## 실행

```bash
pip install -r requirements.txt

# 로컬 테스트 (8M 모델)
python local_train.py

# 전체 학습
python src/train.py \
  --data_path dataset/ \
  --esm_model_name facebook/esm2_t33_650M_UR50D \
  --epochs 20 --batch_size 64 --lr 1e-4 \
  --use_lora True --lora_rank 8 \
  --output_dir outputs

# 실험 추적
mlflow ui --backend-store-uri sqlite:///src/mlflow.db
```

## 파일 구조

```
src/
  model.py                 # TaxonomyAwareESM, AsymmetricLoss
  dataset.py               # ProteinTaxonomyDataset
  train.py                 # 학습 루프, 평가, MLflow
  asymmetric_loss.py       # 손실 함수, IC 가중치
  build_taxonomy_vocab.py  # 분류 어휘 구축
  vectorize_species.py     # TaxID → 정수 벡터
  build_go_vocab.py        # GO 어휘 구축
  cafa_evaluator_driver.py # CAFA 평가
  CAFA-evaluator-PK/       # CAFA 공식 평가 툴킷
dataset/
  learning_superset/       # 훈련 데이터 (Git LFS)
  validation_superset/     # 검증 데이터
  taxon_embedding/         # species_vectors.tsv, vocab/
  go_info/                 # OBO, 조상 행렬
  IA.tsv                   # GO 용어별 IC 값
```
