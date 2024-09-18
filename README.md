# 2024_Samsung-AI-Challenge: Machine Learning Force Fields

MACE 모델을 기반으로 입력된 원자 구조로부터 시스템의 에너지 및 힘을 예측하고 모델의 에너지 예측 값에 대한 uncertainty quantification을 모두 출력하는 모델입니다.
기존의 MACE 모델에서 각 과정 별로 나누어 변형해보았습니다. <br>

1. **Preprocess 단계**에서는 엣지의 연결된 노드 쌍(Hf-Hf/Hf-O/O-O)을 구분하여 일정 비율 만큼의 종류 별 엣지가 이웃 노드로 설정되도록 변형하였습니다.<br>
2. **Train 단계**에서는 EVGF(Edge-Varient Graph Filter)를 추가하여 엣지 별 가중치를 함께 학습하도록 하였습니다.<br>
3. **Evaluate 단계**에서는 에너지 예측 값에 대한 Uncertainty를 출력하도록 하고 앙상블을 추가하였습니다.<br>

자세한 내용은 하단에 있는 내용을 참고해주세요.

<pre>
    #1. MACE_UncertaintyENS
        - Uncertainty에 대한 앙상블 추가

    #2. MACE_AllENS
        - 힘, 에너지와 함께 Uncetainty에 대한 앙상블 추가

    #3. EVGF_MACE_UncertaintyENS
        - EVGF(Edge-Varient Graph Filter) 추가
        - Uncertainty에 대한 앙상블 추가

    #4. EVGF_MACE_AllENS
        - EVGF(Edge-Varient Graph Filter) 추가
        - 힘, 에너지와 함께 Uncetainty에 대한 앙상블 추가

    #5. Masked_EVGF_MACE_ALLENS
        - Hf,O Masked EVGF(Edge-Varient Graph Filter) 추가
        - 힘, 에너지와 함께 Uncetainty에 대한 앙상블 추가

    #6. Distributed_Masked_EVGF_MACE
        - Hf,O Masked EVGF(Edge-Varient Graph Filter) 추가
        - 그래프 생성 시 거리 분포 별로 엣지 쌍(Hf-O,O-O,Hf-Hf) 구분하여 이웃 노드 선택   
</pre>



## 1️⃣ MACE_UncertaintyENS
### Train
- 일반 MACE 모델

### Evaluate
- Energy&Force(EF) : 체크포인트 리스트의 첫번째 모델로 Energy와 Force 예측
- Uncertainty : 체크포인트 리스트의 모든 모델의 에너지의 표준편차로 uncertainty 계산
(/src/md_evaluate/base_evaluator.py의 self.ckpt_list에 체크포인트 추가)

### 명령어
1. Train : ./run_train.sh [GPU] MACE HfO
    ./run_train.sh 0 MACE HfO
   - HfO 파일 복붙해서 추가하기

2. Evaluate : ./run_evaluate_ef.sh [GPU] [체크 포인트 경로] [Leaderboard 데이터 경로]
    ./run_evaluate_ef.sh 0 ../../train_results/HfO/MACE/checkpoints/train-20240908_062602/best_checkpoint.pt ../../datasets/HfO/leaderboard.xyz
   - 모델 훈련 폴더와 예측 폴더 동일하게 유지 (shift_taps도 바뀌면 안됨)
   - 명령어에 있는 체크 포인트는 입력에 포함 X
   - base_evaluator.py의 ckpt_list에 체크포인트 리스트 추가
     (체크 포인트마다 모델 종류가 다를 경우, 오류 발생)

### 수정 파일
1. Train
    - None

2. Evaluate
    - /src/md_evaluate/base_evaluator.py : 체크포인트 리스트로 매개 변수 수정
    - /src/md_evaluate/calculator.py : __init__(), convert_atoms_to_batch(), caculate() 함수 수정, uncertainty 예측 추가
    - /src/md_evaluate/energy_force_evaluator.py : csv 저장 및 출력 추가



## 2️⃣ MACE_AllENS
### Train
- 일반 MACE 모델

### Evaluate
- Energy&Force(EF) : 체크포인트 리스트의 모든 체크포인트의 모델의 평균으로 Energy와 Force 예측
- Uncertainty : 체크포인트 리스트의 모든 모델의 에너지의 표준편차로 uncertainty 계산
(/src/md_evaluate/base_evaluator.py의 self.ckpt_list에 체크포인트 추가)

### 명령어
1. Train : ./run_train.sh [GPU] MACE HfO
    ./run_train.sh 0 MACE HfO
   - HfO 파일 복붙해서 추가하기

2. Evaluate : ./run_evaluate_ef.sh [GPU] [체크 포인트 경로] [Leaderboard 데이터 경로]
    ./run_evaluate_ef.sh 0 ../../train_results/HfO/MACE/checkpoints/train-20240908_062602/best_checkpoint.pt ../../datasets/HfO/leaderboard.xyz
   - 모델 훈련 폴더와 예측 폴더 동일하게 유지 (shift_taps도 바뀌면 안됨)
   - 명령어에 있는 체크 포인트는 입력에 포함 X
   - base_evaluator.py의 ckpt_list에 체크포인트 리스트 추가
     (체크 포인트마다 모델 종류가 다를 경우, 오류 발생)

### 수정 파일
1. Train
    - None

2. Evaluate
    - /src/md_evaluate/base_evaluator.py : 체크포인트 리스트로 매개 변수 수정
    - /src/md_evaluate/calculator.py : __init__(), convert_atoms_to_batch(), caculate() 함수 수정, uncertainty 예측 추가
    - /src/md_evaluate/energy_force_evaluator.py : csv 저장 및 출력 추가

## 3️⃣ EVGF_MACE_UncertaintyENS
### Train
- Edge-Varient Graph Filter(EVGF)를 추가한 MACE 모델

### Evaluate
- Energy&Force(EF) : 체크포인트 리스트의 첫번째 체크포인트의 모델로 Energy와 Force 예측
- Uncertainty : 체크포인트 리스트의 모든 모델의 에너지의 표준편차로 uncertainty 계산
(/src/md_evaluate/base_evaluator.py의 self.ckpt_list에 체크포인트 추가)

### 명령어
1. Train : ./run_train.sh [GPU] MACE HfO
    ./run_train.sh 0 MACE HfO
   - HfO, codebases 파일 복붙해서 추가하기 (**/codebases/mace/mace/modules** Github 코드 유지)
   - EVGF shift_taps 변수 확인 (n_shift_taps = 3일 때 2 hop)
     (path: **RealAgnosticResidualInteractionBlock** /codebases/mace/mace/modules/blocks.py)

2. Evaluate : ./run_evaluate_ef.sh [GPU] [체크 포인트 경로] [Leaderboard 데이터 경로]
    ./run_evaluate_ef.sh 0 ../../train_results/HfO/MACE/checkpoints/train-20240907_152048/best_checkpoint.pt ../../datasets/HfO/leaderboard.xyz
   - 모델 훈련 폴더와 예측 폴더 동일하게 유지 (shift_taps도 바뀌면 안됨)
   - 명령어에 있는 체크 포인트는 입력에 포함 X
   - base_evaluator.py의 ckpt_list에 체크포인트 리스트 추가
     (체크 포인트마다 모델 종류가 다를 경우, 오류 발생)


### 수정 파일
1. Train
    - /codebases/mace/mace/modules/blocks.py : EVGF 클래스 추가 및 Interaction 블럭(RealAgnosticResidualInteractionBlock)에 EVGF 적용
    - /codebases/mace/mace/modules/models.py : ScaleShiftMACE에 GSO 함수 생성 및 설정

2. Evaluate
    - /src/md_evaluate/base_evaluator.py : 체크포인트 리스트로 매개 변수 수정
    - /src/md_evaluate/calculator.py : __init__(), convert_atoms_to_batch(), caculate() 함수 수정, uncertainty 예측 추가
    - /src/md_evaluate/energy_force_evaluator.py : csv 저장 및 출력 추가
  
## 4️⃣ EVGF_MACE_AllENS
### Train
- Edge-Varient Graph Filter(EVGF)를 추가한 MACE 모델

### Evaluate
- Energy&Force(EF) : 체크포인트 리스트의 모든 체크포인트의 모델의 평균으로 Energy와 Force 예측
- Uncertainty : 체크포인트 리스트의 모든 모델의 에너지의 표준편차로 uncertainty 계산
(/src/md_evaluate/base_evaluator.py의 self.ckpt_list에 체크포인트 추가)

### 명령어
1. Train : ./run_train.sh [GPU] MACE HfO
    ./run_train.sh 0 MACE HfO
   - HfO, codebases 파일 복붙해서 추가하기 (**/codebases/mace/mace/modules** Github 코드 유지)
   - EVGF shift_taps 변수 확인 (n_shift_taps = 3일 때 2 hop)
     (path: **RealAgnosticResidualInteractionBlock** /codebases/mace/mace/modules/blocks.py)

2. Evaluate : ./run_evaluate_ef.sh [GPU] [체크 포인트 경로] [Leaderboard 데이터 경로]
    ./run_evaluate_ef.sh 0 ../../train_results/HfO/MACE/checkpoints/train-20240907_152048/best_checkpoint.pt ../../datasets/HfO/leaderboard.xyz
   - 모델 훈련 폴더와 예측 폴더 동일하게 유지 (shift_taps도 바뀌면 안됨)
   - 명령어에 있는 체크 포인트는 입력에 포함 X
   - base_evaluator.py의 ckpt_list에 체크포인트 리스트 추가
     (체크 포인트마다 모델 종류가 다를 경우, 오류 발생)

### 수정 파일
1. Train
    - /codebases/mace/mace/modules/blocks.py : EVGF 클래스 추가 및 Interaction 블럭(RealAgnosticResidualInteractionBlock)에 EVGF 적용
    - /codebases/mace/mace/modules/models.py : ScaleShiftMACE에 GSO 함수 생성 및 설정

2. Evaluate
    - /src/md_evaluate/base_evaluator.py : 체크포인트 리스트로 매개 변수 수정
    - /src/md_evaluate/calculator.py : __init__(), convert_atoms_to_batch(), caculate() 함수 수정, uncertainty 예측 추가
    - /src/md_evaluate/energy_force_evaluator.py : csv 저장 및 출력 추가


## 5️⃣ Masked_EVGF_MACE_ALLENS
### Train
- Edge-Varient Graph Filter(EVGF)에 mask를 추가한 MACE 모델

### Evaluate
- Energy&Force(EF) : 체크포인트 리스트의 모든 체크포인트의 모델의 평균으로 Energy와 Force 예측
- Uncertainty : 체크포인트 리스트의 모든 모델의 에너지의 표준편차로 uncertainty 계산
(/src/md_evaluate/base_evaluator.py의 self.ckpt_list에 체크포인트 추가)

### 명령어
1. Train : ./run_train.sh [GPU] MACE HfO
    ./run_train.sh 0 MACE HfO
   - HfO 파일 복붙해서 추가하기
   - EVGF shift_taps 변수 확인 (n_shift_taps = 3일 때 2 hop)
     (path: **RealAgnosticResidualInteractionBlock** /codebases/mace/mace/modules/blocks.py)

2. Evaluate : ./run_evaluate_ef.sh [GPU] [체크 포인트 경로] [Leaderboard 데이터 경로]
    ./run_evaluate_ef.sh 0 ../../train_results/HfO/MACE/checkpoints/train-20240907_152048/best_checkpoint.pt ../../datasets/HfO/leaderboard.xyz
   - 모델 훈련 폴더와 예측 폴더 동일하게 유지 (shift_taps도 바뀌면 안됨)
   - base_evaluator.py의 ckpt_list에 체크포인트 리스트 추가
     (체크 포인트마다 모델 종류가 다를 경우, 오류 발생)

### 수정 파일
1. Train
    - /codebases/mace/mace/modules/blocks.py : Masked EVGF 클래스 추가 및 Interaction 블럭(RealAgnosticResidualInteractionBlock)에 EVGF 적용
    - /codebases/mace/mace/modules/models.py : ScaleShiftMACE에 GSO 함수 생성 및 설정

2. Evaluate
    - /src/md_evaluate/base_evaluator.py : 체크포인트 리스트로 매개 변수 수정
    - /src/md_evaluate/calculator.py : __init__(), convert_atoms_to_batch(), caculate() 함수 수정, uncertainty 예측 추가
    - /src/md_evaluate/energy_force_evaluator.py : csv 저장 및 출력 추가

## 6️⃣ Distributed_Masked_EVGF_MACE
### Preprocessing
- lmdb 파일 생성 시 가까운 거리 순으로 이웃 원자 선택이 아닌 엣지의 원자 쌍 종류와 종류 별 거리 분포를 반영하여 샘플링
- Hf-O/O-Hf 선택 시 w로, O-O/Hf-Hf 선택 시 1-w의 비율로 선택하도록 구성

### Train
- Edge-Varient Graph Filter(EVGF)에 mask를 추가한 MACE 모델

### Evaluate
- None

### 명령어
1. Train : ./run_train.sh [GPU] MACE HfO
    ./run_train.sh 0 MACE HfO
   - HfO 파일 복붙해서 추가하기
   - EVGF shift_taps 변수 확인 (n_shift_taps = 3일 때 2 hop)
     (path: **RealAgnosticResidualInteractionBlock** /codebases/mace/mace/modules/blocks.py)

2. Evaluate : ./run_evaluate_ef.sh [GPU] [체크 포인트 경로] [Leaderboard 데이터 경로]
    ./run_evaluate_ef.sh 0 ../../train_results/HfO/MACE/checkpoints/train-20240907_152048/best_checkpoint.pt ../../datasets/HfO/leaderboard.xyz
   

### 수정 파일
1. Train
    - /codebases/mace/mace/modules/blocks.py : EVGF 클래스 추가 및 Interaction 블럭(RealAgnosticResidualInteractionBlock)에 EVGF 적용
    - /codebases/mace/mace/modules/models.py : ScaleShiftMACE에 GSO 함수 생성 및 설정

2. Evaluate
    - None

3. Preprocess
    - /src/preprocessing/atoms_to_graphs.py : 원자 쌍 종류 및 거리 분포에 따라 이웃 원자 선택



