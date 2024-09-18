# Distribued_Masked EVGF_MACE_AllENS

### Preprocessing
- lmdb 파일 생성 시 가까운 거리 순으로 이웃 원자 선택이 아닌 엣지의 원자 쌍 종류와 종류 별 거리 분포를 반영하여 샘플링
- Hf-O/O-Hf 선택 시 w로, O-O/Hf-Hf 선택 시 1-w의 비율로 선택하도록 구성

### Train
- Edge-Varient Graph Filter(EVGF)에 mask를 추가한 MACE 모델

### Evaluate
None

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
    None

3. Preprocess
    - /src/preprocessing/atoms_to_graphs.py : 원자 쌍 종류 및 거리 분포에 따라 이웃 원자 선택