# EVGF_MACE_UncertaintyENS

### Train
- Edge-Varient Graph Filter(EVGF)를 추가한 MACE 모델

### Evaluate
- Energy&Force(EF) : 체크포인트 리스트의 첫번째 체크포인트의 모델로 Energy와 Force 예측
- Uncertainty : 체크포인트 리스트의 모든 모델의 에너지의 표준편차로 uncertainty 계산
(yewon/Samsung_MLFF_Models/EVGF_MACE_UncertaintyENS/src/md_evaluate/base_evaluator.py의 self.ckpt_list에 체크포인트 추가)

### 명령어
1. Train : ./run_train.sh [GPU] MACE HfO
    ./run_train.sh 0 MACE HfOcd

2. Evaluate : ./run_evaluate_ef.sh [GPU] [체크 포인트 경로] [Leaderboard 데이터 경로]
    ./run_evaluate_ef.sh 0 ../../train_results/HfO/MACE/checkpoints/train-20240907_152048/best_checkpoint.pt ../../datasets/HfO/leaderboard.xyz


### 수정 파일
1. Train
    - /codebases/mace/mace/modules/blocks.py : EVGF 클래스 추가 및 Interaction 블럭(RealAgnosticResidualInteractionBlock)에 EVGF 적용
    - /codebases/mace/mace/modules/models.py : ScaleShiftMACE에 GSO 함수 생성 및 설정

2. Evaluate
    - /src/md_evaluate/base_evaluator.py : 체크포인트 리스트로 매개 변수 수정
    - /src/md_evaluate/calculator.py : __init__(), convert_atoms_to_batch(), caculate() 함수 수정, uncertainty 예측 추가
    - /src/md_evaluate/energy_force_evaluator.py : csv 저장 및 출력 추가
