# EVGF_MACE_UncertaintyENS

### Train
- Edge-Varient Graph Filter(EVGF)를 추가한 MACE 모델

### Evaluate
- Energy&Force(EF) : 체크포인트 리스트의 첫번째 체크포인트의 모델로 Energy와 Force 예측
- Uncertainty : 체크포인트 리스트의 모든 모델의 에너지의 표준편차로 uncertainty 계산
(/src/md_evaluate/base_evaluator.py의 self.ckpt_list에 체크포인트 추가)

### 제출 결과
- **ID#1** 0908_pred_by_mace_onlyUn.csv : mace + EdgeNet 3hop epoch 100 + Only Uncertainty ensemble (train-20240907_152048/best_checkpoint.pt, ckpt_ep60)

    |ID|EF-metric|OOD|
    |--|---------|---|
    |#1 | 5.6837979706 | 0.5816688889 |

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
