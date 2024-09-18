# 2024_Samsung-MLFF-Challenge
## Machine Learning Force Fields
MACE 모델을 변형하여 입력된 원자 구조로부터 시스템의 에너지 및 힘을 예측하고 모델의 에너지 예측 값에 대한 uncertainty quantification을 모두 출력하는 모델의 레포지터리입니다.

1. MACE_UncertaintyENS
    - Uncertainty에 대한 앙상블 추가

2. MACE_AllENS
    - 힘, 에너지와 함께 Uncetainty에 대한 앙상블 추가

3. EVGF_MACE_UncertaintyENS
    - EVGF(Edge-Varient Graph Filter) 추가
    - Uncertainty에 대한 앙상블 추가

4. EVGF_MACE_AllENS
    - EVGF(Edge-Varient Graph Filter) 추가
    - 힘, 에너지와 함께 Uncetainty에 대한 앙상블 추가

5. Masked_EVGF_MACE_ALLENS
    - Hf,O Masked EVGF(Edge-Varient Graph Filter) 추가
    - 힘, 에너지와 함께 Uncetainty에 대한 앙상블 추가

5. Distributed_Masked_EVGF_MACE
    - Hf,O Masked EVGF(Edge-Varient Graph Filter) 추가
    - 그래프 생성 시 거리 분포 별로 엣지 쌍(Hf-O,O-O,Hf-Hf) 구분하여 이웃 노드 선택



