#!/bin/bash

DATA="HfO"

BENCHMARK_HOME="/data1/project/myungjin/0911_Hf_O/1"
DATADIR=${BENCHMARK_HOME}/datasets/${DATA}
OUTDIR=${BENCHMARK_HOME}/datasets/${DATA}

# If you want to prepare .lmdb which saves just atom cloud (containing just coordinates), set "cloud".
# Or if you want to have graph (containing coordinates as well as edges), set "graph"
outdata_type=$2

if [ $outdata_type == "cloud" ]; then

# Train/Valid/Test sets
python preprocess.py \
    --train-data ${DATADIR}/Trainset.xyz \
    --train-data-output-name train \
    --valid-data ${DATADIR}/Validset.xyz \
    --valid-data-output-name valid \
    --test-data ${DATADIR}/Testset.xyz \
    --test-data-output-name test \
    --out-path ${OUTDIR} \


elif [ $outdata_type == "graph" ]; then

rmax=$3
maxneigh=$4

# Train/Valid/Test sets
python preprocess.py \
    --train-data ${DATADIR}/Trainset.xyz \
    --train-data-output-name train \
    --valid-data ${DATADIR}/Validset.xyz \
    --valid-data-output-name valid \
    --test-data ${DATADIR}/Testset.xyz \
    --test-data-output-name test \
    --out-path ${OUTDIR} \
    --r-max $rmax \
    --max-neighbors $maxneigh \


python preprocess.py \
    --train-data /data1/project/yewon/5/datasets/HfO/Trainset.xyz \
    --train-data-output-name train \
    --valid-data /data1/project/yewon/5/datasets/HfO/Validset.xyz \
    --valid-data-output-name valid \
    --test-data /data1/project/yewon/5/datasets/HfO/Testset.xyz \
    --test-data-output-name test \
    --out-path /data1/project/yewon/5/datasets/HfO \
    --r-max 15.0 \
    --max-neighbors 50


fi
