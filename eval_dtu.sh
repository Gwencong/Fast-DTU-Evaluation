#!/usr/bin/env bash
PRED_DIR="data/Predict/dmvsnet"
GT_DIR="data/SampleSet/MVS Data"

CUDA_VISIBLE_DEVICES=2 python eval_dtu.py \
    --voxel_factor 1.28 \
    --pred_dir $PRED_DIR \
    --gt_dir "$GT_DIR" \
    --save \
    --num_workers 1 \