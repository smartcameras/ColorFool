#!/bin/bash

clear 

# Path to images and results
DATASET=../Dataset/
RESULT_PATH=./SegmentationResults/

# Segmentation model
MODEL_PATH=models
MASKTYPE=smooth

# Inference
python -u SemanticMasks.py \
  --model_path $MODEL_PATH \
  --dataset $DATASET \
  --arch_encoder resnet50dilated \
  --arch_decoder ppm_deepsup \
  --fc_dim 2048 \
  --result $RESULT_PATH \
  --mask_type $MASKTYPE \
  --gpu 0
