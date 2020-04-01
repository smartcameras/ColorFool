#!/bin/bash

MODELS=(alexnet resnet18 resnet50)

clear
for model in "${MODELS[@]}"
do

	echo ColorFool attacking $model 
	python -W ignore ColorFool.py --model=$model
 
done
