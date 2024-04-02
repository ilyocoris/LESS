#!/bin/bash

# ./less/scripts/data_selection/matching.sh

dim=8192
gradient_path=./data/out/mistral-test-wmt/grads/train/checkpoint-{}-adam/dim$dim/all_orig.pt
train_file_names=enzh
ckpts="22 44 66 88"
checkpoint_weights="1.6877e-05 1.2859e-05 7.7030e-06 2.5616e-06"

validation_gradient_path=./data/out/mistral-test-wmt/grads/val/checkpoint-{}-sgd/dim$dim/all_orig.pt
target_task_names=ende
output_path=/data/out/mistral-test-wmt/selected

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

python3 -m less.data_selection.matching \
--gradient_path $gradient_path \
--train_file_names $train_file_names \
--ckpts $ckpts \
--checkpoint_weights $checkpoint_weights \
--validation_gradient_path $validation_gradient_path \
--target_task_names $target_task_names \
--output_path $output_path
