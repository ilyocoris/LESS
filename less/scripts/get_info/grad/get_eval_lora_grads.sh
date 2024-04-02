#!/bin/bash

# for validation data, we should always get gradients with sgd
CKPTS=(22 44 66 88)
for CKPT in ${CKPTS[@]}; do
    # task=$1 # tydiqa, mmlu
    gradient_type=sgd
    train_file=./data/translation-less/ende.jsonl
    model=./data/out/mistral-test-wmt/checkpoint-$CKPT
    output_path=./data/out/mistral-test-wmt/grads/val/checkpoint-$CKPT-$gradient_type
    dims=8192

    if [[ ! -d $output_path ]]; then
        mkdir -p $output_path
    fi

    python3 -m less.data_selection.get_info \
    --info_type grads \
    --model_path $model \
    --output_path $output_path \
    --gradient_projection_dimension $dims \
    --gradient_type $gradient_type \
    --train_file $train_file
    # --data_dir data_dir
    # --task $task \
done
