#!/bin/bash

#./less/scripts/get_info/grad/get_train_lora_grads.sh
# make into a for loop on ckpt
CKPTS=(22 44 66 88)
for CKPT in ${CKPTS[@]}; do
    gradient_type=adam
    train_file=./data/translation-less/enzh.jsonl
    model=./data/out/mistral-test-wmt/checkpoint-$CKPT
    output_path=./data/out/mistral-test-wmt/grads/train/checkpoint-$CKPT-$gradient_type
    dims=8192 # dimension of projection, can be a list

    if [[ ! -d $output_path ]]; then
        mkdir -p $output_path
    fi

    python3 -m less.data_selection.get_info \
    --train_file $train_file \
    --info_type grads \
    --model_path $model \
    --output_path $output_path \
    --gradient_projection_dimension $dims \
    --gradient_type $gradient_type \
    --max_samples 2000
done
