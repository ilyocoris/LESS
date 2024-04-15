#!/bin/bash
# ./less/scripts/get_info/grad/get_eval_lora_grads.sh
# for validation data, we should always get gradients with sgd
CKPTS=(229 459 688 916)
for CKPT in ${CKPTS[@]}; do
    # task=$1 # tydiqa, mmlu
    gradient_type=sgd
    train_file=./data/out/medical/datasets/medical_meadow_wikidoc.jsonl
    model=./data/out/medical/mistral-medical/checkpoint-$CKPT
    output_path=./data/out/medical/mistral-medical/grads/val/checkpoint-$CKPT-$gradient_type
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
