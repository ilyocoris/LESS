#!/bin/bash

#./less/scripts/train/warmup_lora_train.sh

source less/scripts/train/base_training_args.sh

data_dir=./data
model_path=mistralai/Mistral-7B-v0.1
percentage=0.1
data_seed=101
job_name=mistral-medical

output_dir=./data/out/medical/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

# train_files=("$data_dir/train/processed/flan_v2/flan_v2_data.jsonl"
#     "$data_dir/train/processed/cot/cot_data.jsonl"
#     "$data_dir/train/processed/dolly/dolly_data.jsonl"
# "$data_dir/train/processed/oasst1/oasst1_data.jsonl")

train_files=("$data_dir/out/medical/datasets/train.jsonl")


# use fsdp for large models
# if [[ $model_path == "meta-llama/Llama-2-13b-hf" ]]; then
#     base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
#     elif [[ $model_path == "mistralai/Mistral-7B-v0.1" ]]; then
#     base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
# fi

training_args="$base_training_args \
--model_name_or_path $model_path \
--output_dir $output_dir \
--percentage $percentage \
--data_seed $data_seed \
--train_files ${train_files[@]} 2>&1 | tee $output_dir/train.log"


#eval "$header" "$training_args"
# add `NCCL_P2P_DISABLE="1"` and `NCCL_IB_DISABLE="1"` to disable NCCL P2P and IB
# eval "NCCL_P2P_DISABLE=\"1\" NCCL_IB_DISABLE=\"1\" $header" "$training_args"
# add cuda visible devices 0 to 7
eval "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" "NCCL_P2P_DISABLE=\"1\" NCCL_IB_DISABLE=\"1\" $header" "$training_args"