#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=specificity_llama_refinement
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=11500m
#SBATCH --time=7-02:00:00
#SBATCH --account=wangluxy1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4

CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --stage dpo \
    --do_train yes \
    --model_name_or_path /home/inair/data/llama3_models/llama-3-8b-instruct-hf \
    --adapter_name_or_path /home/inair/data/revision_saves/sft_mwrite_feedback_generation_8b_instruct \
    --dataset mwrite_reward_modeling \
    --dataset_dir data \
    --template llama3 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir /home/inair/data/revision_saves/dpo_rm_mwrite_feedback_generation_8b_instruct \
    --overwrite_cache yes \
    --overwrite_output_dir yes \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 1e-5 \
    --num_train_epochs 20 \
    --max_samples 1000 \
    --val_size 0.1 \
    --dpo_ftx 1.0 \
    --plot_loss yes \
    --fp16 yes