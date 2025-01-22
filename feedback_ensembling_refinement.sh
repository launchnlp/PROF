#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=ensembling_refinement
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=11500m
#SBATCH --time=7-02:00:00
#SBATCH --account=wangluxy1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Defining all the variables
model_path=/home/inair/data/llama-2-13b-hf
adapter_path=/home/inair/data/revision_saves/sft_essay_writing_350_gpt_35_turbo_13b
output_dir=/home/inair/data/revision_saves/sft_essay_writing_350_gpt_35_turbo_feedback_ensembling_dpo_13b
init_data_path=/home/inair/data/revision_supervised/sft_essay_writing_350_gpt_35_turbo_2.json
num_refinement=6
working_data=test_ranking
working_data_path=/home/inair/argument_revision/LLaMA-Factory/data/test_ranking_3.json
num_train_epochs=20
feedback_ensembling_prompt=feedback_ensembling/system_prompt_judge.txt
max_num_samples=1000
feedback_model=gpt-35-turbo
exploration_size=3

# creating the output directory if it does not exist
mkdir -p $output_dir

# copying the initial data to the output dir as training_data_0
cp $init_data_path $output_dir/feedback_0.json

# copying the adapter to the output dir as adapter_0
cp -r $adapter_path $output_dir/adapter_0

# iterating over the number of refinements
for i in $(seq 1 $num_refinement)
do
    echo "Refinement iteration: $i"
    
    # defining the path where intermediate feedback will be stored
    feedback_intermediate_path=$output_dir/feedback_$i.json
    echo "Feedback intermediate path: $feedback_intermediate_path"

    # running the feedback ensembling command
    echo "Running the feedback ensembling command"
    python feedback_generation/feedback_ensembling_script.py --input_file $output_dir/feedback_$((i-1)).json --model_path $model_path --adapter_path $output_dir/adapter_$((i-1)) --exploration_size $exploration_size --feedback_model $feedback_model --feedback_ensembling_prompt $feedback_ensembling_prompt --output_file $feedback_intermediate_path --device cuda --max_num_samples $max_num_samples

    # copying the generated output_file to working_data_path
    cp $feedback_intermediate_path $working_data_path

    # running the dpo training command for LLaMA-Factory
    echo "Running the training command"
    CUDA_VISIBLE_DEVICES=0 python LLaMA-Factory/src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path $model_path \
    --adapter_name_or_path $output_dir/adapter_$((i-1)) \
    --dataset $working_data \
    --dataset_dir LLaMA-Factory/data \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir $output_dir/adapter_$((i)) \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
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
    --num_train_epochs $num_train_epochs \
    --max_samples 1000 \
    --val_size 0.1 \
    --dpo_ftx 1.0 \
    --plot_loss \
    --fp16
done