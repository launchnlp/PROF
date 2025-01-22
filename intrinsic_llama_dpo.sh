#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=specificity_llama_refinement
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=11500m
#SBATCH --time=4-02:00:00
#SBATCH --account=wangluxy1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4

# Defining all the variables
model_path=/home/inair/data/llama3_models/llama-3-8b-instruct-hf
adapter_path=/home/inair/data/revision_saves/sft_mwrite_feedback_generation_8b_instruct
output_dir=/home/inair/data/revision_saves/mwrite_specificity_dpo_8b_instruct_llmjudge_gpt4
init_data_path=/home/inair/data/econ_data/assignment_2_processed/supervised_data/feedback_generation_train_unique.json
num_refinement=3
working_data=test_ranking_4
working_data_path=/home/inair/argument_revision/LLaMA-Factory/data/test_ranking_4.json
num_train_epochs=20
feedback_model=gpt-4-turbo
feedback_scoring_prompt=/home/inair/argument_revision/feedback_evaluator/pedagogy_prompt.txt
max_num_samples=100000
template=llama3
padding_side=left
device=cuda:0

# print all the variables
echo "Model path: $model_path"
echo "Adapter path: $adapter_path"
echo "Output directory: $output_dir"
echo "Initial data path: $init_data_path"
echo "Number of refinements: $num_refinement"
echo "Working data: $working_data"
echo "Working data path: $working_data_path"
echo "Number of training epochs: $num_train_epochs"
echo "Feedback model: $feedback_model"
echo "Feedback scoring prompt: $feedback_scoring_prompt"
echo "Max number of samples: $max_num_samples"
echo "Template: $template"
echo "Padding side: $padding_side"
echo "Device: $device"

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

    # running the refinement command
    echo "Running the refinement command"
    python feedback_generation/feedback_intrinsic_llama.py --input_file $output_dir/feedback_$((i-1)).json --model_path $model_path --adapter_path $output_dir/adapter_$((i-1)) --feedback_model $feedback_model --feedback_scoring_prompt $feedback_scoring_prompt --output_file $feedback_intermediate_path --device $device --max_num_samples $max_num_samples --template $template --padding_side $padding_side

    # copying the generated output_file to working_data_path
    cp $feedback_intermediate_path $working_data_path

    # running the training command for LLaMA-Factory
    echo "Running the training command"
    CUDA_VISIBLE_DEVICES=0 python LLaMA-Factory/src/train.py \
    --stage dpo \
    --do_train yes \
    --model_name_or_path $model_path \
    --adapter_name_or_path $output_dir/adapter_$((i-1)) \
    --dataset $working_data \
    --dataset_dir LLaMA-Factory/data \
    --template llama3 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir $output_dir/adapter_$((i)) \
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
    --num_train_epochs $num_train_epochs \
    --max_samples 1000 \
    --val_size 0.1 \
    --dpo_ftx 1.0 \
    --plot_loss yes \
    --fp16 yes

done

