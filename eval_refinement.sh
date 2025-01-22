#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=specificity_refinement
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
adapter_folder=/home/inair/data/revision_saves/sft_essay_writing_350_gpt_35_turbo_specificity_dpo_13b
num_refinement=3
feedback_system_prompt=feedback_generation/system_prompt_2.txt
student_model=gpt-35-turbo
student_system_prompt=feedback_applicator/system_prompt_2.txt
reward_model=gpt-35-turbo
reward_system_prompt=essay_evaluator/system_prompt_2.txt
results_dir=/home/inair/data/revision_output/sft_essay_writing_350_gpt_35_turbo_specificity_dpo_13b
test_file=/home/inair/data/essay_test.json
student_type=cognitive

# echo adapter folder
echo "Adapter folder: $adapter_folder"

# creating the results directory
mkdir -p $results_dir

# iterating from 0 to num_refinement
for i in $(seq 0 $num_refinement)
do

    # logging the current iteration
    echo "Refinement iteration: $i"

    # defining the target file path
    target_file=$results_dir/feedback_$i.json

    # adapter_path
    adapter_path=$adapter_folder/adapter_$i

    # generating the feedback using the above adapter
    python feedback_generation/llama_feedback_generator.py --input_file $test_file --output_file $target_file --model_path $model_path --adapter_path $adapter_path --device cuda:0 --system_file $feedback_system_prompt

    # applying the feedback using student feedback applicator
    python feedback_applicator/openai_feedback_applicator.py --input_file $target_file --output_file $target_file --chat_model $student_model --system_file $student_system_prompt --student_type $student_type

    # finally evaluating the essay
    python essay_evaluator/openai_essay_evaluator.py --input_file $target_file --chat_model $reward_model --key revised_output --system_file $reward_system_prompt --student_type $student_type
done