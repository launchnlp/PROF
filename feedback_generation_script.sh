#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=specificity_refinement
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=11500m
#SBATCH --time=00-10:00:00
#SBATCH --account=wangluxy1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Defining all the variables
model_path=/home/inair/data/llama3_models/llama-3-8b-instruct-hf
adapter_folder=/home/inair/data/revision_saves/mwrite_specificity_dpo_8b_instruct_temp_1.0_combine_gpt4_init
num_refinement=3
feedback_system_prompt=/home/inair/data/econ_data/assignment_2_processed/peer_review_system_prompt.txt
results_dir=/home/inair/data/revision_output/mwrite_specificity_dpo_8b_instruct_temp_1.0_combine_gpt4_init_ctrl
test_file=/home/inair/data/econ_data/assignment_2_processed/supervised_data/feedback_generation_test.json
template=llama3
max_num_samples=1000

# echo adapter folder
echo "Adapter folder: $adapter_folder"

# creating the results directory
mkdir -p $results_dir

for i in $(seq 0 $num_refinement)
do

    # logging the current iteration
    echo "Refinement iteration: $i"

    # defining the target file path
    target_file=$results_dir/feedback_$i.json

    # adapter_path
    adapter_path=$adapter_folder/adapter_$i

    # generating the feedback using the above adapter
    python feedback_generation/llama_feedback_generator.py --input_file $test_file --output_file $target_file --model_path $model_path --adapter_path $adapter_path --device cuda:0 --system_file $feedback_system_prompt --template $template --max_num_samples $max_num_samples
done


