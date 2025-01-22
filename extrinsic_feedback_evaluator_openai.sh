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
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4

# Defining all the variables
student_model=gpt-35-turbo-student-applicator
results_dir=/home/inair/data/revision_output/mwrite_specificity_dpo_8b_instruct_temp_1.0_combine_gpt4
num_refinement=3
student_system_prompt=/home/inair/data/econ_data/assignment_2_processed/student_applicator_system_prompt.txt
reward_model=gpt-4-turbo
reward_system_prompt=/home/inair/data/econ_data/assignment_2_processed/score_essay_system_prompt.txt
student_type=normal

# echo all the above parameters
echo "Extrinsic Evaluation using Openai api"
echo "Student model: $student_model"
echo "Output folder: $results_dir"
echo "Number of refinements: $num_refinement"
echo "Student system prompt: $student_system_prompt"
echo "Reward model: $reward_model"
echo "Reward system prompt: $reward_system_prompt"
echo "Student type: $student_type"

# creating a list of temperature and iterating over it
temperatures=(0.7 0.85 1.0)

# iterating from 0 to num_refinement
for repeat in {0..3}
do
    for temperature in ${temperatures[@]}
    do
        for i in $(seq 0 $num_refinement)
        do

            # logging the current iteration
            echo "Repeat iteration: $repeat"
            echo "Refinement iteration: $i"
            echo "Temperature: $temperature"

            # defining the target file path
            target_file=$results_dir/feedback_$i.json

            # applying the feedback using student feedback applicator
            python feedback_applicator/openai_feedback_applicator.py --input_file $target_file --output_file $target_file --system_file $student_system_prompt --student_type $student_type --chat_model $student_model --temperature $temperature --version_2

            # finally evaluating the essay
            python essay_evaluator/openai_essay_evaluator.py --input_file $target_file --chat_model $reward_model --key revised_output --system_file $reward_system_prompt
        done
    done
done