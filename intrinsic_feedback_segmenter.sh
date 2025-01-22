#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=instrinsic_feedback_evaluator
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=11500m
#SBATCH --time=7-02:00:00
#SBATCH --account=wangluxy1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4

input_folder=/home/inair/data/revision_output/mwrite_specificity_dpo_8b_instruct_temp_1.0_combine_gpt4_init
system_file=/home/inair/argument_revision/feedback_intrinsic_analysis/feedback_element_system_prompt.txt
consistency_segment_file=/home/inair/argument_revision/feedback_intrinsic_analysis/element_consistency_system_prompt.txt
chat_model=gpt-4-turbo
temperature=0.0
key=feedback
consistency_key=feedback_segment
num_refinement=3

# printing the input_folder
echo "Intrinsic Segmentation and Consistency Evaluation of the feedback using OpenAI"
echo "Input folder: $input_folder"

# iterating over the refinement steps
for i in $(seq 0 $num_refinement)
do
    # echoing the iteration number
    echo "Refinement iteration: $i"

    # defining the target file path
    target_file=$input_folder/feedback_$i.json

    # evaluating the feedback
    python feedback_intrinsic_analysis/openai_feedback_segmenter.py --input_file $target_file --system_file $system_file --chat_model $chat_model --temperature $temperature --key $key

    # evaluating the consistency
    python feedback_intrinsic_analysis/openai_feedback_segmenter.py --input_file $target_file --system_file $consistency_segment_file --key $consistency_key --chat_model $chat_model --temperature $temperature

done
