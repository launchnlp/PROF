import os
import json
import torch
import argparse
from tqdm import tqdm
from typing import List, Dict, Union, Tuple
from utils.prompt_utils import prompt_parser
from utils.model_utils import load_model_and_tokenizer_simplified, create_feedback_instruction
from essay_evaluator.openai_essay_evaluator import batch_score_essay
from feedback_applicator.openai_feedback_applicator import batch_revise_essay_v2


def select_best_score_index(
	score_list: List[Dict[str, Union[str, float]]]
) -> Tuple[int, int]:
    '''
        Function responsible for selecting the best score index
    '''

    aggregated_score = []
    for score_dict in score_list:
        score_val = 0.0
        for attributes in score_dict.keys():
            score_val += score_dict[attributes]['score']
        aggregated_score.append(score_val)

    # find the best index
    best_index = aggregated_score.index(max(aggregated_score))
    
    # replace all the scores with 0.0 with 100.0 to find the worst index
    for index, score_val in enumerate(aggregated_score):
        if score_val == 0.0:
            aggregated_score[index] = 100.0

    # find the worst index
    worst_index = aggregated_score.index(min(aggregated_score))

    return best_index, worst_index


def feedback_filtering_raft_openai(
    input_file: str,
    model_path: str,
    adapter_path: str,
    student_model: str,
    exploration_size: int,
    reward_model: str,
    student_system_prompt: str,
    reward_system_prompt: str,
    output_file: str,
    model_device: str = 'cuda:0',
    max_num_samples: int = 10,
    student_type: str = 'normal',
    dpo: bool = False,
    student_temperature: float = 1.0
):
    '''
        Function responsible for filtering the feedback using RAFT
    '''

    # creating a target folder corresponding to output_file
    target_folder = os.path.dirname(output_file)
    os.makedirs(target_folder, exist_ok=True)

    # reading the input file
    if not os.path.exists(input_file):
        raise FileNotFoundError('Input file not found: {}', input_file)
    with open(input_file, 'r') as f:
        data_list = json.load(f)

    # reading the system prompt files
    with open(student_system_prompt, 'r') as f:
        student_system_prompt = f.read()
        student_prompt = prompt_parser(student_system_prompt)
    with open(reward_system_prompt, 'r') as f:
        reward_system_prompt = f.read()
        reward_prompt = prompt_parser(reward_system_prompt)

    # loading the model and adapter
    model, tokenizer = load_model_and_tokenizer_simplified(
        model_path,
        adapter_path,
        padding_side='left',
        template_name='llama3'
    )
    model.to(model_device)

    # iterating over the data list for new feedback generations
    for index, data in tqdm(enumerate(data_list)):

        if index >= max_num_samples:
            break
        
        # tokenize the prompt (to be changed later - this is incorrect schema)
        prompt_ids = create_feedback_instruction(
            tokenizer,
            data['system'],
            title='',
            content=data['input'],
            template='llama3'
        )
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(model_device)

        # generate the feedback
        with torch.no_grad():
            output_tensor = model.generate(
                prompt_tensor,
                max_new_tokens=900,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_k=0.0,
                top_p=0.95,
                num_return_sequences=exploration_size,
            )
            relevant_output = output_tensor[:, prompt_tensor.shape[1]:]
            feedback_list = tokenizer.batch_decode(relevant_output, skip_special_tokens=True)


        # batchify the data
        batched_title = [''] * len(feedback_list)
        batched_content = [data['input']] * len(feedback_list)
        batched_feedback = [[f] for f in feedback_list]

        # generating the revised essay list
        revised_essay_list = batch_revise_essay_v2(
            student_prompt,
            batched_title,
            batched_content,
            batched_feedback,
            student_model,
            student_temperature
        )

        # scoring the essays using the reward model
        score_list = batch_score_essay(
            reward_prompt,
            revised_essay_list,
            reward_model,
            temperature=0.0
        )

        # selecting the best index
        best_index, worst_index = select_best_score_index(score_list)

        # selecting the best feedback and updating the data
        if dpo:
            data['chosen'] = feedback_list[best_index]
            data['rejected'] = feedback_list[worst_index]
            data['revised_essay'] = [revised_essay_list[best_index], revised_essay_list[worst_index]]
            data['score'] = [score_list[best_index], score_list[worst_index]]
        else:
            data['output'] = feedback_list[best_index]
            data['revised_essay'] = revised_essay_list[best_index]
            data['score'] = score_list[best_index]

    # saving the filtered feedback
    with open(output_file, 'w') as f:
        json.dump(data_list[:max_num_samples], f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('sft_data_generator for feedback generation')
    parser.add_argument(
        '--input_file',
        type=str,
        help='Input file containing the data list',
        default='/home/inair/data/econ_data/assignment_2_processed/supervised_data/feedback_generation_train_unique.json'
    )
    parser.add_argument(
        '--model_path', 
        type=str,
        help='Model path for generating the candidates',
        default='/home/inair/data/llama3_models/llama-3-8b-instruct-hf'
    )
    parser.add_argument(
        '--adapter_path',
        type=str,
        help='Path to lora adapter',
        default='/home/inair/data/revision_saves/sft_mwrite_feedback_generation_8b_instruct'
    )
    parser.add_argument(
        '--student_model',
        type=str,
        help='Name of the openai student model that you have finetuned',
        default='gpt-35-turbo-student-applicator'
    )
    parser.add_argument(
        '--exploration_size',
        type=int,
        help='Number of candidates to be generated for each input',
        default=5
    )
    parser.add_argument(
        '--reward_model',
        type=str,
        help='Path to the reward model',
        default='gpt-35-turbo'
    )
    parser.add_argument(
        '--student_system_prompt',
        type=str,
        help='Path to the student system prompt file',
        default='/home/inair/data/econ_data/assignment_2_processed/student_applicator_system_prompt.txt'
    )
    parser.add_argument(
        '--reward_system_prompt',
        type=str,
        help='Path to the reward system prompt file',
        default='/home/inair/data/econ_data/assignment_2_processed/score_essay_system_prompt.txt'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Output file to store the feedback',
        default='/home/inair/data/revision_supervised/feedback_generation_train_unique_changed.json'
    )
    parser.add_argument(
        '--model_device',
        type=str,
        help='Device to be used for the model',
        default='cuda:0'
    )
    parser.add_argument(
        '--max_num_samples',
        type=int,
        help='Maximum number of samples to be processed',
        default=4
    )
    parser.add_argument(
        '--student_type',
        type=str,
        help='Type of student model to be used',
        default='normal'
    )
    parser.add_argument(
        '--dpo',
        action='store_true',
        help='Flag to enable DPO'
    )
    parser.add_argument(
        '--student_temperature',
        type=float,
        help='Temperature to be used for the student model',
        default=0.88
    )
    args = parser.parse_args()

    feedback_filtering_raft_openai(
        args.input_file,
        args.model_path,
        args.adapter_path,
        args.student_model,
        args.exploration_size,
        args.reward_model,
        args.student_system_prompt,
        args.reward_system_prompt,
        args.output_file,
        args.model_device,
        args.max_num_samples,
        args.student_type,
        args.dpo,
        args.student_temperature
    )
