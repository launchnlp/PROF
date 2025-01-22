import os
import json
import torch
import argparse
from tqdm import tqdm
from typing import List, Dict, Union, Tuple
from utils.prompt_utils import prompt_parser
from utils.model_utils import load_model_and_tokenizer_simplified, create_feedback_instruction
from feedback_applicator.openai_feedback_applicator import batch_revise_essay
from essay_evaluator.openai_essay_evaluator import batch_score_essay

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

    return aggregated_score.index(max(aggregated_score)), aggregated_score.index(min(aggregated_score))


def feedback_filtering_raft(
    input_file: str,
    model_path: str,
    adapter_path: str,
    exploration_size: int,
    student_model: str,
    reward_model: str,
    student_system_prompt: str,
    reward_system_prompt: str,
    specificity_reward: bool,
    transferability_reward: bool,
    output_file: str,
    device: str = 'cuda',
    max_num_samples: int = 10,
    student_type: str = 'normal',
    dpo: bool = False
):
    '''
        Function responsible for filtering the feedback using RAFT
    '''

    # check if at least one reward is enabled
    if not specificity_reward and not transferability_reward:
        raise ValueError('At least one reward should be enabled')

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
    model, tokenizer = load_model_and_tokenizer_simplified(model_path, adapter_path)
    model.to(device)

    # iterating over the data list for new feedback generations
    for index, data in tqdm(enumerate(data_list)):

        if index >= max_num_samples:
            break
        
        # tokenize the prompt (to be changed later - this is incorrect schema)
        title = data['input'].split('\n\n')[0]
        content = '\n\n'.join(data['input'].split('\n\n')[1:])
        prompt_ids = create_feedback_instruction(
            tokenizer,
            data['system'],
            title=title,
            content=content,
        )
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(device)
        
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
                top_p=1.0,
                num_return_sequences=exploration_size
            )
            relevant_output = output_tensor[:, prompt_tensor.shape[1]:]
            feedback_list = tokenizer.batch_decode(relevant_output, skip_special_tokens=True)

        # storing the previous feedback into the feedback list
        feedback_list.append(data['output'])

        if specificity_reward:

            # applying the student feedback applicator for each feedback
            revised_essay_list = batch_revise_essay(
                student_prompt,
                [title for _ in range(exploration_size)],
                [content for _ in range(exploration_size)],
                feedback_list,
                student_model,
                temperature=0.0
            )
            if student_type == 'cognitive':
                revised_essay_list = [
                    f'{title}\n' + str(essay).split('[THINKING_SEP]')[1].strip() if '[THINKING_SEP]' in essay else essay for essay in revised_essay_list
                ]

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
            data['output'] = [feedback_list[best_index], feedback_list[worst_index]]
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
        default='/home/inair/data/revision_supervised/sft_essay_writing_350_gpt_35_turbo_2.json'
    )
    parser.add_argument(
        '--model_path', 
        type=str,
        help='Model path for generating the candidates',
        default='/home/inair/data/llama-2-7b-hf'
    )
    parser.add_argument(
        '--adapter_path',
        type=str,
        help='Path to lora adapter',
        default='/home/inair/data/revision_saves/sft_essay_writing_350_gpt_35_turbo_2'
    )
    parser.add_argument(
        '--exploration_size',
        type=int,
        help='Number of candidates to be generated for each input',
        default=5
    )
    parser.add_argument(
        '--student_model',
        type=str,
        help='Path to the student model',
        default='gpt-35-turbo'
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
        default='feedback_applicator/system_prompt_2.txt'
    )
    parser.add_argument(
        '--reward_system_prompt',
        type=str,
        help='Path to the reward system prompt file',
        default='essay_evaluator/system_prompt_2.txt'
    )
    parser.add_argument(
        '--specificity_reward',
        action='store_true',
        help='Flag to enable specificity reward'
    )
    parser.add_argument(
        '--transferability_reward',
        action='store_true',
        help='Flag to enable transferability reward'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Output file to store the feedback',
        default='/home/inair/data/revision_supervised/sft_essay_writing_350_gpt_35_turbo_changed.json'
    )
    parser.add_argument(
        '--device',
        type=str,
        help='Device to be used for the model',
        default='cuda'
    )
    parser.add_argument(
        '--max_num_samples',
        type=int,
        help='Maximum number of samples to be processed',
        default=1000
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
    args = parser.parse_args()

    feedback_filtering_raft(
        args.input_file,
        args.model_path,
        args.adapter_path,
        args.exploration_size,
        args.student_model,
        args.reward_model,
        args.student_system_prompt,
        args.reward_system_prompt,
        args.specificity_reward,
        args.transferability_reward,
        args.output_file,
        args.device,
        args.max_num_samples,
        args.student_type,
        args.dpo
    )