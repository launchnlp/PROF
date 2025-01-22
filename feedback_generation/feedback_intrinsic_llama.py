import os
import json
import torch
import argparse
from copy import deepcopy
from tqdm import tqdm
from typing import List, Dict, Union
from utils.prompt_utils import prompt_parser
from utils.model_utils import load_model_and_tokenizer_simplified, create_feedback_instruction
from feedback_evaluator.openai_feedback_evaluator import batch_score_feedback
from feedback_generation.feedback_filtering_llama import select_best_score_index


def feedback_intrinsic_llama_wrapper(
    input_file: str,
    model_path: str,
    adapter_path: str,
    exploration_size: int,
    feedback_model: str,
    feedback_scoring_prompt: str,
    output_file: str,
    device: str = 'cuda',
    max_num_samples: int = 10,
    template: str = 'llama3',
    padding_side: str = 'left'
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

    # reading the system prompt file for feedback ensembling
    with open(feedback_scoring_prompt, 'r') as f:
        feedback_scoring_prompt_text = f.read()
        feedback_scoring_prompt_messages = prompt_parser(feedback_scoring_prompt_text)

    # loading the model and adapter
    model, tokenizer = load_model_and_tokenizer_simplified(
        model_path,
        adapter_path,
        padding_side=padding_side,
        template_name=template
    )
    model.to(device)

    # iterating over the data list for new feedback generations
    new_data_list = []
    for index, data in tqdm(enumerate(data_list)):

        if index >= max_num_samples:
            break
        
        # tokenize the prompt (to be changed later - this is incorrect schema)
        prompt_ids = create_feedback_instruction(
            tokenizer,
            data['system'],
            title='',
            content=data['input'],
            template=template
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
                top_p=0.95,
                num_return_sequences=exploration_size
            )
            relevant_output = output_tensor[:, prompt_tensor.shape[1]:]
            feedback_list = tokenizer.batch_decode(relevant_output, skip_special_tokens=True)

        # scoring the feedback
        score_list = batch_score_feedback(
            feedback_scoring_prompt_messages,
            [data['input']] * len(feedback_list),
            feedback_list,
            chat_model=feedback_model,
            temperature=0.0
        )

        # selecting the best and worse index
        best_index, worst_index = select_best_score_index(score_list)

        # adding the new data to the list
        new_data = deepcopy(data)
        new_data['chosen'] = feedback_list[best_index]
        new_data['rejected'] = feedback_list[worst_index]
        new_data['chosen_score'] = score_list[best_index]
        new_data['rejected_score'] = score_list[worst_index]
        new_data_list.append(new_data)

    # saving the filtered feedback
    with open(output_file, 'w') as f:
        json.dump(new_data_list, f, indent=4)


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
        '--exploration_size',
        type=int,
        help='Number of candidates to be generated for each input',
        default=5
    )
    parser.add_argument(
        '--feedback_model',
        type=str,
        help='OpenAI LLM used for selecting feedback',
        default='gpt-35-turbo'
    )
    parser.add_argument(
        '--feedback_scoring_prompt',
        type=str,
        help='Path to the student system prompt file',
        default='/home/inair/argument_revision/feedback_evaluator/pedagogy_prompt.txt'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Output file to store the feedback',
        default='/home/inair/data/revision_supervised/sft_essay_writing_350_gpt_35_turbo_2.json'
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
        '--template',
        type=str,
        help='template to be used for the feedback generation llama',
        default='llama3'
    )
    parser.add_argument(
        '--padding_side',
        type=str,
        help='Padding side for the feedback generation',
        default='left'
    )
    args = parser.parse_args()

    # calling the function for feedback ensembling
    feedback_intrinsic_llama_wrapper(
        args.input_file,
        args.model_path,
        args.adapter_path,
        args.exploration_size,
        args.feedback_model,
        args.feedback_scoring_prompt,
        args.output_file,
        args.device,
        args.max_num_samples,
        args.template,
        args.padding_side
    )