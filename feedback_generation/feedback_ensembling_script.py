import os
import json
import torch
import argparse
from copy import deepcopy
from tqdm import tqdm
from typing import List, Dict, Union
from utils.prompt_utils import prompt_parser
from utils.model_utils import load_model_and_tokenizer_simplified, create_feedback_instruction
from feedback_ensembling.openai_feedback_ensembler import feedback_selector

def selector_parser(output_string: str) -> Union[str, Union[int, None], Union[int, None]]:
    '''
        Function to parse the output from feedback_selector
    '''

    # getting the rationale and score
    try:
        if '[THINKING_SEP]' in output_string:
            rationale, score = output_string.split('[THINKING_SEP]')
        else:
            score = output_string.split('\n')[-1]
            rationale = output_string
        
        # getting the best and worst index
        best_index, worst_index = score.strip().split(', ')
        best_index = int(best_index) - 1
        worst_index = int(worst_index) - 1
        return rationale, best_index, worst_index
    
    # if unsuccessful
    except:

        # checking first, second, third, fourth or fifth is in the string and getting their positions
        # the word appearing before is the first index and the later word is the second index
        output_string_lower = output_string.lower()
        word_list = ['first', 'second', 'third', 'fourth', 'fifth']
        present_word_list = []
        position_word_list = []
        
        # populating the present word list
        for word in word_list:
            if word in output_string_lower:
                present_word_list.append(word_list.index(word))
                position_word_list.append(output_string_lower.find(word))
        
        # check if the length of the present word list is 2 and 
        # the word with lower position is the best index
        if len(present_word_list) == 2:
            best_index = present_word_list[position_word_list.index(min(position_word_list))]
            worst_index = present_word_list[position_word_list.index(max(position_word_list))]
            return output_string, best_index, worst_index
        return output_string, None, None


def feedback_ensembler(
    input_file: str,
    model_path: str,
    adapter_path: str,
    exploration_size: int,
    feedback_model: str,
    feedback_ensembling_system_prompt: str,
    output_file: str,
    device: str = 'cuda',
    max_num_samples: int = 10,
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
    with open(feedback_ensembling_system_prompt, 'r') as f:
        feedback_ensembling_system_prompt = f.read()
        feedback_ensembling_prompt = prompt_parser(feedback_ensembling_system_prompt)

    # loading the model and adapter
    model, tokenizer = load_model_and_tokenizer_simplified(model_path, adapter_path)
    model.to(device)

    # iterating over the data list for new feedback generations
    new_data_list = []
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

        # generating the selected feedback
        feedback_selector_output = feedback_selector(
            feedback_ensembling_prompt,
            title,
            content,
            feedback_list,
            chat_model=feedback_model,
            temperature=0.0
        )
        rationale, best_index, worst_index = selector_parser(feedback_selector_output)

        # adding the new data to the list
        if best_index is not None and worst_index is not None and max([best_index, worst_index]) < len(feedback_list):
            new_data = deepcopy(data)
            new_data['output'] = [feedback_list[best_index], feedback_list[worst_index]]
            new_data['rationale'] = rationale
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
        '--feedback_model',
        type=str,
        help='OpenAI LLM used for selecting feedback',
        default='gpt-35-turbo'
    )
    parser.add_argument(
        '--feedback_ensembling_prompt',
        type=str,
        help='Path to the student system prompt file',
        default='feedback_ensembling/system_prompt_judge.txt'
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
    args = parser.parse_args()

    # calling the function for feedback ensembling
    feedback_ensembler(
        args.input_file,
        args.model_path,
        args.adapter_path,
        args.exploration_size,
        args.feedback_model,
        args.feedback_ensembling_prompt,
        args.output_file,
        args.device,
        args.max_num_samples
    )