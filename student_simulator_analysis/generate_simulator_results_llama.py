import os
import json
import torch
import argparse
from tqdm import tqdm
from utils.model_utils import (
    load_model_and_tokenizer_simplified,
    create_student_applicator_instruction,
    create_feedback_instruction
)

def generate_simulator_results_llama_wrapper(
    input_file: str,
    model_path: str,
    adapter_path: str,
    system_file: str,
    output_file: str,
    template: str,
    padding_side: str,
    device: str,
    processed_input: bool = False,
    num_explorations: bool = 1
) -> None:
    '''
        Augments the input file data with student simulator results
    '''

    # check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f'Input file {input_file} not found')
    with open(input_file, 'r') as f:
        data_list = json.load(f)

    # check if the system file exists
    if not os.path.exists(system_file):
        raise FileNotFoundError(f'System file {system_file} not found')
    with open(system_file, 'r') as f:
        system_prompt = f.read()

    # load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer_simplified(
        model_name=model_path,
        adapter_name=adapter_path,
        padding_side=padding_side,
        template_name=template
    )
    model.to(device)

    # setting up temperature grid from 0 to 1 inclusive with 0.05 step with numpy
    temperature_grid = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    # iterating over the data
    for data in tqdm(data_list):
        
        # create a key for storing the student student simulator results
        data['student_simulator_results'] = {}

        # creating the instruction
        if not processed_input:
            prompt_ids = create_student_applicator_instruction(
                tokenizer,
                instruction=system_prompt,
                title='',
                content=data['input'],
                feedback_list=data['output'],
                template=template   
            )
        
        # the input is already processed
        else:
            prompt_ids = create_feedback_instruction(
                tokenizer,
                instruction=system_prompt,
                title='',
                content=data['input'],
                template=template
            )
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(device)

        # iterating over the temperature grid
        for temperature in tqdm(temperature_grid):

            # generate the revised essay using the model
            with torch.no_grad():

                # generate the feedback for num_explorations == 1
                if num_explorations == 1:
                    output_tensor = model.generate(
                        prompt_tensor,
                        max_new_tokens=900,
                        temperature=temperature,
                        pad_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=temperature != 0.0,
                    )
                    relevant_output = output_tensor[0, prompt_tensor.shape[1]:]
                    feedback = tokenizer.decode(relevant_output, skip_special_tokens=True)

                else:
                    output_tensor = model.generate(
                        prompt_tensor,
                        max_new_tokens=900,
                        temperature=temperature,
                        pad_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=temperature != 0.0,
                        num_return_sequences=num_explorations if temperature != 0.0 else 1
                    )
                    relevant_output = output_tensor[:, prompt_tensor.shape[1]:]
                    feedback = tokenizer.batch_decode(relevant_output, skip_special_tokens=True)

            # store the feedback in the student simulator results
            data['student_simulator_results'][temperature] = feedback

    # writing the output file
    with open(output_file, 'w') as f:
        json.dump(data_list, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate simulation revised essays using different essays')
    parser.add_argument('--input_file', type=str, help='Input file with all the feedback for analysis', default='/home/inair/data/econ_data/assignment_2_processed/supervised_data/student_applicator_test_combine.json')
    parser.add_argument('--model_path', type=str, help='Model path to generate feedback', default='/home/inair/data/llama3_models/llama-3-8b-instruct-hf')
    parser.add_argument('--adapter_path', type=str, help='Adapter path to generate feedback', default='/home/inair/data/revision_saves/sft_mwrite_student_applicator_combine_8b_instruct')
    parser.add_argument('--system_file', type=str, help='System file with all the feedback for analysis', default='/home/inair/data/econ_data/assignment_2_processed/student_applicator_system_prompt.txt')
    parser.add_argument('--output_file', type=str, help='Output file with all the feedback for analysis', default='/home/inair/data/revision_output/feedback_generation_test_student_simulator_combine_results_llama.json')
    parser.add_argument('--template', type=str, help='Template for feedback generation', default='llama3')
    parser.add_argument('--padding_side', type=str, help='Padding side for feedback generation', default='left')
    parser.add_argument('--device', type=str, help='Device for feedback generation', default='cuda:0')
    parser.add_argument('--processed_input', action='store_true', help='Whether the input file is processed or not')
    parser.add_argument('--num_explorations', type=int, help='Number of explorations to run', default=1)
    args = parser.parse_args()

    # call the wrapper function
    generate_simulator_results_llama_wrapper(
        input_file=args.input_file,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        system_file=args.system_file,
        output_file=args.output_file,
        template=args.template,
        padding_side=args.padding_side,
        device=args.device,
        processed_input=args.processed_input,
        num_explorations=args.num_explorations
    )
