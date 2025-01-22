import os
import json
import torch
import argparse
from tqdm import tqdm
from utils.model_utils import load_model_and_tokenizer_simplified, create_student_applicator_instruction

def llama_feedback_application(
    input_file: str,
    output_file: str,
    system_file: str,
    model_path: str,
    adapter_path: str,
    student_type: str,
    feedback_key: str,
    device: str,
    template: str,
    padding_side: str,
    temperature: float
):
    '''
        Applies feedback to each of the datapoints in the input file
    '''

    # reading the system file
    if not os.path.exists(system_file):
        raise Exception('System file does not exist: {}'.format(system_file))
    with open(system_file, 'r') as f:
        system_prompt = f.read()

    # reading the input file
    if not os.path.exists(input_file):
        raise Exception('Input file does not exist: {}'.format(input_file))
    with open(input_file, 'r') as f:
        data_list = json.load(f)

    # loading the model
    model, tokenizer = load_model_and_tokenizer_simplified(
        model_path,
        adapter_path,
        padding_side=padding_side,
        template_name=template
    )
    model = model.to(device)

    # applying feedback to each of the datapoints
    for data in tqdm(data_list):

        # tokenize the instruction
        prompt_ids = create_student_applicator_instruction(
            tokenizer,
            instruction=system_prompt,
            title='',
            content=data['input'],
            feedback_list=[data[feedback_key]] if type(data[feedback_key]) == str else data[feedback_key],
            template=template
        )
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(device)

        # generate the revised_essay using the model
        with torch.no_grad():
            output_tensor = model.generate(
                prompt_tensor,
                max_new_tokens=1000,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_k=0.0,
                top_p=0.95,
            )
            relevant_output = output_tensor[:, prompt_tensor.shape[1]:]
            new_essay = tokenizer.batch_decode(relevant_output, skip_special_tokens=True)[0]

        # processing the gpt output
        if student_type == 'normal':
            data['revised_output'] = new_essay
        elif student_type == 'cognitive':
            if '[THINKING_SEP]' in new_essay:
                data['revised_output'] = new_essay.split('[THINKING_SEP]')[1].strip()
                data['student_thinking'] = new_essay.split('[THINKING_SEP]')[0].strip()
            else:
                data['revised_output'] = new_essay
                data['student_thinking'] = ''

    # writing the output file
    with open(output_file, 'w') as f:
        json.dump(data_list, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Input file containing the data', default='/home/inair/data/revision_output/mwrite_specificity_dpo_8b_instruct_llmjudge/feedback_1.json')
    parser.add_argument('--output_file', type=str, help='Output file to write the data', default='/home/inair/data/revision_output/student_applicator_llama_8b.json')
    parser.add_argument('--system_file', type=str, help='System file containing the prompt', default='/home/inair/data/econ_data/assignment_2_processed/student_applicator_system_prompt.txt')
    parser.add_argument('--model_path', type=str, default='/home/inair/data/llama3_models/llama-3-8b-instruct-hf')
    parser.add_argument('--adapter_path', type=str, default='/home/inair/data/revision_saves/sft_mwrite_student_applicator_combine_8b_instruct')
    parser.add_argument('--student_type', type=str, default='normal', help='type of prompting used for student')
    parser.add_argument('--feedback_key', type=str, default='output', help='Key to use which contains feedback')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for the model')
    parser.add_argument('--template', type=str, default='llama3', help='Template to use for the model')
    parser.add_argument('--padding_side', type=str, default='left', help='Padding side to use for the model')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature to use for the model')
    args = parser.parse_args()

    llama_feedback_application(
        args.input_file,
        args.output_file,
        args.system_file,
        args.model_path,
        args.adapter_path,
        args.student_type,
        args.feedback_key,
        args.device,
        args.template,
        args.padding_side,
        args.temperature
    )
