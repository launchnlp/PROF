import os
import json
import torch
import argparse
from tqdm import tqdm
from utils.model_utils import load_model_and_tokenizer_simplified, create_feedback_instruction

def llama_feedback_generation(
    input_file: str,
    output_file: str,
    system_file: str,
    model_path: str = '/home/inair/data/llama-2-7b-hf',
    adapter_path: str = '/home/inair/data/essay_finetuning/llama-2-7b-essay_writing_350',
    temperature: float = 0.0,
    device: str = 'cuda',
    template: str = 'llama3',
    padding_side: str = 'left',
    max_num_samples: int = 1000
):
    '''
        Applies the openai api to each datapoint in the input file
    '''

    # reading the system prompt
    if not os.path.exists(system_file):
        raise FileNotFoundError('System prompt file not found: {}', system_file)
    with open(system_file, 'r') as f:
        system_prompt = f.read()

    # reading the input file
    if not os.path.exists(input_file):
        raise FileNotFoundError('Input file not found: {}', input_file)
    with open(input_file, 'r') as f:
        data_list = json.load(f)

    # load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer_simplified(
        model_path,
        adapter_path,
        padding_side=padding_side,
        template_name=template
    )
    model.to(device)

    # iterating over the data_list
    for index, data in tqdm(enumerate(data_list)):

        # adding a break condition if index exceeds max_num_samples
        if index >= max_num_samples:
            break
        
        # tokenize the prompt
        prompt_ids = create_feedback_instruction(
            tokenizer,
            system_prompt,
            title='',
            content=data['input'],
            template=template
        )
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(device)
        
        # generate the feedback
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
        data['feedback'] = feedback

    # writing the output file
    with open(output_file, 'w') as f:
        json.dump(data_list[:max_num_samples], f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Uses the trained model to generate feedback')
    parser.add_argument('--input_file', type=str, default='/home/inair/data/essay_test.json')
    parser.add_argument('--output_file', type=str, default='/home/inair/data/revision_output/essay_test_feedback_sft_7b.json')
    parser.add_argument('--system_file', type=str, default='feedback_generation/system_prompt.txt')
    parser.add_argument('--model_path', type=str, default='/home/inair/data/llama-2-7b-hf')
    parser.add_argument('--adapter_path', type=str, default='/home/inair/data/revision_saves/sft_essay_writing_350_gpt_35_turbo')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--template', type=str, default='llama3')
    parser.add_argument('--padding_side', type=str, default='left')
    parser.add_argument('--max_num_samples', type=int, default=1000)
    args = parser.parse_args()

    llama_feedback_generation(
        input_file=args.input_file,
        output_file=args.output_file,
        system_file=args.system_file,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        device=args.device,
        template=args.template,
        padding_side=args.padding_side,
        max_num_samples=args.max_num_samples
    )