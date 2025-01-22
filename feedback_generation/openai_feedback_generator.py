import os
import json
import argparse
from tqdm import tqdm
from utils.openai_utils import openai_chat_api


def generate_feedback(
    system_message: str,
    input_essay: str,
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
) -> str:
    '''
        Generates feedback for the input essay
    '''

    # calling the openai chat api
    messages = [
        {
            'role': 'system',
            'content': system_message
        },
        {
            'role': 'user',
            'content': input_essay
        }
    ]
    response = openai_chat_api(messages, chat_model, temperature=temperature)
    try:
        return response.choices[0].message.content.strip()
    except:
        return "No change required."


def openai_feedback_generation(
    input_file: str,
    output_file: str,
    system_file: str,
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
    sample_feedback: bool = False,
    num_feedback: int = 5
):
    '''
        Applies the openai api to each datapoint in the input file
    '''

    # check if the sample feedback is true and temperature is not 0.0, otherwise throw error
    if sample_feedback and temperature == 0.0 and num_feedback > 1:
        raise ValueError('Temperature cannot be 0.0 and num_feedback cannot be equal to 1 when sample feedback is True')
    
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

    # create an output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # iterating over the data list
    for data in tqdm(data_list):
        input_essay = data['input']
        if sample_feedback:
            feedback_list = []
            for _ in range(num_feedback):
                feedback_list.append(generate_feedback(system_prompt, input_essay, chat_model, temperature))
                data['feedback_list'] = feedback_list
        else:
            feedback = generate_feedback(system_prompt, input_essay, chat_model, temperature)
            data['feedback'] = feedback

    # writing the data to the output file
    with open(output_file, 'w') as f:
        json.dump(data_list, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script for generating feedback from the openai chat api')
    parser.add_argument('--input_file', type=str, default='/home/inair/data/econ_data/assignment_2_processed/supervised_data/feedback_generation_test.json')
    parser.add_argument('--output_file', type=str, default='/home/inair/data/revision_output/mwrite_gpt-35-turbo/feedback_0.json')
    parser.add_argument('--system_file', type=str, default='/home/inair/data/econ_data/assignment_2_processed/few_shot_peer_review_system_prompt.txt')
    parser.add_argument('--chat_model', type=str, default='gpt-35-turbo')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--sample_feedback', action='store_true')
    parser.add_argument('--num_feedback', type=int, default=3)
    args = parser.parse_args()

    openai_feedback_generation(
        input_file=args.input_file,
        output_file=args.output_file,
        system_file=args.system_file,
        chat_model=args.chat_model,
        temperature=args.temperature,
        sample_feedback=args.sample_feedback,
        num_feedback=args.num_feedback
   )