import os
import json
import argparse

def sft_data_generation(
    input_file: str,
    system_prompt: str,
    output_file: str
):
    '''
        Function responsible for generating sft data for training
    '''

    # creating a target folder corresponding to output_file
    target_folder = os.path.dirname(output_file)
    os.makedirs(target_folder, exist_ok=True)

    # reading the input file
    if not os.path.exists(input_file):
        raise FileNotFoundError('Input file not found: {}', input_file)
    with open(input_file, 'r') as f:
        data_list = json.load(f)

    # reading the system file
    with open(system_prompt, 'r') as f:
        system_prompt = f.read()

    # generating the sft data
    sft_data = []
    for data in data_list:
        if 'feedback_list' in data:
            for feedback in data['feedback_list']:
                sft_data.append({
                    'instruction': '',
                    'input': '{title}\n\n{content}'.format(
                        title=data['instruction'],
                        content=data['output'],
                    ),
                    'system': system_prompt,
                    'output': feedback
                })
        else:
            sft_data.append({
                'instruction': '',
                'input': '{title}\n\n{content}'.format(
                    title=data['instruction'],
                    content=data['output'],
                ),
                'system': system_prompt,
                'output': data['feedback']
            }) 

    # saving the sft data
    with open(output_file, 'w') as f:
        json.dump(sft_data, f, indent=4)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('sft_data_generator for feedback generation')
    parser.add_argument('--input_file', type=str, help='Input file containing the data list', default='/home/inair/data/revision_output/essay_writing_350_feedback_list_gpt_35_turbo_applicator_gpt_35_turbo.json')
    parser.add_argument('--system_prompt', type=str, help='System prompt to be used for generating the sft data', default='feedback_generation/system_prompt.txt')
    parser.add_argument('--output_file', type=str, help='Output file to store the feedback', default='/home/inair/data/revision_supervised/sft_essay_writing_350_gpt_35_turbo.json')
    args = parser.parse_args()

    sft_data_generation(
        args.input_file,
        args.system_prompt,
        args.output_file
    )