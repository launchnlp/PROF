'''
    Converts a function into evaluation form
    The dataset used for generating the feedback is not compatible for evaluations
    This program converts the dataset into evaluation form that is compatible with other scripts
'''
import os
import json
import argparse

def convert_into_eval_form(
    input_file: str,
    output_file: str
):
    '''
        Converts the input file into evaluation form
    '''
    # reading the input file
    if not os.path.exists(input_file):
        raise FileNotFoundError('Input file not found: {}', input_file)
    with open(input_file, 'r') as f:
        data_list = json.load(f)
    
    # iterating over the data list
    new_data_list = []
    for data in data_list:
        new_data = {
            'instruction': data['input'].split('\n\n')[0],
            'input': '',
            'output': '\n\n'.join(data['input'].split('\n\n')[1:]),
            'feedback': data['output']
        }
        new_data_list.append(new_data)
    
    # writing the data to the output file
    with open(output_file, 'w') as f:
        json.dump(new_data_list, f, indent=4
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('For converting into evaluation form')
    parser.add_argument('--input_file', type=str, help='Input file containing the data', default='//home/inair/data/revision_saves/sft_essay_writing_350_gpt_35_turbo_correct_specificity/feedback_0.json')
    parser.add_argument('--output_file', type=str, help='Output file containing the data', default='/home/inair/data/revision_saves/sft_essay_writing_350_gpt_35_turbo_correct_specificity/feedback_0_eval.json')
    args = parser.parse_args()

    convert_into_eval_form(
        input_file=args.input_file,
        output_file=args.output_file
    )