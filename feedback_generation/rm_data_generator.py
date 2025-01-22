import os
import json
import argparse
from typing import List

def compare_value_vectors(
    val_1: List[str],
    val_2: List[str],
    strategy: str = 'simple'
) -> int:
    '''
        Returns 1 if val_1 is better than val_2, -1 if val_2 is better than val_1, 0 otherwise
    '''

    # if the mean of one is greater than the other and element wise greater than the other
    if strategy == 'simple':
        val_1_mean = sum(val_1) / len(val_1)
        val_2_mean = sum(val_2) / len(val_2)
        if val_1_mean > val_2_mean and all([v1 > v2 for v1, v2 in zip(val_1, val_2)]):
            return 1
        elif val_2_mean > val_1_mean and all([v2 > v1 for v1, v2 in zip(val_1, val_2)]):
            return -1
        else:
            return 0
        
    return 0



def rm_data_generation(
    input_file: str,
    system_prompt: str,
    strategy: str,
    output_file: str
):
    '''
        Function responsible for generating reward modelling data for training
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

    # generating the reward modelling data
    metrics = ['persuasiveness', 'eloquence', 'evidence', 'correctness', 'relevance', 'specificity']
    rm_data = []
    for data in data_list:
        value_vectors = [
            [obj[metric]['score'] for metric in metrics] for obj in data['revised_output_list_score']
        ]
        
        # add pairwise comparisons for each pair of value vectors
        for i in range(len(value_vectors)):
            for j in range(i+1, len(value_vectors)):
                comparison = compare_value_vectors(value_vectors[i], value_vectors[j], strategy)
                
                # selecting the better feedback
                if comparison == 0:
                    continue
                elif comparison == 1:
                    pairwise_feedback = [data['feedback_list'][i], data['feedback_list'][j]]
                elif comparison == -1:
                    pairwise_feedback = [data['feedback_list'][j], data['feedback_list'][i]]

                
                # appending this data to the rm_data
                rm_data.append({
                    'system': system_prompt,
                    'instruction': '',
                    'input': '{title}\n\n{content}'.format(
                        title=data['instruction'],
                        content=data['output'],
                    ),
                    'output': pairwise_feedback,
                })

    # saving the reward modelling data
    with open(output_file, 'w') as f:
        json.dump(rm_data, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Reward modelling generator for feedback generation')
    parser.add_argument('--input_file', type=str, help='Input file containing the data list', default='/home/inair/data/revision_output/essay_writing_350_feedback_list_gpt_35_turbo_applicator_gpt_35_turbo.json')
    parser.add_argument('--system_prompt', type=str, help='System prompt to be used for generating the sft data', default='feedback_generation/system_prompt.txt')
    parser.add_argument('--strategy', type=str, help='Strategy for candidate selection', default='simple')
    parser.add_argument('--output_file', type=str, help='Output file for the generated data', default='/home/inair/data/revision_supervised/rm_essay_writing_350_gpt_35_turbo_simple.json')
    args = parser.parse_args()
    
    rm_data_generation(
        args.input_file,
        args.system_prompt,
        args.strategy,
        args.output_file
    )
