import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Union
from nltk.tokenize import word_tokenize
from utils.prompt_utils import prompt_parser
from feedback_evaluator.openai_feedback_evaluator import score_feedback, batch_score_feedback

def update_score_dict_using_segment(
    score: Union[Dict, List[Dict]],
    score_dict: Dict[str, Dict[str, int]],
    list_applicator: bool = False
):
    '''
        Updating the score dict using the scores inferred from openai
    '''

    # iterating over the elements of 
    score_list = [score] if not list_applicator else score
    for score_item_list in score_list:
        for score_item in score_item_list:

            # check the type of the score_item
            if 'category' in score_item.keys() and score_item['category'] in ['praise', 'problem', 'solution', 'summary']:
                score_dict[score_item['category']]['count'] += 1
                score_dict[score_item['category']]['word_count'] += len(word_tokenize(score_item['segment']))
                
                # updating the localized attribute
                if 'localization' in score_item.keys() and score_item['localization'] == 'localized':
                    score_dict[score_item['category']]['localization'] += 1

                # updating the consistency attribute
                if 'consistency' in score_item.keys() and score_item['consistency'] == 'consistent':
                    score_dict[score_item['category']]['consistency'] += 1

                # updating the scope_local attribute
                if 'scope' in score_item.keys() and score_item['scope'] == 'local':
                    score_dict[score_item['category']]['scope_local'] += 1

                # check if the score_item is an affective compliment
                if 'affective' in score_item.keys() and score_item['affective'] == 'compliment':
                    score_dict[score_item['category']]['affective_compliment'] += 1

                # check if the score_item is an affective downplay
                if 'affective' in score_item.keys() and score_item['affective'] == 'downplay':
                    score_dict[score_item['category']]['affective_downplay'] += 1

                # check if the score_item is an affective neutral
                if 'affective' in score_item.keys() and score_item['affective'] == 'neutral':
                    score_dict[score_item['category']]['affective_neutral'] += 1

    return score_dict

def openai_feedback_segmentation_wrapper(
    input_file: str,
    system_file: str,
    chat_model: str = 'gpt-4-turbo',
    temperature: float = 0.0,
    key: str = 'feedback',
    list_applicator: bool = False
):
    '''
        Generates the score for feedback files using the openai chat api
    '''

    # check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError('Input file {} not found'.format(input_file))
    with open(input_file, 'r') as f:
        data_list = json.load(f)

    # check if the system file exists
    if not os.path.exists(system_file):
        raise FileNotFoundError('System file {} not found'.format(system_file))
    with open(system_file, 'r') as f:
        system_message = f.read()
    system_prompt = prompt_parser(system_message)

    # initializing the score dict
    score_dict = defaultdict(lambda: {
        'count': 0,
        'localization': 0,
        'scope_local': 0,
        'affective_compliment': 0,
        'affective_downplay': 0,
        'affective_neutral': 0,
        'word_count': 0,
        'consistency': 0
    })
    num_elements = 0

    # iterating over the data
    for data in tqdm(data_list):
        
        # if the list applicator is unset
        if not list_applicator:
            essay = data['input']
            feedback = data[key]
            score = score_feedback(
                system_prompt,
                essay,
                feedback,
                chat_model=chat_model,
                temperature=temperature
            )
            data['{}_segment'.format(key)] = score

            # updating the score_dict
            update_score_dict_using_segment(score, score_dict, list_applicator=False)
            num_elements += 1

        # if the list applicator is set
        else:

            # check if the key exists and is a list
            if key not in data or type(data[key]) != list:
                raise ValueError('Key {} not found or not a list in data'.format(key))
            
            # get the essay and feedback lists
            essay = data['input']
            feedback_list = data[key]
            essay_list = [essay] * len(feedback_list)

            # get the scores
            score_list = batch_score_feedback(
                system_prompt,
                essay_list,
                feedback_list,
                chat_model=chat_model,
                temperature=temperature
            )
            data['{}_segment'.format(key)] = score_list

            # updating the score_dict
            update_score_dict_using_segment(score_list, score_dict, list_applicator=True)
            num_elements += len(feedback_list)

    # divide the score dict by the number of data items
    for key in score_dict.keys():
        count_category = score_dict[key]['count']
        for sub_key in score_dict[key].keys():
            dividor = count_category if sub_key == 'word_count' else num_elements
            score_dict[key][sub_key] /= dividor
    print(score_dict)

    # save the data
    with open(input_file, 'w') as f:
        json.dump(data_list, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Input file containing the data', default='/home/inair/data/revision_output/mwrite_gpt-35-turbo/feedback_0.json')
    parser.add_argument('--system_file', type=str, help='System file containing the prompt', default='/home/inair/argument_revision/feedback_intrinsic_analysis/feedback_element_system_prompt.txt')
    parser.add_argument('--chat_model', type=str, default='gpt-4-turbo', help='Chat model to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for the model')
    parser.add_argument('--key', type=str, default='feedback')
    parser.add_argument('--list_applicator', action='store_true', help='Whether to apply the feedback to the list of essays')
    args = parser.parse_args()

    openai_feedback_segmentation_wrapper(
        args.input_file,
        args.system_file,
        args.chat_model,
        args.temperature,
        args.key,
        args.list_applicator
    )
