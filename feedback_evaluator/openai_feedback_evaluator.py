import os
import json
import argparse
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict
from utils.prompt_utils import prompt_parser
from essay_evaluator.openai_essay_evaluator import score_essay, batch_score_essay

def score_feedback(
    prompt: List[Dict[str, str]],
    essay: str,
    feedback: str,
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
) -> Dict[str, float]:
    '''
        Scores the feedback for its corresponding essay
    '''

    # creating the combined essay feedback string
    essay_feedback_string = 'Essay: {essay}\n\nFeedback: {feedback}'.format(
        essay=essay,
        feedback=feedback
    )

    # calling the score_essay function
    return score_essay(
        prompt,
        essay_feedback_string,
        chat_model=chat_model,
        temperature=temperature
    )

def batch_score_feedback(
    prompt: List[Dict[str, str]],
    essay: List[str],
    feedback: List[str],
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
) -> List[Dict[str, float]]:
    '''
        Scores the feedback for its corresponding essay
    '''

    # creating the combined essay feedback string
    essay_feedback_strings = [
        'Essay: {essay}\n\nFeedback: {feedback}'.format(
            essay=e,
            feedback=f
        ) for e, f in zip(essay, feedback)
    ]

    # calling the batch_score_essay function
    return batch_score_essay(
        prompt,
        essay_feedback_strings,
        chat_model=chat_model,
        temperature=temperature
    )

def openai_feedback_evaluation_wrapper(
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
    score_dict = defaultdict(lambda: {'score': 0.0})

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
            data['{}_score'.format(key)] = score

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
            data['{}_score'.format(key)] = score_list

            # get the average score dict
            score = defaultdict(lambda: {'score': 0.0})
            for score_item in score_list:
                for k, v in score_item.items():
                    if 'score' in v.keys():
                        score[k]['score'] += (v['score'] / len(score_list))

        # update the score dict
        for k, v in score.items():
            if 'score' in v.keys():
                score_dict[k]['score'] += v['score']

    # divide the score dict by the number of data items
    for k in score_dict:
        score_dict[k]['score'] /= len(data_list)

    # save the data
    with open(input_file, 'w') as f:
        json.dump(data_list, f)
    print(score_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Input file containing the data', default='/home/inair/data/revision_output/mwrite_specificity_dpo_8b_instruct_temp_0.7_combine/feedback_0.json')
    parser.add_argument('--system_file', type=str, help='System file containing the prompt', default='/home/inair/argument_revision/feedback_evaluator/pedagogy_prompt.txt')
    parser.add_argument('--chat_model', type=str, default='gpt-35-turbo', help='Chat model to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for the model')
    parser.add_argument('--key', type=str, default='feedback')
    parser.add_argument('--list_applicator', action='store_true', help='Whether to apply the feedback to the list of essays')
    args = parser.parse_args()

    openai_feedback_evaluation_wrapper(
        args.input_file,
        args.system_file,
        args.chat_model,
        args.temperature,
        args.key,
        args.list_applicator
    )
