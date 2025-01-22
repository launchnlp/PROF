import os
import asyncio
import json
import argparse
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict
from utils.prompt_utils import prompt_parser
from utils.openai_utils import openai_chat_api, openai_chat_api_batch


def score_essay(
    prompt: List[Dict[str, str]],
    essay: str,
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
) -> Dict[str, float]:
    '''
        Applies feedback to the input essay along with the feedback
    '''

    # creating the full message
    message = prompt + [
        {
            'role': 'user',
            'content': essay
        }
    ]

    # calling the openai chat api
    response = openai_chat_api(message, chat_model, temperature=temperature)
    try:
        return json.loads(response.choices[0].message.content.strip())
    except:
        return {}
    
def batch_score_essay(
    prompt: List[Dict[str, str]],
    essay: List[str],
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
) -> List[Dict[str, float]]:
    '''
        Applies feedback to the input essay along with the feedback
    '''

    # creating the full message
    messages_batch = []
    for e in essay:
        messages = prompt + [
            {
                'role': 'user',
                'content': e
            }
        ]
        messages_batch.append(messages)

    # calling the openai chat api
    responses = asyncio.run(openai_chat_api_batch(messages_batch, engine=chat_model, temperature=temperature))
    scores = []
    for response in responses:
        try:
            score_obj = json.loads(response.choices[0].message.content.strip())
            if type(score_obj) != dict:
                score_obj = {}
            scores.append(score_obj)
        except:
            scores.append({})
    return scores

def openai_essay_evaluation(
    input_file: str,
    system_file: str,
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
    key: str = 'output',
    list_applicator: bool = False,
):
    '''
        Applies feedback to each of the datapoints in the input file
    '''

    # reading the system file
    if not os.path.exists(system_file):
        raise Exception('System file does not exist: {}'.format(system_file))
    with open(system_file, 'r') as f:
        system_message = f.read()
    prompt = prompt_parser(system_message)

    # reading the input file
    if not os.path.exists(input_file):
        raise Exception('Input file does not exist: {}'.format(input_file))
    with open(input_file, 'r') as f:
        data_list = json.load(f)

    # initializing the score
    score_dict = defaultdict(float)

    # generating score for each data item
    for data in tqdm(data_list):
        if not list_applicator:
            input_essay = '{}\n\n{}'.format(data['instruction'], data[key])
            score = score_essay(
                prompt,
                input_essay,
                chat_model,
                temperature
            )
        else:
            score_list = []
            for _, essay in enumerate(data[key]):
                input_essay = '{}\n\n{}'.format(data['instruction'], essay)
                score_current = score_essay(
                    prompt,
                    input_essay,
                    chat_model,
                    temperature
                )
                score_list.append(score_current)
            score = score_list

        # retaining the score information in the json due to its usefulness
        data['{}_score'.format(key)] = score

        # updating the score_dict with the score object
        if not list_applicator:
            for attribute in score.keys():
                if 'score' in score[attribute]:
                    score_dict[attribute] += float(score[attribute]['score'])

    # computing the average statistics
    if not list_applicator:
        final_score = []
        for attribute in score_dict:
            score_dict[attribute] /= len(data_list)
            final_score.append(score_dict[attribute])
        print(score_dict)
        print('Average Score: {}'.format(sum(final_score) / len(final_score)))

    # writing the output to the file
    with open(input_file, 'w') as f:
        json.dump(data_list, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Input file containing the data', default='/home/inair/data/revision_output/essay_test_feedback_gpt_35_turbo_applicator_gpt_35_turbo.json')
    parser.add_argument('--system_file', type=str, help='System file containing the prompt', default='essay_evaluator/system_prompt.txt')
    parser.add_argument('--chat_model', type=str, default='gpt-35-turbo', help='Chat model to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for the model')
    parser.add_argument('--key', type=str, default='output')
    parser.add_argument('--list_applicator', action='store_true', help='Whether to apply the feedback to the list of essays')
    args = parser.parse_args()

    openai_essay_evaluation(
        args.input_file,
        args.system_file,
        args.chat_model,
        args.temperature,
        args.key,
        args.list_applicator,
    )
