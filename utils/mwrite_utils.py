import os
import json
import random
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Union, Tuple, Literal, Any
from utils.prompt_utils import prompt_parser
from utils.openai_utils import openai_chat_api

def format_rubric_data(
    rubric_data: str,
    rubric_df: pd.DataFrame,
    value: str = 'comment'
) -> List[Dict[str, str]]:
    '''
        Format the rubric data into a dictionary
    '''

    # parsing the rubric data and df
    rubric_data_list = json.loads(rubric_data)
    rubric_schema_list = json.loads(rubric_df['data'][0])

    # creating the final dictionary
    final_list = []
    for data in rubric_data_list:

        # search for the element in rubric schema list with id == data['criterion_id']
        for rubric_schema in rubric_schema_list:
            if rubric_schema['id'] == data['criterion_id']:
                if rubric_schema['long_description'] is None:
                    key_string = rubric_schema['description']
                else:
                    key_string = rubric_schema['description'] + ': ' + rubric_schema['long_description']
                final_list.append({
                    'description': key_string,
                    'comment': data[value]
                })
                break

    return final_list

def format_feedback(
    guided_data: List[Dict[str, str]]
) -> str:
    '''
        Format the feedback into a string
    '''
    return '\n'.join([f"{data['description'].strip()}: {data['comment'].strip()}" for data in guided_data])


def split_dataset(
    data: Dict[str, Dict],
    train_ratio: float = 0.8,
    random_seed: int = 0
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    '''
        Split the dataset into train and test
    '''

    # set the random seed
    random.seed(random_seed)

    # get the keys
    keys = list(data.keys())

    # shuffle the keys
    random.shuffle(keys)

    # split the keys
    train_keys = keys[:int(train_ratio * len(keys))]
    test_keys = keys[int(train_ratio * len(keys)):]

    # create the train and test data
    train_data = {key: data[key] for key in train_keys}
    test_data = {key: data[key] for key in test_keys}
    return train_data, test_data


def find_file_endswith(root: str, ext: str) -> Union[str, None]:
    '''
        Find the file with the given extension in the root directory
    '''

    for _, _, files in os.walk(root):
        for file in files:
            if file.endswith(ext):
                return os.path.join(root, file)
    return None

def convert_parsed_data_into_student_applicator_data(
    parsed_data: Dict[str, Dict],
    student_applicator_system_prompt: str,
    combine_feedback: bool = False,
    combine_feedback_model: Literal['gpt-35-turbo', 'gpt-4o'] = 'gpt-4o',
    combine_system_prompt: str = '/home/inair/argument_revision/gpt_finetuning/single_feedback_converter_prompt.txt'
) -> List[Dict]:
    '''
        Convert the parsed data into student applicator paired data
    '''

    # initializing the message list if combine feedback is to used
    if combine_feedback:
        if not os.path.exists(combine_system_prompt):
            raise Exception('System file does not exist: {}'.format(combine_system_prompt))
        with open(combine_system_prompt, 'r') as f:
            system_prompt = f.read()
        combine_message_list = prompt_parser(system_prompt)

    final_data = []
    for _, data in tqdm(parsed_data.items()):

        # check if the revised_essay is present
        if 'revised_essay' not in data:
            continue

        # create the final feedback
        final_feedback = ''
        for index, review_id in enumerate(data['feedback'].keys()):
            feedback_object = data['feedback'][review_id]
            final_feedback += 'Reviewer {}\n{}\n\n'.format(index + 1, format_feedback(feedback_object)) 
        
        # combine the final feedback by calling open ai
        if combine_feedback:
            single_feedback_response = openai_chat_api(
                messages = combine_message_list + [{'role': 'user', 'content': final_feedback}],
                engine = combine_feedback_model
            )
            
            # if the response is not none modifying the final feedback
            if single_feedback_response is not None:
                single_feedback = single_feedback_response.choices[0].message.content
                final_feedback = 'Reviewer 1\n' + single_feedback.strip()

        final_data.append({
            'instruction': '',
            'input': 'Essay: {}\n\nFeedback: {}'.format(data['essay'], final_feedback),
            'output': data['revised_essay'],
            'system': student_applicator_system_prompt,
            'revised_overall_score': data['revised_overall_score'] if 'revised_overall_score' in data else None,
            'revised_score': data['revised_score'] if 'revised_score' in data else None
        })

    return final_data

def convert_parsed_data_into_feedback_reward_modeling(
    parsed_data: Dict[str, Dict],
    feedback_generation_system_prompt: str
) -> List[Dict[str, Any]]:
    '''
        Convert the parsed data into feedback reward modeling paired data
    '''
    final_data = []

    # iterating over the data points in the parsed data
    for _, data in tqdm(parsed_data.items()):

        # iterating over the datapoints in the feedback_score
        reviewer_id_list = list(data['feedback_score'].keys())
        for index_i in range(len(reviewer_id_list)):
            for index_j in range(index_i + 1, len(reviewer_id_list)):
                
                # ignoring if the score is equal
                if data['feedback_score'][reviewer_id_list[index_i]] == data['feedback_score'][reviewer_id_list[index_j]]:
                    continue

                # identifying index with smaller and larger score
                if data['feedback_score'][reviewer_id_list[index_i]] < data['feedback_score'][reviewer_id_list[index_j]]:
                    smaller_index = index_i
                    larger_index = index_j
                else:
                    smaller_index = index_j
                    larger_index = index_i

                # creating the final data
                final_data.append({
                    'instruction': '',
                    'input': data['essay'],
                    'system': feedback_generation_system_prompt,
                    'chosen': format_feedback(data['feedback'][reviewer_id_list[larger_index]]),
                    'rejected': format_feedback(data['feedback'][reviewer_id_list[smaller_index]]),
                    'revised_essay': data['revised_essay'] if 'revised_essay' in data else None,
                    'revised_overall_score': data['revised_overall_score'] if 'revised_overall_score' in data else None,
                    'revised_score': data['revised_score'] if 'revised_score' in data else None,
                    'chosen_score': data['feedback_score'][reviewer_id_list[larger_index]],
                    'rejected_score': data['feedback_score'][reviewer_id_list[smaller_index]]
                })

    return final_data
            

def convert_parsed_data_into_feedback_generation_data(
    parsed_data: Dict[str, Dict],
    feedback_generation_system_prompt: str,
    mode: Literal['train', 'test'] = 'train'
) -> List[Dict]:
    '''
        Convert the parsed data into feedback generation paired data
    '''
    final_data = []
    for _, data in tqdm(parsed_data.items()):

        # if the mode is train
        if mode == 'train':
            for review_id in data['feedback'].keys():
                feedback_object = data['feedback'][review_id]
                final_data.append({
                    'instruction': '',
                    'input': data['essay'],
                    'system': feedback_generation_system_prompt,
                    'output': format_feedback(feedback_object),
                    'revised_essay': data['revised_essay'] if 'revised_essay' in data else None,
                    'revised_overall_score': data['revised_overall_score'] if 'revised_overall_score' in data else None,
                    'revised_score': data['revised_score'] if 'revised_score' in data else None
                })

        else:
            final_data.append({
                'instruction': '',
                'input': data['essay'],
                'system': feedback_generation_system_prompt,
                'revised_essay': data['revised_essay'] if 'revised_essay' in data else None,
                'revised_overall_score': data['revised_overall_score'] if 'revised_overall_score' in data else None,
                'revised_score': data['revised_score'] if 'revised_score' in data else None,
                'output': [format_feedback(data['feedback'][review_id]) for review_id in data['feedback'].keys()]
            })

    return final_data

def make_data_list_unique(
    data_list: List[Dict[str, Any]],
    key: str
):
    '''
        Removes those elements from the data list whose value of key is same while keeping the first occurence
    '''

    # create a set of keys
    key_set = set()

    # create the final list
    final_list = []

    # iterate over the data list
    for data in data_list:
        if data[key] not in key_set:
            key_set.add(data[key])
            final_list.append(data)

    return final_list