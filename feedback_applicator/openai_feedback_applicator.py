import os
import json
import asyncio
import argparse
from tqdm import tqdm
from typing import List, Dict
from utils.prompt_utils import prompt_parser
from utils.openai_utils import openai_chat_api, openai_chat_api_batch

def revise_essay(
    prompt: List[Dict[str, str]],
    title: str,
    essay: str,
    feedback: str,
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
    max_tokens: int = 1000
) -> str:
    '''
        Applies feedback to the input essay along with the feedback
    '''

    # creating the full message
    message = prompt + [
        {
            'role': 'user',
            'content': 'Essay: {}\n{}\n\nFeedback: {}'.format(title, essay, feedback)
        }
    ]

    # calling the openai chat api
    response = openai_chat_api(message, chat_model, temperature=temperature, max_tokens=max_tokens)
    try:
        return response.choices[0].message.content.strip()
    except:
        return essay
    
def revise_essay_v2(
    prompt: List[Dict[str, str]],
    title: str,
    essay: str,
    feedback_list: List[str],
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
    max_tokens: int = 1000
) -> str:
    '''
        Applies feedback to the input essay along with the feedback
    '''

    # creating the feedback string
    feedback_string = ''
    for index, feedback in enumerate(feedback_list):
        feedback_string += 'Reviewer {}\n{}\n\n'.format(index + 1, feedback)

    return revise_essay(prompt, title, essay, feedback_string, chat_model, temperature, max_tokens)

    
def batch_revise_essay(
    prompt: List[Dict[str, str]],
    title: List[str],
    essay: List[str],
    feedback: List[str],
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
    max_tokens: int = 1000
) -> List[str]:
    '''
        Applies each element of feedback to the corresponding element of title and essay
        in an efficient manner using asyncio
    '''

    # creating a batch of messages
    messages_batch = []
    for t, e, f in zip(title, essay, feedback):
        messages = prompt + [
            {
                'role': 'user',
                'content': 'Essay: {}\n{}\n\nFeedback: {}'.format(t, e, f)
            }
        ]
        messages_batch.append(messages)

    # calling async open ai
    responses = asyncio.run(openai_chat_api_batch(messages_batch, engine=chat_model, temperature=temperature, max_tokens=max_tokens))

    # processing th responses
    processed_responses = []
    for response in responses:
        try:
            processed_responses.append(response.choices[0].message.content.strip())
        except:
            processed_responses.append('')
    return processed_responses

def batch_revise_essay_v2(
    prompt: List[Dict[str, str]],
    title: List[str],
    essay: List[str],
    feedback_list: List[List[str]],
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
    max_tokens: int = 1000
) -> List[str]:
    '''
        Applies each element of feedback to the corresponding element of title and essay
        in an efficient manner using asyncio
    '''

    # creating a batch of messages
    messages_batch = []
    for t, e, f in zip(title, essay, feedback_list):
        feedback_string = ''
        for index, feedback in enumerate(f):
            feedback_string += 'Reviewer {}\n{}\n\n'.format(index + 1, feedback)
        messages = prompt + [
            {
                'role': 'user',
                'content': 'Essay: {}\n{}\n\nFeedback: {}'.format(t, e, feedback_string)
            }
        ]
        messages_batch.append(messages)

    # calling async open ai
    responses = asyncio.run(openai_chat_api_batch(messages_batch, engine=chat_model, temperature=temperature, max_tokens=max_tokens))

    # processing th responses
    processed_responses = []
    for response in responses:
        try:
            processed_responses.append(response.choices[0].message.content.strip())
        except:
            processed_responses.append('')
    return processed_responses

def openai_feedback_application(
    input_file: str,
    output_file: str,
    system_file: str,
    chat_model: str = 'gpt-35-turbo',
    temperature: float = 0.0,
    list_applicator: bool = False,
    student_type: str = 'normal',
    version_2: bool = False
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

    # choose appropriate version of the function
    revise_essay_final_function = revise_essay if not version_2 else revise_essay_v2

    # applying feedback to each of the datapoints
    for data in tqdm(data_list):
        if list_applicator and 'feedback_list' in data:
            revised_output_list = []
            for feedback in data['feedback_list']:
                new_essay = revise_essay_final_function(
                    prompt,
                    data['instruction'],
                    data['input'],
                    feedback if not version_2 else [feedback],
                    chat_model,
                    temperature
                )

                # processing the gpt output
                if student_type == 'normal':
                    revised_output_list.append(new_essay)
                elif student_type == 'cognitive':
                    if '[THINKING_SEP]' in new_essay:
                        revised_output_list.append(new_essay.split('[THINKING_SEP]')[1].strip())
                    else:
                        revised_output_list.append(new_essay)
            data['revised_output_list'] = revised_output_list

        else:

            new_essay = revise_essay_final_function(
                prompt,
                data['instruction'],
                data['input'],
                data['feedback'] if not version_2 else [data['feedback']],
                chat_model,
                temperature
            )

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
    parser.add_argument('--output_file', type=str, help='Output file to write the data', default='/home/inair/data/revision_output/essay_test_feedback_gpt_35_turbo_applicator_gpt_35_turbo.json')
    parser.add_argument('--system_file', type=str, help='System file containing the prompt', default='/home/inair/data/econ_data/assignment_2_processed/student_applicator_system_prompt.txt')
    parser.add_argument('--chat_model', type=str, default='gpt-35-turbo-student-applicator', help='Chat model to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for the model')
    parser.add_argument('--list_applicator', action='store_true', help='List the applicators available')
    parser.add_argument('--student_type', type=str, default='normal')
    parser.add_argument('--version_2', action='store_true', help='Version 2 of the applicator')
    args = parser.parse_args()

    openai_feedback_application(
        args.input_file,
        args.output_file,
        args.system_file,
        args.chat_model,
        args.temperature,
        args.list_applicator,
        args.student_type,
        args.version_2
    )
