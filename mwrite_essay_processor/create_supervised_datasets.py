import os
import json
import argparse
from utils.mwrite_utils import (
    split_dataset,
    convert_parsed_data_into_student_applicator_data,
    convert_parsed_data_into_feedback_reward_modeling,
    convert_parsed_data_into_feedback_generation_data,
    make_data_list_unique
)

def create_supervised_datasets(
    parsed_data: str,
    student_applicator_system_prompt: str,
    feedback_generation_system_prompt: str,
    save_dir: str,
    random_seed: int = 0
):
    '''
        Create supervised datasets for feedback generation and student applicator
    '''

    # check if the parsed_data is a file
    if not os.path.isfile(parsed_data):
        raise ValueError(f"Parsed data {parsed_data} is not a file")
    with open(parsed_data, 'r') as file:
        parsed_data = json.load(file)

    # check if the student_applicator_system_prompt is a file
    if not os.path.isfile(student_applicator_system_prompt):
        raise ValueError(f"Student applicator system prompt {student_applicator_system_prompt} is not a file")
    with open(student_applicator_system_prompt, 'r') as file:
        student_applicator_system_prompt = file.read()

    # check if the feedback_generation_system_prompt is a file
    if not os.path.isfile(feedback_generation_system_prompt):
        raise ValueError(f"Feedback generation system prompt {feedback_generation_system_prompt} is not a file")
    with open(feedback_generation_system_prompt, 'r') as file:
        feedback_generation_system_prompt = file.read()

    # create the save_directory
    os.makedirs(save_dir, exist_ok=True)

    # split the data into test and train
    train_parsed_data, test_parsed_data = split_dataset(parsed_data, random_seed=random_seed)

    # create the reward modeling dataset for the feedback generation system
    train_reward_modeling_data = convert_parsed_data_into_feedback_reward_modeling(train_parsed_data, feedback_generation_system_prompt)
    test_reward_modeling_data = convert_parsed_data_into_feedback_reward_modeling(test_parsed_data, feedback_generation_system_prompt)

    # save the reward modeling dataset
    with open(os.path.join(save_dir, 'reward_modeling_train.json'), 'w') as file:
        json.dump(train_reward_modeling_data, file, indent=4)
    with open(os.path.join(save_dir, 'reward_modeling_test.json'), 'w') as file:
        json.dump(test_reward_modeling_data, file, indent=4)

    # create the student applicator supervised dataset
    # train_student_applicator_data = convert_parsed_data_into_student_applicator_data(train_parsed_data, student_applicator_system_prompt)
    # test_student_applicator_data = convert_parsed_data_into_student_applicator_data(test_parsed_data, student_applicator_system_prompt)
    # train_combine_student_applicator_data = convert_parsed_data_into_student_applicator_data(
    #     train_parsed_data,
    #     student_applicator_system_prompt,
    #     combine_feedback=True,
    #     combine_feedback_model='gpt-4o'
    # )
    # test_combine_student_applicator_data = convert_parsed_data_into_student_applicator_data(
    #     test_parsed_data,
    #     student_applicator_system_prompt,
    #     combine_feedback=True,
    #     combine_feedback_model='gpt-4o'
    # )

    # # save the student applicator supervised dataset
    # with open(os.path.join(save_dir, 'student_applicator_train.json'), 'w') as file:
    #     json.dump(train_student_applicator_data, file, indent=4)
    # with open(os.path.join(save_dir, 'student_applicator_test.json'), 'w') as file:
    #     json.dump(test_student_applicator_data, file, indent=4)
    # with open(os.path.join(save_dir, 'student_applicator_train_combine.json'), 'w') as file:
    #     json.dump(train_combine_student_applicator_data, file, indent=4)
    # with open(os.path.join(save_dir, 'student_applicator_test_combine.json'), 'w') as file:
    #     json.dump(test_combine_student_applicator_data, file, indent=4)

    # create the feedback generation supervised dataset
    train_feedback_generation_data = convert_parsed_data_into_feedback_generation_data(train_parsed_data, feedback_generation_system_prompt, 'train')
    test_feedback_generation_data = convert_parsed_data_into_feedback_generation_data(test_parsed_data, feedback_generation_system_prompt, 'test')

    # create a version of the feedback generation supervised dataset where the value of input is unique
    train_feedback_generation_data_unique = make_data_list_unique(train_feedback_generation_data, key='input')

    # save the feedback generation supervised dataset
    with open(os.path.join(save_dir, 'feedback_generation_train.json'), 'w') as file:
        json.dump(train_feedback_generation_data, file, indent=4)
    with open(os.path.join(save_dir, 'feedback_generation_train_unique.json'), 'w') as file:
        json.dump(train_feedback_generation_data_unique, file, indent=4)
    with open(os.path.join(save_dir, 'feedback_generation_test.json'), 'w') as file:
        json.dump(test_feedback_generation_data, file, indent=4)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Create supervised datasets for feedback generation and student applicator')
    parser.add_argument('--parsed_data', type=str, help='Path to the parsed data', default='/home/inair/data/econ_data/assignment_2_processed/parsed.json')
    # parser.add_argument('--student_applicator_system_prompt', type=str, default='/home/inair/data/econ_data/assignment_2_processed/student_applicator_system_prompt.txt')
    parser.add_argument('--student_applicator_system_prompt', type=str, default='/home/inair/data/econ_data/assignment_2_processed/peer_review_system_prompt.txt')
    parser.add_argument('--feedback_generation_system_prompt', type=str, default='/home/inair/data/econ_data/assignment_2_processed/peer_review_system_prompt.txt')
    parser.add_argument('--save_dir', type=str, help='The directory to save the supervised datasets', default='/home/inair/data/econ_data/assignment_2_processed/supervised_data')
    parser.add_argument('--random_seed', type=int, default=0)
    args = parser.parse_args()

    create_supervised_datasets(
        parsed_data=args.parsed_data,
        student_applicator_system_prompt=args.student_applicator_system_prompt,
        feedback_generation_system_prompt=args.feedback_generation_system_prompt,
        save_dir=args.save_dir,
        random_seed=args.random_seed
    )