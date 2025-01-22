import os
import json
import argparse
from tqdm import tqdm
from utils.prompt_utils import prompt_parser
from feedback_applicator.openai_feedback_applicator import revise_essay_v2, batch_revise_essay_v2

def generate_simulator_results_openai_wrapper(
    input_file: str,
    model_name: str,
    system_file: str,
    output_file: str,
    num_explorations: int = 1
) -> None:
    '''
        Augments the input file data with student simulator results
    '''

    # check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f'Input file {input_file} not found')
    with open(input_file, 'r') as f:
        data_list = json.load(f)

    # check if the system file exists
    if not os.path.exists(system_file):
        raise FileNotFoundError(f'System file {system_file} not found')
    with open(system_file, 'r') as f:
        system_prompt = f.read()
    system_prompt = prompt_parser(system_prompt)

    # setting up temperature grid from 0 to 1 inclusive with 0.05 step with numpy
    temperature_grid = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    # iterating over the data
    for data in tqdm(data_list):
        
        # create a key for storing the student student simulator results
        data['student_simulator_results'] = {}

        # processing the input essay and feedback
        essay, feedback = data['input'].split('Feedback: Reviewer 1')
        feedback = feedback.strip()
        essay = essay.split('Essay:')[1].strip()

        # iterating over the temperature grid
        for temperature in tqdm(temperature_grid):

            # generating revised essay from the openai
            if num_explorations == 1:
                revised_output = revise_essay_v2(
                    system_prompt,
                    '',
                    essay,
                    [feedback],
                    model_name,
                    temperature
                )
            elif num_explorations > 1:
                revised_output = batch_revise_essay_v2(
                    system_prompt,
                    [''] * num_explorations,
                    [essay] * num_explorations,
                    [[feedback]] * num_explorations,
                    model_name,
                    temperature
                )

            # store the feedback in the student simulator results
            data['student_simulator_results'][temperature] = revised_output

    # writing the output file
    with open(output_file, 'w') as f:
        json.dump(data_list, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate simulation revised essays using different essays')
    parser.add_argument('--input_file', type=str, help='Input file with all the feedback for analysis', default='/home/inair/data/econ_data/assignment_2_processed/supervised_data/student_applicator_test_combine.json')
    parser.add_argument('--model_name', type=str, help='Model name to use', default='gpt-35-turbo-student-applicator')
    parser.add_argument('--system_file', type=str, help='System file with all the feedback for analysis', default='/home/inair/data/econ_data/assignment_2_processed/student_applicator_system_prompt.txt')
    parser.add_argument('--output_file', type=str, help='Output file with all the feedback for analysis', default='/home/inair/data/revision_output/feedback_generation_test_student_simulator_combine_results_openai.json')
    parser.add_argument('--num_explorations', type=int, help='Number of explorations to run', default=1)
    args = parser.parse_args()

    # call the wrapper function
    generate_simulator_results_openai_wrapper(
        input_file=args.input_file,
        model_name=args.model_name,
        system_file=args.system_file,
        output_file=args.output_file,
        num_explorations=args.num_explorations
    )
