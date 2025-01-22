import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from matplotlib import pyplot as plt
from utils.prompt_utils import prompt_parser
from essay_evaluator.openai_essay_evaluator import batch_score_essay

temperature_grid = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
score_normalization = 30.0

def compute_score_sum(
    score_obj: Dict[str, Dict[str, float]],
    score_normalization: float = 1.0
) -> float:
    '''
        Compute the sum of the scores
    '''

    # computing the sum of the scores
    score_sum = 0.0
    for key in score_obj:
        if 'score' in score_obj[key]:
            score_sum += score_obj[key]['score']
    return score_sum / score_normalization

def plot_performance_list(
    temperature_wise_performance: List[float],
    actual_performance: float,
    initial_performance: float,
    temperature_wise_label: str = 'Average Quality of Student Simulator Generated Revised Essays',
    actual_label: str = 'Average Quality of Student Written Revised Essays',
    initial_label: str = 'Average Quality of Student Written Essays before Revision',
    title: str = 'Revised Essay Quality for Student Simulator and Student',
    xlabel: str = 'Temperature',
    ylabel: str = 'Essay Quality',
    save_path: str = '/home/inair/argument_revision/student_simulator_analysis/modification_plot.png'
) -> None:
    '''
        Plots the modifications by the student simulator and the humans
    '''

    # Colors for the metrics  
    color_temperature_wise = '#1f77b4'  
    color_actual = '#ff7f0e'
    color_initial = '#2ca02c'

    # Plotting the data  
    plt.figure(figsize=(10, 6))  

    # Plot the student simulator results
    plt.plot(
        temperature_grid,
        temperature_wise_performance,
        color=color_temperature_wise,
        label=temperature_wise_label,
        linestyle='-',
        marker='o'
    )

    # Plot the human results
    plt.axhline(y=actual_performance, color=color_actual, label=actual_label, linestyle='--')
    plt.axhline(y=initial_performance, color=color_initial, label=initial_label, linestyle='--')

    # Set the title and labels
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Set the legend
    plt.legend(fontsize=12)

    # Save the plot
    plt.savefig(save_path)

def quality_analysis_parser(
    input_file: str,
    system_file: str,
    output_folder: str,
    score_model: str = 'gpt-35-turbo',
    key: str = 'revised_essay',
    list_applicator: bool = False
) -> None:
    '''
        Parse the input file and generate the quality analysis of the student simulator
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
        system_message = f.read()
    system_prompt = prompt_parser(system_message)

    # creating output directory
    os.makedirs(output_folder, exist_ok=True)

    # storing the average performance
    temperature_wise_performance = np.array([0.0 for _ in range(len(temperature_grid))])
    actual_performance = 0.0
    initial_performance = 0.0
    estimated_actual_performance_list = []
    assigned_actual_performance_list = []

    # iterating over the data list
    for data in tqdm(data_list):

        # computing the list of essays
        actual_essay = data[key]
        if key == 'revised_essay':
            initial_essay = data['input']
        else:
            initial_essay = data['input'].split('Feedback:')[0]
        student_simulated_essays = [data['student_simulator_results'][str(temp)] for temp in temperature_grid]

        # if list applicator is not used
        if not list_applicator:
            all_essays = [actual_essay] + student_simulated_essays + [initial_essay]

            # getting the scores for these essays
            scores = batch_score_essay(system_prompt, all_essays, score_model)
            score_scalar_list = [compute_score_sum(score, score_normalization) for score in scores]

            # updating the performance
            actual_performance += score_scalar_list[0]
            initial_performance += score_scalar_list[-1]
            temperature_wise_performance += np.array(score_scalar_list[1:-1])
            estimated_actual_performance_list.append(score_scalar_list[0])
            assigned_actual_performance_list.append(data['revised_overall_score'] / score_normalization)

            # logging the scores
            print(f'Actual essay score: {score_scalar_list[0]}')
            print(f'Initial essay score: {score_scalar_list[-1]}')
            print(f'Temperature wise scores: {score_scalar_list[1:-1]}')

        # if list applicator is used
        else:

            # computing static score
            static_essays = [actual_essay, initial_essay]
            static_scores = batch_score_essay(system_prompt, static_essays, score_model)
            static_score_scalar_list = [compute_score_sum(score, score_normalization) for score in static_scores]

            # computing the student simulated scores for each temperature
            score_scalar_list = []
            for temp in temperature_grid:
                student_simulated_essays = data['student_simulator_results'][str(temp)]
                student_simulated_scores = batch_score_essay(system_prompt, student_simulated_essays, score_model)
                student_simulated_score_scalar_list = [compute_score_sum(score, score_normalization) for score in student_simulated_scores]
                print('Student simulated scores:', student_simulated_score_scalar_list, 'temp:', temp)
                score_scalar_list.append(np.mean(student_simulated_score_scalar_list))

            # updating the performance
            actual_performance += static_score_scalar_list[0]
            initial_performance += static_score_scalar_list[-1]
            temperature_wise_performance += np.array(score_scalar_list)
            estimated_actual_performance_list.append(static_score_scalar_list[0])
            assigned_actual_performance_list.append(data['revised_overall_score'] / score_normalization)

            # logging the scores
            print(f'Actual essay score: {actual_performance}')
            print(f'Initial essay score: {initial_performance}')
            print(f'Temperature wise scores: {score_scalar_list}')

    # estimate the pearson correlation between the estimated and assigned scores
    estimated_actual_performance_list = np.array(estimated_actual_performance_list)
    assigned_actual_performance_list = np.array(assigned_actual_performance_list)
    correlation = np.corrcoef(estimated_actual_performance_list, assigned_actual_performance_list)[0, 1]
    print(f'Pearson correlation between estimated and assigned scores: {correlation}')

    # estimate the mean average error between the estimated and assigned scores
    mean_average_error = np.mean(np.abs(estimated_actual_performance_list - assigned_actual_performance_list))
    print(f'Mean Average Error between estimated and assigned scores: {mean_average_error}')

    # compute the average performance
    actual_performance /= len(data_list)
    initial_performance /= len(data_list)
    temperature_wise_performance /= len(data_list)
    print(f'Average performance: {actual_performance}')
    print(f'Initial performance: {initial_performance}')
    print(f'Temperature wise performance: {temperature_wise_performance}')

    # saving the plot
    plot_performance_list(
        temperature_wise_performance,
        actual_performance,
        initial_performance,
        save_path=os.path.join(output_folder, 'revised_essay_quality_plot.png')
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/home/inair/data/revision_output/feedback_generation_test_student_simulator_combine_results_openai.json', help='Path to the input file')
    parser.add_argument('--system_file', type=str, default='/home/inair/data/econ_data/assignment_2_processed/score_essay_system_prompt.txt', help='Path to the system file')
    parser.add_argument('--output_folder', type=str, default='/home/inair/argument_revision/student_simulator_analysis/openai', help='Path to the output folder')
    parser.add_argument('--score_model', type=str, default='gpt-35-turbo', help='Score model to be used for scoring the essays')
    parser.add_argument('--key', type=str, default='revised_essay', help='Where the original revised essay is stored')
    parser.add_argument('--list_applicator', action='store_true', help='whether each key in the student applicator points to a list')
    args = parser.parse_args()

    quality_analysis_parser(
        args.input_file,
        args.system_file,
        args.output_folder,
        args.score_model,
        args.key,
        args.list_applicator
    )