import os
import json
import pprint
import difflib
import argparse
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, List, Dict, Union
from nltk.tokenize import word_tokenize, sent_tokenize
from matplotlib import pyplot as plt

# global constants
temperature_grid = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

def compute_diff(sentence1: str, sentence2: str) -> Tuple[str, str]:
    '''
        Using the difflib library to get the list of consecutive words that has been added or deleted
    '''

    # Split sentences into words  
    words1 = sentence1.split()  
    words2 = sentence2.split()  
      
    # Create a Differ object and compute the difference  
    differ = difflib.Differ()  
    diff = list(differ.compare(words1, words2))  
      
    # Initialize lists to store added and deleted words  
    added_words = []  
    deleted_words = []  
      
    # Temporary lists to store consecutive added or deleted words  
    temp_added = []  
    temp_deleted = []  
      
    # Iterate through the diff result  
    for word in diff:  
        if word.startswith('+ '):  # Word was added  
            if temp_deleted:  
                deleted_words.append(' '.join(temp_deleted))  
                temp_deleted = []  
            temp_added.append(word[2:])  
        elif word.startswith('- '):  # Word was deleted  
            if temp_added:  
                added_words.append(' '.join(temp_added))  
                temp_added = []  
            temp_deleted.append(word[2:])  
        else:  # No change, flush temp lists if needed  
            if temp_added:  
                added_words.append(' '.join(temp_added))  
                temp_added = []  
            if temp_deleted:  
                deleted_words.append(' '.join(temp_deleted))  
                temp_deleted = []  
      
    # Flush remaining words in temp lists  
    if temp_added:  
        added_words.append(' '.join(temp_added))  
    if temp_deleted:  
        deleted_words.append(' '.join(temp_deleted))  
      
    return added_words, deleted_words

def count_modifications(sentence_list: List[str]) -> Dict[str, int]:
    '''
        Count the number of word level, phrase level and sentence level items in sentence_list
    '''

    # initialize the count dictionary
    count_dict = defaultdict(int)

    for sentence in sentence_list:

        # tokenize the sentence and check if its word, phrase or sentence level
        words = word_tokenize(sentence)
        if len(words) == 1:
            count_dict['word_level'] += 1
        elif len(words) > 1 and len(words) < 5:
            count_dict['phrase_level'] += 1
        else:

            # sentence tokenize the sentence
            sentence_sub_list = sent_tokenize(sentence)
            for sentence_sub in sentence_sub_list:
                words_sub = word_tokenize(sentence_sub)
                if len(words_sub) == 1:
                    count_dict['word_level'] += 1
                elif len(words_sub) > 1 and len(words_sub) < 5:
                    count_dict['phrase_level'] += 1
                else:
                    count_dict['sentence_level'] += 1

    return count_dict

def plot_modification_list(
    addition_list: List[float],
    deletion_list: List[float],
    constant_addition: float,
    constant_deletion: float,
    addition_list_label: str = 'Additions by Student Simulator',
    deletion_list_label: str = 'Deletions by Student Simulator',
    constant_addition_label: str = 'Additions by Human',
    constant_deletion_label: str = 'Deletions by Human',
    title: str = 'Phrase Level Modification',
    xlabel: str = 'Temperature',
    ylabel: str = 'Modification Count',
    save_path: str = '/home/inair/argument_revision/student_simulator_analysis/modification_plot.png'
) -> None:
    '''
        Plots the modifications by the student simulator and the humans
    '''

    # Colors for the metrics  
    color_deletion = '#d62728'  # Darker shade of red  
    color_addition = '#2ca02c'  # Darker shade of green  
    color_deletion_light = '#ff9896'  # Lighter shade of red  
    color_addition_light = '#98df8a'  # Lighter shade of green 

    # Plotting the data  
    plt.figure(figsize=(10, 6))  

    # Plot the student simulator results
    plt.plot(temperature_grid, addition_list, color=color_addition, label=addition_list_label, linestyle='-', marker='o')
    plt.plot(temperature_grid, deletion_list, color=color_deletion, label=deletion_list_label, linestyle='-', marker='o')

    # Plot the human results
    plt.axhline(y=constant_addition, color=color_addition_light, label=constant_addition_label, linestyle='--')
    plt.axhline(y=constant_deletion, color=color_deletion_light, label=constant_deletion_label, linestyle='--')

    # Set the title and labels
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Set the legend
    plt.legend(fontsize=12)

    # Save the plot
    plt.savefig(save_path)


def lexical_diff_analysis_wrapper(
    input_file: str,
    output_folder: str,
    processed_input: bool
) -> Tuple[List[Dict[str, Dict[str, Union[int, float]]]], Dict[str, Dict[str, Union[int, float]]]]:
    '''
        Computes the number of word level, phrase level and sentence level modifications for each temperature
    '''

    # checking if the input_file is valid
    if not os.path.exists(input_file):
        raise FileNotFoundError(f'Input file {input_file} not found')
    with open(input_file, 'r') as f:
        data_list = json.load(f)

    # create the output folder
    os.makedirs(output_folder, exist_ok=True)

    # creating the final dictionary to store results
    modification_dict_list = [{
        'addition': defaultdict(int),
        'deletion': defaultdict(int)
    } for _ in range(len(temperature_grid))]
    actual_modification_dict = {
        'addition': defaultdict(int),
        'deletion': defaultdict(int)
    }

    # iterating over the elements of the data list
    for data in tqdm(data_list):

        # getting the input essay based on the processed input
        if processed_input:
            input_essay = data['input'].split('Feedback:')[0]
            if 'Essay:' in input_essay:
                input_essay = input_essay.split('Essay:')[1].strip()
            revised_essay_key = 'output'
        else:
            input_essay = data['input']
            revised_essay_key = 'revised_essay'

        # iterating over the temperature grid
        for temperature_index, temperature in enumerate(temperature_grid):

            # get the added and deleted words for the student simulator results
            added_words, deleted_words = compute_diff(input_essay, data['student_simulator_results'][str(temperature)])

            # count the number of word level, phrase level and sentence level modifications
            added_count_dict = count_modifications(added_words)
            deleted_count_dict = count_modifications(deleted_words)

            # update the modification dictionary
            for key in added_count_dict:
                modification_dict_list[temperature_index]['addition'][key] += added_count_dict[key]
            for key in deleted_count_dict:
                modification_dict_list[temperature_index]['deletion'][key] += deleted_count_dict[key]

        # get the added and deleted words for the actual results
        added_words, deleted_words = compute_diff(input_essay, data[revised_essay_key])

        # count the number of word level, phrase level and sentence level modifications
        added_count_dict = count_modifications(added_words)
        deleted_count_dict = count_modifications(deleted_words)

        # update the actual modification dictionary
        for key in added_count_dict:
            actual_modification_dict['addition'][key] += added_count_dict[key]
        for key in deleted_count_dict:
            actual_modification_dict['deletion'][key] += deleted_count_dict[key]

    # normalize all the values by the total number of samples
    total_samples = len(data_list)
    for modification_dict in modification_dict_list:
        for key in modification_dict['addition']:
            modification_dict['addition'][key] /= total_samples
        for key in modification_dict['deletion']:
            modification_dict['deletion'][key] /= total_samples
    for key in actual_modification_dict['addition']:
        actual_modification_dict['addition'][key] /= total_samples
    for key in actual_modification_dict['deletion']:
        actual_modification_dict['deletion'][key] /= total_samples


    pprint.pprint(modification_dict_list)
    pprint.pprint(actual_modification_dict)
    return modification_dict_list, actual_modification_dict

    # plot the phrase modifications
    plot_modification_list(
        [modification_dict['addition']['phrase_level'] for modification_dict in modification_dict_list],
        [modification_dict['deletion']['phrase_level'] for modification_dict in modification_dict_list],
        actual_modification_dict['addition']['phrase_level'],
        actual_modification_dict['deletion']['phrase_level'],
        title='Phrase Level Modification',
        xlabel='Temperature',
        ylabel='Modification Count',
        save_path=os.path.join(output_folder, 'phrase_level_modification_plot.png')
    )

    # plot the word modifications
    plot_modification_list(
        [modification_dict['addition']['word_level'] for modification_dict in modification_dict_list],
        [modification_dict['deletion']['word_level'] for modification_dict in modification_dict_list],
        actual_modification_dict['addition']['word_level'],
        actual_modification_dict['deletion']['word_level'],
        title='Word Level Modification',
        xlabel='Temperature',
        ylabel='Modification Count',
        save_path=os.path.join(output_folder, 'word_level_modification_plot.png')
    )

    # plot the sentence modifications
    plot_modification_list(
        [modification_dict['addition']['sentence_level'] for modification_dict in modification_dict_list],
        [modification_dict['deletion']['sentence_level'] for modification_dict in modification_dict_list],
        actual_modification_dict['addition']['sentence_level'],
        actual_modification_dict['deletion']['sentence_level'],
        title='Sentence Level Modification',
        xlabel='Temperature',
        ylabel='Modification Count',
        save_path=os.path.join(output_folder, 'sentence_level_modification_plot.png')
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/home/inair/data/revision_output/feedback_generation_test_student_simulator_results_llama.json', help='Path to the input file')
    parser.add_argument('--output_folder', type=str, default='/home/inair/argument_revision/student_simulator_analysis/', help='Path to the output folder')
    parser.add_argument('--processed_input', action='store_true', help='Whether the input file is processed or not')
    args = parser.parse_args()

    lexical_diff_analysis_wrapper(args.input_file, args.output_folder, args.processed_input)