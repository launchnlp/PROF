import os
from typing import List
from matplotlib import pyplot as plt
from student_simulator_analysis.lexical_diff_analysis import lexical_diff_analysis_wrapper

llama_file = '/home/inair/data/revision_output/feedback_generation_test_student_simulator_results_llama.json'
openai_file = '/home/inair/data/revision_output/feedback_generation_test_student_simulator_combine_results_openai.json'
output_folder = '/home/inair/argument_revision/student_simulator_analysis'
temperature_grid = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

# Enable LaTeX style  
plt.rc('text', usetex=True)  
plt.rc('font', family='serif') 
plt.rcParams.update({'font.size': 18})

def plot_modification_list(
    llama_addition_list: List[float],
    llama_deletion_list: List[float],
    openai_addition_list: List[float],
    openai_deletion_list: List[float],
    constant_addition: float,
    constant_deletion: float,
    llama_addition_list_label: str = 'Additions by \\texttt{llama3-8b}',
    llama_deletion_list_label: str = 'Deletions by \\texttt{llama3-8b}',
    openai_addition_list_label: str = 'Additions by \\texttt{gpt-3.5}',
    openai_deletion_list_label: str = 'Deletions by \\texttt{gpt-3.5}',
    constant_addition_label: str = 'Additions by Human',
    constant_deletion_label: str = 'Deletions by Human',
    title: str = 'Phrase Level Modification',
    xlabel: str = 'Temperature',
    ylabel: str = 'Number of modifications',
    save_path: str = '/home/inair/argument_revision/student_simulator_analysis/modification_plot.pdf'
) -> None:
    '''
        Plots the modifications by the student simulator and the humans
    '''

    # Colors for the metrics  
    llama_color_deletion = 'blue'  
    llama_color_addition = 'darkorange'  
    openai_color_deletion = 'navy'  
    openai_color_addition = 'orangered'  
    color_deletion_light = 'aqua'  
    color_addition_light = 'orange'

    # Plotting the data  
    plt.figure(figsize=(10, 6))  

    # Plot the student simulator results
    plt.plot(temperature_grid, llama_addition_list, color=llama_color_addition, label=llama_addition_list_label, linestyle='-', marker='o')
    plt.plot(temperature_grid, llama_deletion_list, color=llama_color_deletion, label=llama_deletion_list_label, linestyle='-', marker='o')

    # Plot the openai results with different markers
    plt.plot(temperature_grid, openai_addition_list, color=openai_color_addition, label=openai_addition_list_label, linestyle='-', marker='x')
    plt.plot(temperature_grid, openai_deletion_list, color=openai_color_deletion, label=openai_deletion_list_label, linestyle='-', marker='x')

    # Plot the human results
    plt.axhline(y=constant_addition, color=color_addition_light, label=constant_addition_label, linestyle='--')
    plt.axhline(y=constant_deletion, color=color_deletion_light, label=constant_deletion_label, linestyle='--')

    # Set the title and labels
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Set the legend
    plt.legend(fontsize=16)

    # Save the plot
    plt.savefig(save_path)

if __name__ == '__main__':

    # call the wrapper function for llama and openai
    llama_lexical_modifications, actual_modifications = lexical_diff_analysis_wrapper(
        llama_file, output_folder, False
    )
    openai_lexical_modifications, _ = lexical_diff_analysis_wrapper(
        openai_file, output_folder, True
    )

    # plotting the phrase level results
    plot_modification_list(
        [modification_dict['addition']['phrase_level'] for modification_dict in llama_lexical_modifications],
        [modification_dict['deletion']['phrase_level'] for modification_dict in llama_lexical_modifications],
        [modification_dict['addition']['phrase_level'] for modification_dict in openai_lexical_modifications],
        [modification_dict['deletion']['phrase_level'] for modification_dict in openai_lexical_modifications],
        actual_modifications['addition']['phrase_level'],
        actual_modifications['deletion']['phrase_level'],
        title='Phrase Level Modification',
        xlabel='Temperature',
        ylabel='Number of Modifications',
        save_path=os.path.join(output_folder, 'phrase_level_modification_plot.pdf')
    )

    # plotting the word level results
    plot_modification_list(
        [modification_dict['addition']['word_level'] for modification_dict in llama_lexical_modifications],
        [modification_dict['deletion']['word_level'] for modification_dict in llama_lexical_modifications],
        [modification_dict['addition']['word_level'] for modification_dict in openai_lexical_modifications],
        [modification_dict['deletion']['word_level'] for modification_dict in openai_lexical_modifications],
        actual_modifications['addition']['word_level'],
        actual_modifications['deletion']['word_level'],
        title='Word Level Modification',
        xlabel='Temperature',
        ylabel='Number of Modifications',
        save_path=os.path.join(output_folder, 'word_level_modification_plot.pdf')
    )

    # plotting the sentence level results
    plot_modification_list(
        [modification_dict['addition']['sentence_level'] for modification_dict in llama_lexical_modifications],
        [modification_dict['deletion']['sentence_level'] for modification_dict in llama_lexical_modifications],
        [modification_dict['addition']['sentence_level'] for modification_dict in openai_lexical_modifications],
        [modification_dict['deletion']['sentence_level'] for modification_dict in openai_lexical_modifications],
        actual_modifications['addition']['sentence_level'],
        actual_modifications['deletion']['sentence_level'],
        title='Sentence Level Modification',
        xlabel='Temperature',
        ylabel='Number of Modifications',
        save_path=os.path.join(output_folder, 'sentence_level_modification_plot.pdf')
    )
