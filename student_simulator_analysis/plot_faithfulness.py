from typing import List
from matplotlib import pyplot as plt

temperature_grid = [0.7, 0.85, 1.0]
llama_faithfulness = [1.0, 2.3, 3.8]
llama_unfaithfulness = [0.8, 0.7, 3.5]
openai_faithfulness = [2.0, 2.7, 3.0]
openai_unfaithfulness = [0.8, 1.2, 2.5]
actual_faithfulness = 5.3
actual_unfaithfulness = 0.3

# Enable LaTeX style  
plt.rc('text', usetex=True)  
plt.rc('font', family='serif') 
plt.rcParams.update({'font.size': 18})

def plot_performance_list(
    llama_temperature_wise_performance: List[float],
    openai_temperature_wise_performance: List[float],
    actual_performance: float,
    llama_temperature_wise_label: str = 'Faithful Revisions by \\texttt{llama3-8b-instruct}',
    openai_temperature_wise_label: str = 'Faithful Revisions by \\texttt{gpt-35-turbo}',
    actual_label: str = 'Faithful Revisions by Human',
    title: str = 'Average Number of Faithful Revisions',
    xlabel: str = 'Temperature',
    ylabel: str = 'Faithful Revisions',
    save_path: str = '/home/inair/argument_revision/student_simulator_analysis/faithfulness_plot.pdf'
) -> None:
    '''
        Plots the modifications by the student simulator and the humans
    '''

    # Colors for the metrics  
    llama_color_temperature_wise = '#1f77b4'
    openai_color_temperature_wise = '#ff7f0e'
    color_actual = '#2ca02c'

    # Plotting the data  
    plt.figure(figsize=(10, 6))  

    # Plot the student simulator results
    plt.plot(
        temperature_grid,
        llama_temperature_wise_performance,
        color=llama_color_temperature_wise,
        label=llama_temperature_wise_label,
        linestyle='-',
        marker='o'
    )
    plt.plot(
        temperature_grid,
        openai_temperature_wise_performance,
        color=openai_color_temperature_wise,
        label=openai_temperature_wise_label,
        linestyle='-',
        marker='x'
    )

    # Plot the human results
    plt.axhline(y=actual_performance, color=color_actual, label=actual_label, linestyle='--')

    # Set the title and labels
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.6)

    # only label temperature grid
    plt.xticks(temperature_grid)

    # Set the legend
    plt.legend(fontsize=18)

    # Save the plot
    plt.savefig(save_path)

if __name__ == '__main__':
    plot_performance_list(
        llama_faithfulness,
        openai_faithfulness,
        actual_faithfulness,
    )

    plot_performance_list(
        llama_unfaithfulness,
        openai_unfaithfulness,
        actual_unfaithfulness,
        llama_temperature_wise_label = 'Unfaithful Revisions by \\texttt{llama3-8b-instruct}',
        openai_temperature_wise_label = 'Unfaithful Revisions by \\texttt{gpt-35-turbo}',
        actual_label = 'Unfaithful Revisions by Human',
        title = 'Average Number of Unfaithful Revisions',
        xlabel = 'Temperature',
        ylabel = 'Unfaithful Revisions',
        save_path = '/home/inair/argument_revision/student_simulator_analysis/unfaithfulness_plot.pdf'
    )