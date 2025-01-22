from typing import List
from matplotlib import pyplot as plt

temperature_grid = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
llama_performance = [0.77893519, 0.77229938, 0.78450617, 0.76905864, 0.75617284, 0.74544753, 0.7349537]
openai_performance = [0.81097222, 0.81115741, 0.81412037, 0.8112963,  0.81925926, 0.81097222, 0.82023148]
initial_performance = 0.7708333333333333
revised_performance = 0.8277777777777775

# Enable LaTeX style  
plt.rc('text', usetex=True)  
plt.rc('font', family='serif') 
plt.rcParams.update({'font.size': 18})

def plot_performance_list(
    llama_temperature_wise_performance: List[float],
    openai_temperature_wise_performance: List[float],
    actual_performance: float,
    initial_performance: float,
    llama_temperature_wise_label: str = '\\texttt{llama3-8b} Revised Essay Quality',
    openai_temperature_wise_label: str = '\\texttt{gpt-3.5} Revised Essay Quality',
    actual_label: str = 'Actual Revised Essay Quality',
    initial_label: str = 'Initial Essay Quality',
    title: str = 'Revised Essay Quality for Student Simulators and Student',
    xlabel: str = 'Temperature',
    ylabel: str = 'Essay Quality',
    save_path: str = '/home/inair/argument_revision/student_simulator_analysis/quality_plot.pdf'
) -> None:
    '''
        Plots the modifications by the student simulator and the humans
    '''

    # Colors for the metrics  
    llama_color_temperature_wise = '#1f77b4'
    openai_color_temperature_wise = '#ff7f0e'
    color_actual = '#2ca02c'
    color_initial = '#d62728'

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
    plt.axhline(y=initial_performance, color=color_initial, label=initial_label, linestyle='--')

    # Set the title and labels
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Set the legend
    plt.legend(fontsize=15)

    # Save the plot
    plt.savefig(save_path)

if __name__ == '__main__':
    plot_performance_list(
        llama_performance,
        openai_performance,
        revised_performance,
        initial_performance
    )