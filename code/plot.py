import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_environment(env_name):
    redq_file = f"../results/{env_name}_redq_results.csv" #read results files
    sac_file = f"../results/{env_name}_SAC_results.csv"

    if not os.path.exists(redq_file) or not os.path.exists(sac_file): #check existance of files
        print(f"skipping {env_name}: CSV files not found.")
        return

    df_redq = pd.read_csv(redq_file) #csv to pandas dataframe
    df_sac = pd.read_csv(sac_file)

    df_redq['Algorithm'] = 'REDQ (UTD=20, N=10)' #add a column to save the algorithm name and parameters
    df_sac['Algorithm'] = 'SAC (UTD=1, N=2)'

    df_combined = pd.concat([df_redq, df_sac], ignore_index=True) #join both dataframes

    sns.set_theme(style="whitegrid") #visual style
    plt.figure(figsize=(10, 6))

    sns.lineplot( #create the plot
        data=df_combined,
        x='Step',
        y='Average_Return',
        hue='Algorithm',
        errorbar='sd',  #plots the standard deviation as the shaded region
        linewidth=2
    )

    plt.title(f'Learning Curve: {env_name}', fontsize=16, fontweight='bold') #graph labels
    plt.xlabel('Environment Steps', fontsize=14)
    plt.ylabel('Average Return', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')

    plt.ylim(0, 6000) #fixed scale

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))) #format the numbers in x axis

    output_filename = f"../results/{env_name}_comparison_plot.png" #save as png in results folder
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

    print(f"successfully generated {output_filename}!")


if __name__ == "__main__":
    plot_environment('Hopper-v4') #generate a graph for each environment
    plot_environment('Walker2d-v4')