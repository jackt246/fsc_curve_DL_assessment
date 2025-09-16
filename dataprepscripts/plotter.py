import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

#data csv
file_path = 'missing_data.csv'

# load csv into pandas DataFrame
df = pd.read_csv(file_path)
df = df.dropna()

# load progress if exists
progress_file = 'plotting_progress.txt'

# convert strings to lists for plotting
df['masked_fsc_y'] = df['masked_fsc_y'].apply(ast.literal_eval)
df['corrected_fsc_y'] = df['corrected_fsc_y'].apply(ast.literal_eval)
df['phase_randomised_y'] = df['phase_randomised_y'].apply(ast.literal_eval)
df['fsc_x'] = df['fsc_x'].apply(ast.literal_eval)


def plotter(df, loc):
    row = df.iloc[loc]

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.lineplot(x=row['fsc_x'], y=row['masked_fsc_y'], label='Masked FSC')
    sns.lineplot(x=row['fsc_x'], y=row['corrected_fsc_y'], label='Corrected Masked FSC')
    sns.lineplot(x=row['fsc_x'], y=row['phase_randomised_y'], label='Phase Randomised FSC')

    plt.title(row['entry'])
    plt.xlabel('Spatial Frequency (1/Å)')
    plt.ylabel('Correlation')
    plt.axvline(x=row['1/resolution'], color='red', linestyle='--', label='resolution')
    plt.grid(True)
    plt.legend()

    return fig



for i in range(len(df)):
    try:
        fig = plotter(df, i)
        fig.savefig(f"fscplots/{df.iloc[i]['entry']}.png", bbox_inches='tight')
        plt.close(fig)

        # Save progress after each iteration
        with open(progress_file, 'w') as f:
            f.write(str(df.iloc[i]['entry']))

    except Exception as e:
        print(f"Error on index {i}: {e}")
        break


