import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

#data csv
file_path = 'data.csv'

# load csv into pandas DataFrame
df = pd.read_csv(file_path)

# convert strings to lists for plotting
df['calculated_fsc_y'] = df['calculated_fsc_y'].apply(ast.literal_eval)
df['corrected_fsc_y'] = df['corrected_fsc_y'].apply(ast.literal_eval)
df['phase_randomised_y'] = df['phase_randomised_y'].apply(ast.literal_eval)
df['fsc_x'] = df['fsc_x'].apply(ast.literal_eval)

row = df.iloc[0]

sns.lineplot(x=row['fsc_x'], y=row['calculated_fsc_y'], label='Calculated Unmasked FSC')
sns.lineplot(x=row['fsc_x'], y=row['corrected_fsc_y'], label='Corrected Masked FSC')
sns.lineplot(x=row['fsc_x'], y=row['phase_randomised_y'], label='Phase Randomised FSC')
plt.title(row['entry'])
plt.xlabel('Spatial Frequency (1/Å)')
plt.ylabel('Correlation')
plt.axvline(x=row['1/resolution'], color='red', linestyle='--', label='resolution')
plt.grid(True)
plt.show()

