import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

#data csv
file_path = 'classification_results.csv'

# load csv into pandas DataFrame
df = pd.read_csv(file_path)
df = df.dropna()

df_filtered = df[df['Result'] == 'Incorrect']
print(len(df_filtered))


# Plot Predicted Probability (0_Typical) by index, colored by True Label
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_filtered, x=df_filtered.index, y='Predicted Probability (0_Typical)', hue='True Label', palette='Set1')
plt.title('Predicted Probability (0_Typical) by True Label')
plt.xlabel('Index')
plt.ylabel('Predicted Probability (0_Typical)')
plt.legend(title='True Label')
plt.tight_layout()
plt.show()


