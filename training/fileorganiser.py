import pandas as pd
import os
import shutil

'''
This is a quick script for extracting EMD-ID and classification type from a CSV file. The CSV file should be an export from label studio.
'''


def get_label_studio_information(csv_path):
    """
    Extracts EMD-ID and classification type from a CSV file exported from Label Studio.

    Parameters:
    csv_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: DataFrame containing EMD-ID and Type.
    """
    # Load the CSV file
    classes = pd.read_csv(csv_path)

    # Extract EMD-ID from the image column (currently file paths)
    extracted_image_ids = classes['image'].str.extract(r'(EMD-\d{4,5})', expand=False)
    image_presence_condition = extracted_image_ids.notna()

    # Extract the desired classification type from the 'choice' column
    classes['Type'] = None
    classes.loc[classes['choice'].str.contains('Typical', case=False, na=False), 'Type'] = '0_Typical'
    classes.loc[classes['choice'].str.contains('Atypical', case=False, na=False), 'Type'] = '1_Atypical'
    type_presence_condition = classes['Type'].notna()

    filtered_rows_df = classes[image_presence_condition & type_presence_condition].copy()

    # Generate the final DataFrame with EMD-ID and Type
    final_df = pd.DataFrame({
        'EMD-ID': extracted_image_ids[filtered_rows_df.index],
        'Type': filtered_rows_df['Type']
    })

    return final_df

df = get_label_studio_information('../Labelling/project-1-at-2025-08-19-07-53-06d9b5d0.csv')

# Track missing files
missing_files = []

for i in range(len(df)):
    emd_id = df.iloc[i]['EMD-ID']
    choice = df.iloc[i]['Type']
    fsc_path = f'../dataprepscripts/fscplots/{emd_id}.png'
    target_path = f'trainingdata/Typicality/{choice}/{emd_id}.png'
    if os.path.exists(fsc_path):
        shutil.copy(fsc_path, target_path)
    else:
        print(f"File not found: {fsc_path}, skipping...")
        missing_files.append(emd_id)
        continue
    print(f"Processed EMD-ID: {emd_id}, Type: {choice}")

# Write missing files to a text file
with open("../dataprepscripts/missing_files.txt", "w") as f:
    for missing_id in missing_files:
        f.write(missing_id + "\n")
print(f"Missing file list saved to missing_files.txt")