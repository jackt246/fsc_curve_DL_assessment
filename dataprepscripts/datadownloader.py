from emdb.client import EMDB
import matplotlib.pyplot
import pandas as pd
from tqdm import tqdm
import os

# initialize the EMDB client
client = EMDB()

# Search for entries with a half map filename
results = client.search('half_map_filename:[* TO *]')

# Define the chunk size for saving data
chunk_size = 100
output_csv = 'data.csv'
failed_entries_file = 'failed_entries.txt'

# prep containers for data and failed entries
data = []
failed_entries = []

# Function to process a single entry
def process_entry(entry):
    try:
        validation = entry.get_validation()
        result = {
            'entry': entry.id,
            'calculated_fsc_y': validation.plots.fsc.fsc,
            'corrected_fsc_y': validation.plots.fsc.fsc_corrected,
            'phase_randomised_y': validation.plots.fsc.phaserandomization,
            'fsc_x': validation.plots.fsc.level,
            'resolution': validation.plots.fsc.resolution,
            '1/resolution': 1 / validation.plots.fsc.resolution
        }
        return result
    except TimeoutError as e:
        print(f"Timeout occurred for entry {entry.id}: {e}")
        return None
    except Exception as e:
        print(f"Error processing entry {entry.id}: {e}")
        return None

# function to save failed entries to a file
def save_failed_entries(failed_entries, failed_entries_file):
    with open(failed_entries_file, 'a') as f:
        for entry_id in failed_entries:
            f.write(f"{entry_id}\n")


def main():
    # Check if the CSV file exists to determine the starting point
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        start_index = len(existing_df)
        data = existing_df.to_dict('records')  # Load existing data
        print(f"Resuming from entry {start_index}")
    else:
        start_index = 0
        data = []
        print("Starting from scratch")

    # Load failed entries
    if os.path.exists(failed_entries_file):
        with open(failed_entries_file, 'r') as f:
            loaded_failed_entries = {line.strip() for line in f}
    else:
        loaded_failed_entries = set()
    failed_entries = []

    # Process entries in chunks
    for i in tqdm(range(start_index, len(results)), initial=start_index, total=len(results),
                      desc="Processing entries"):
        entry = results[i]
        if entry.id in failed_entries:
            print(f"Skipping failed entry {entry.id}")
            continue

        result = process_entry(entry)
        if result is None:
            failed_entries.append(entry.id)
        else:
            data.append(result)

        # Save the DataFrame every chunk_size entries
        if (i + 1) % chunk_size == 0:
            df = pd.DataFrame(data)
            if os.path.exists(output_csv):
                df.to_csv(output_csv, mode='a', header=False, index=False)  # Append to existing file
            else:
                df.to_csv(output_csv, index=False)  # Create a new file
            data = []  # Clear the data list

            save_failed_entries(failed_entries, failed_entries_file)
            failed_entries = []

    # Save any remaining entries
    if data:
        df = pd.DataFrame(data)
        if os.path.exists(output_csv):
            df.to_csv(output_csv, mode='a', header=False, index=False)  # Append to existing file
        else:
            df.to_csv(output_csv, index=False)  # Create a new file
    if failed_entries:
        save_failed_entries(failed_entries, failed_entries_file)

if __name__ == "__main__":
    main()