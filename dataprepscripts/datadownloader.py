from emdb.client import EMDB
import matplotlib.pyplot
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import threading

# initialize the EMDB client
client = EMDB()

# Search for entries with a half map filename
results = client.search('half_map_filename:[* TO *]')

# iterate through the results and collect the validation data
data_lock = threading.Lock()
data = []
failed_entries = []


for entry in tqdm(results, desc="Processing entries"):
    try:
        validation = entry.get_validation()
        data.append({
            'entry': entry.id,
            'calculated_fsc_y': validation.plots.fsc.fsc,
            'corrected_fsc_y': validation.plots.fsc.fsc_corrected,
            'phase_randomised_y': validation.plots.fsc.phaserandomization,
            'fsc_x': validation.plots.fsc.level,
            'resolution': validation.plots.fsc.resolution,
            '1/resolution': 1 / validation.plots.fsc.resolution
        })
    except Exception as e:
        print(f"Error processing entry {entry.id}: {e}")
        failed_entries.append(entry.id)

# convert dictionary to pandas DataFrame
df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)

# Save the failed entries to a text file
with open('failed_entries.txt', 'w') as f:
    for entry in failed_entries:
        f.write(f"{entry}\n")
