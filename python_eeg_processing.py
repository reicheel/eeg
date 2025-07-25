######## this code uses the .set file exported from matlab and creates separate excel sheets for each event code 
####### and one spreadsheet for the entire block with all events 

import mne
import pandas as pd
import re
import os

# Define file paths
set_file = r"path"
output_dir = r"path"

# Extract block number from the filename (e.g., cam###_block3.set → block3)
block_number = re.search(r'block\d+', set_file)
block_suffix = f"_{block_number.group()}" if block_number else ""

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load EEG epochs dataset
epochs = mne.io.read_epochs_eeglab(set_file)

# Print info about the dataset
print(epochs.info)
print(epochs)

# Convert epochs to a Pandas DataFrame (time-series format)
df = epochs.to_data_frame()

# Extract event markers from epochs (each epoch corresponds to one event)
event_codes = epochs.events[:, 2]  # Extract event IDs

# Expand event markers to match time-series data
n_times = df.shape[0]  # Total number of rows in time-series dataframe
n_epochs = len(event_codes)  # Number of epochs

#Each epoch has multiple time points, so we need to repeat event codes
n_timepoints_per_epoch = n_times // n_epochs  # Approximate number of time points per epoch
df["event_code"] = [event_codes[i // n_timepoints_per_epoch] for i in range(n_times)]

#Save full epoch data with block number in filename
erp_csv = os.path.join(output_dir, f"erp_data{block_suffix}.csv")
df.to_csv(erp_csv, index=False)
print(f"ERP data saved as '{erp_csv}'")

#Extract and save average ERP per event code
evoked_dict = {cond: epochs[cond].average() for cond in epochs.event_id}

for cond, evoked in evoked_dict.items():
    df_evoked = evoked.to_data_frame()

    #Sanitize condition names for filenames
    safe_cond = re.sub(r'[^\w\-_]', '_', cond)  # Replace special characters with "_"
    
    #Save average ERP data with block number in filename
    csv_filename = os.path.join(output_dir, f"erp_average_{safe_cond}{block_suffix}.csv")
    df_evoked.to_csv(csv_filename, index=False)
    print(f"Saved ERP average for condition {cond} as '{csv_filename}'")



####################################################################################################
#### uses the spreadsheets from the mne script to plot waveforms for participants over 3 blocks ####
####################################################################################################


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

file_paths = [
    r"path",
    r"path",
    r"path""
]
output_dir = r"path"
electrode = "Pz"  # Change to "Cz", "Fz", etc. as needed 
p300_window = (300, 500)  # P300 time window (ms)
lpp_window = (500, 1200)  # LPP time window (ms)

os.makedirs(output_dir, exist_ok=True)

# defines Sport vs. Non-Sport conditions
condition_pairs = {
    "Neutral": ("B1(Sport_Neutral)", "B1(NS_Neutral)"),
    "Threat": ("B2(Sport_Threat)", "B2(NS_Threat)"),
    "Intense": ("B3(Sport_Intense)", "B3(NS_Intense)")
}

# defines Sport-Only and Non-Sport-Only conditions
sport_conditions = {
    "Sport_Neutral": "B1(Sport_Neutral)",
    "Sport_Threat": "B2(Sport_Threat)",
    "Sport_Intense": "B3(Sport_Intense)"
}

nonsport_conditions = {
    "NS_Neutral": "B1(NS_Neutral)",
    "NS_Threat": "B2(NS_Threat)",
    "NS_Intense": "B3(NS_Intense)"
}

# Load all block data and concatenate them
all_data = []
for file in file_paths:
    df = pd.read_csv(file)
    all_data.append(df)

erp_data = pd.concat(all_data, ignore_index=True)

# Function to compute mean and 95% CI
def compute_mean_ci(data, condition):
    """Computes mean and 95% CI for a given condition"""
    grouped = data.groupby("time")[electrode]
    mean = grouped.mean()
    sem = grouped.sem()  # Standard error of the mean
    ci = 1.96 * sem  # 95% Confidence Interval
    return mean, ci

#### plot for sport vs non-sport ###
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

for i, (label, (sport_cond, ns_cond)) in enumerate(condition_pairs.items()):
    ax = axes[i]

    # Compute mean and CI for Sport
    sport_data = erp_data[erp_data["condition"] == sport_cond]
    sport_mean, sport_ci = compute_mean_ci(sport_data, sport_cond)

    # Compute mean and CI for Non-Sport
    ns_data = erp_data[erp_data["condition"] == ns_cond]
    ns_mean, ns_ci = compute_mean_ci(ns_data, ns_cond)

    # Convert time to ms
    time_values = sport_mean.index.values * 1000

    # Plot Sport condition with CI
    ax.plot(time_values, sport_mean, label=f"Sport {label}", color="blue")
    ax.fill_between(time_values, sport_mean - sport_ci, sport_mean + sport_ci, color="blue", alpha=0.2)

    # Plot Non-Sport condition with CI
    ax.plot(time_values, ns_mean, label=f"Non-Sport {label}", color="red")
    ax.fill_between(time_values, ns_mean - ns_ci, ns_mean + ns_ci, color="red", alpha=0.2)

    # Highlight P300 and LPP windows
    ax.axvspan(p300_window[0], p300_window[1], color='gray', alpha=0.2, label="P300 Window")
    ax.axvspan(lpp_window[0], lpp_window[1], color='blue', alpha=0.1, label="LPP Window")

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylabel("ERP Amplitude (µV)")
    ax.set_title(f"Sport vs. Non-Sport ({label}) at {electrode}")
    ax.legend()

axes[-1].set_xlabel("Time (ms)")
plt.tight_layout()

# Save figure
save_path = os.path.join(output_dir, "Participant_ERP_Sport_vs_NonSport.png")
plt.savefig(save_path, dpi=300)
print(f"Saved plot: {save_path}")

plt.show()

### plot of sport conditions only ###
fig, ax = plt.subplots(figsize=(10, 5))

for label, cond in sport_conditions.items():
    cond_data = erp_data[erp_data["condition"] == cond]
    mean_erp = cond_data.groupby("time")[electrode].mean()
    
    # Convert time to ms
    time_values = mean_erp.index.values * 1000
    
    # Plot Sport condition
    ax.plot(time_values, mean_erp, label=label)

# Highlight P300 and LPP windows
ax.axvspan(p300_window[0], p300_window[1], color='gray', alpha=0.2, label="P300 Window")
ax.axvspan(lpp_window[0], lpp_window[1], color='blue', alpha=0.1, label="LPP Window")

ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("ERP Amplitude (µV)")
ax.set_title(f"Sport Conditions at {electrode}")
ax.legend()

# Save figure
save_path = os.path.join(output_dir, "Participant_ERP_Sport_Only.png")
plt.savefig(save_path, dpi=300)
print(f"Saved plot: {save_path}")

plt.show()

### non sport conditions only ###
fig, ax = plt.subplots(figsize=(10, 5))

for label, cond in nonsport_conditions.items():
    cond_data = erp_data[erp_data["condition"] == cond]
    mean_erp = cond_data.groupby("time")[electrode].mean()
    
    # Convert time to ms
    time_values = mean_erp.index.values * 1000
    
    # Plot Non-Sport condition
    ax.plot(time_values, mean_erp, label=label)

# Highlight P300 and LPP windows
ax.axvspan(p300_window[0], p300_window[1], color='gray', alpha=0.2, label="P300 Window")
ax.axvspan(lpp_window[0], lpp_window[1], color='blue', alpha=0.1, label="LPP Window")

ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("ERP Amplitude (µV)")
ax.set_title(f"Non-Sport Conditions at {electrode}")
ax.legend()

# Save figure
save_path = os.path.join(output_dir, "Participant_ERP_NonSport_Only.png")
plt.savefig(save_path, dpi=300)
print(f"Saved plot: {save_path}")

plt.show()



####################################################################################################
################# this section merges individual subjects blocks together ##########################
####################################################################################################

import pandas as pd
import os
from tqdm import tqdm  # Progress bar

# Define participant list (adjust based on actual participant IDs)
participants = [f"subject_{str(i).zfill(3)}" for i in range(##, ##)]

# Define the path where ERP data is stored
base_path = r"path"
output_path = os.path.join(base_path, "_data_processing")

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Store all participants' combined data for study-wide merging
all_participants_data = []

print("\nProcessing participant data...\n")

# Loop through participants with a progress bar
for participant in tqdm(participants, desc="Processing Participants", unit="participant"):
    participant_path = os.path.join(base_path, participant, "eeg", "Processed")

    # Define ERP block file paths
    file_paths = {
        "block1": os.path.join(participant_path, "erp_data_block1.csv"),
        "block2": os.path.join(participant_path, "erp_data_block2.csv"),
        "block3": os.path.join(participant_path, "erp_data_block3.csv"),
    }
    
    # Check if all files exist
    if all(os.path.exists(fp) for fp in file_paths.values()):
        # Load and label block data
        erp_data = {block: pd.read_csv(path) for block, path in file_paths.items()}
        for block, df in erp_data.items():
            df["block_number"] = block  # Add block identifier
            df["subjectID"] = participant  # Add participant ID

        # Merge all blocks for the participant
        participant_combined_df = pd.concat(erp_data.values(), ignore_index=True)

        # Save participant combined data
        participant_combined_path = os.path.join(output_path, f"{participant}_eeg.csv")
        participant_combined_df.to_csv(participant_combined_path, index=False)

        # Append to study-wide dataset
        all_participants_data.append(participant_combined_df)
    
    else:
        print(f"⚠️ Warning: Missing data for {participant}, skipping.")


####################################################################################################
##################### this section merges all subjects together ####################################
####################################################################################################


import pandas as pd
import os
from tqdm import tqdm  # Progress bar

# Define participant list
participants = [f"subject_{str(i).zfill(3)}" for i in range(##, ##)]

# Define base path and output path
base_path = r"path"
output_path = os.path.join(base_path, "_data_processing")

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Store all participants' data for merging
all_participants_data = []

print("\nProcessing participant data...\n")

for participant in tqdm(participants, desc="Processing Participants"):
    participant_file = os.path.join(base_path, participant, "eeg", "Processed", f"{participant}_eeg.csv")
    
    if os.path.exists(participant_file):
        try:
            df = pd.read_csv(participant_file, low_memory=False)
            df.insert(0, "participant_id", participant)  # Add participant ID column
            all_participants_data.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {participant_file}: {e}")
    else:
        print(f"⚠️ File not found: {participant_file}")

# Merge all participants' data into a study-wide dataset
if all_participants_data:
    study_wide_df = pd.concat(all_participants_data, ignore_index=True)
    
    # Save the study-wide dataset
    study_wide_path = os.path.join(output_path, "eeg.csv")
    study_wide_df.to_csv(study_wide_path, index=False)
    print(f"\n✅ Study-wide dataset saved: {study_wide_path}")
else:
    print("\n⚠️ No participant data found to merge.")