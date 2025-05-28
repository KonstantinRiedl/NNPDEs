import os
import torch
from glob import glob

# Base path where all run folders are saved
base_directory = "results/AllenCahnEquation/difficult/Adam/001"
output_folder = os.path.join(base_directory)
os.makedirs(output_folder, exist_ok=True)

# Collect all the result folders (sorted to keep order consistent)
run_folders = sorted(glob(os.path.join(base_directory, "individualruns/run*")))

# Initialize an empty dictionary for aggregating data
aggregated_results = {}


# Loop over result folders
for idx, folder in enumerate(run_folders):

    result_file = os.path.join(folder, "results.pt")
    print(f"{result_file} read and added.")
    if not os.path.exists(result_file):
        print(f"Warning: {result_file} not found, skipping.")
        continue
    
    run_data = torch.load(result_file)

    for key, value in run_data.items():
        if idx == 0:
            # On first run, initialize the aggregation buffer with extra dimension
            aggregated_results[key] = value
            
        else:
            # Concatenate along the new last dimension (num_runs)
            aggregated_results[key] = torch.cat((aggregated_results[key], value), dim=-1)

# Save aggregated results
torch.save(aggregated_results, os.path.join(output_folder, "results.pt"))
print(f"Combined results saved to {output_folder}/results.pt")

# --- Copy config.json from the first run folder to combined_results
import shutil

# Copy config file from the first run
config_src = os.path.join(run_folders[0], "config.json")
config_dst = os.path.join(output_folder, "config.json")
if os.path.exists(config_src):
    shutil.copyfile(config_src, config_dst)
    print(f"Copied config.json to {config_dst}")
else:
    print("Warning: config.json not found in the first run folder.")

# # --- Collect and merge best model files
# best_model_files = sorted(glob(os.path.join(base_results_dir, "20*_*", "best_g_beta*_N*.pt")))

# for file_path in best_model_files:
#     filename = os.path.basename(file_path)
#     dest_path = os.path.join(output_folder, filename)
#     if not os.path.exists(dest_path):
#         shutil.copyfile(file_path, dest_path)
#         print(f"Copied {filename} to {output_folder}")
#     else:
#         print(f"Skipping duplicate model: {filename}")