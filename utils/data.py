import os
import shutil
import kagglehub

# Download the latest version of the dataset using kagglehub
download_path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")
print("Downloaded dataset folder:", download_path)

# Specify your desired destination folder (change this path as needed)
destination_path = "./images/ai_vs_human_generated_dataset"

# Ensure the parent directory exists
os.makedirs(os.path.dirname(destination_path), exist_ok=True)

# Check if the destination folder already exists
if os.path.exists(destination_path):
    print(f"Destination folder '{destination_path}' already exists. Please remove it or choose a different path.")
else:
    # Move the downloaded dataset folder to the destination
    shutil.move(download_path, destination_path)
    print(f"Dataset moved to: {destination_path}")

