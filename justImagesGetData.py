import requests
from datasets import load_dataset
import time
import os
import json
import csv
import random
from PIL import Image

# Constants
DATASET_NAME = "xingjunm/WildDeepfake"
CACHE_DIR = "D:/huggingface_datasets"
TRAIN_FRACTION = 1/300  # Fraction of dataset to use for training
TEST_FRACTION = 1/900   # Fraction of dataset to use for testing
OUTPUT_FOLDER = "ai_vs_human_gener"  # Main output folder
TRAIN_FOLDER = os.path.join(OUTPUT_FOLDER, "train_data")
TEST_FOLDER = os.path.join(OUTPUT_FOLDER, "test_data_v2")

# Create output directories
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# Monkey patch requests to increase timeout globally
original_request = requests.Session.request
def timeout_request(self, *args, **kwargs):
    kwargs.setdefault("timeout", 120)
    return original_request(self, *args, **kwargs)
requests.Session.request = timeout_request

# Function to save a PIL Image object
def save_pil_image(image, filename):
    try:
        # Save the PIL Image directly
        image.save(filename)
        return True
    except Exception as e:
        print(f"  ⚠️ Error saving image: {e}")
        return False

# Function to extract label from URL
def get_label_from_url(url):
    if "fake" in url.lower():
        return "AI"  # Deepfakes are AI-generated
    elif "real" in url.lower():
        return "Human"  # Real images are human-generated
    else:
        return "Unknown"

def download_and_process_dataset():
    print("🔄 Starting dataset processing...")
    
    try:
        # Load dataset with streaming to avoid loading everything
        dataset = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR, 
                          trust_remote_code=True, streaming=True)
        
        # Process train and test splits
        splits = list(dataset.keys())
        print(f"📊 Available splits: {splits}")
        
        # Initialize CSV files
        train_csv_path = os.path.join(OUTPUT_FOLDER, "train.csv")
        test_csv_path = os.path.join(OUTPUT_FOLDER, "test.csv")
        
        # Write headers
        with open(train_csv_path, 'w', newline='') as train_file:
            train_writer = csv.writer(train_file)
            train_writer.writerow(["image_path", "label"])
        
        with open(test_csv_path, 'w', newline='') as test_file:
            test_writer = csv.writer(test_file)
            test_writer.writerow(["image_path", "label"])
        
        train_count = 0
        test_count = 0
        total_count = 0
        
        # Process each split
        for split_name in splits:
            print(f"⬇️ Processing split: {split_name}")
            
            # Use the streaming iterator
            stream = dataset[split_name]
            
            # Calculate how many samples we'll extract from this split
            for example in stream:
                total_count += 1
                
                # Extract label from URL
                if "__url__" in example:
                    url = example["__url__"]
                    label = get_label_from_url(url)
                else:
                    label = "Unknown"
                
                # Decide if this should go to train, test, or be skipped
                if total_count % 300 == 0:
                    # Save to train set
                    if 'png' in example:
                        train_count += 1
                        image_filename = f"train_image_{train_count}.png"
                        image_path = os.path.join(TRAIN_FOLDER, image_filename)
                        
                        # Save image
                        saved = save_pil_image(example['png'], image_path)
                        
                        if saved:
                            print(f"  ✅ Saved train image {train_count}: {label}")
                            
                            # Update CSV
                            with open(train_csv_path, 'a', newline='') as train_file:
                                train_writer = csv.writer(train_file)
                                rel_path = os.path.join("train_data", image_filename)
                                train_writer.writerow([rel_path, label])
                
                elif total_count % 900 == 0:
                    # Save to test set
                    if 'png' in example:
                        test_count += 1
                        image_filename = f"test_image_{test_count}.png"
                        image_path = os.path.join(TEST_FOLDER, image_filename)
                        
                        # Save image
                        saved = save_pil_image(example['png'], image_path)
                        
                        if saved:
                            print(f"  ✅ Saved test image {test_count}: {label}")
                            
                            # Update CSV
                            with open(test_csv_path, 'a', newline='') as test_file:
                                test_writer = csv.writer(test_file)
                                rel_path = os.path.join("test_data_v2", image_filename)
                                test_writer.writerow([rel_path, label])
                
                # Progress indicator
                if total_count % 1000 == 0:
                    print(f"  📊 Processed {total_count} examples, Train: {train_count}, Test: {test_count}")
                
                # Stop if we have enough samples
                if train_count >= 100 and test_count >= 30:
                    print("✅ Collected enough samples, stopping.")
                    break
        
        print(f"✅ Dataset processing completed: {train_count} train images, {test_count} test images")
        print(f"📂 Train CSV saved to: {train_csv_path}")
        print(f"📂 Test CSV saved to: {test_csv_path}")
        
        # Print absolute paths for easy access
        print(f"📂 Output folder: {os.path.abspath(OUTPUT_FOLDER)}")
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")

# Run the download and processing function
download_and_process_dataset()