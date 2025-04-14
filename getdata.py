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
OUTPUT_FOLDER = "ai_vs_human_gener"  # Main output folder
TRAIN_FOLDER = os.path.join(OUTPUT_FOLDER, "train_data")
TEST_FOLDER = os.path.join(OUTPUT_FOLDER, "test_data_v2")

# Define sample sizes - low values for testing
TRAIN_SAMPLES = 80000  # Number of images in train set
TEST_SAMPLES = 10000    # Number of images in test set

# Split sizes from previous exploration
TRAIN_SPLIT_SIZE = 1014437
TEST_SPLIT_SIZE = 165662

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

# Function to save a PIL Image object at full resolution
def save_pil_image(image, filename):
    try:
        # Check if image is at full 512x512 resolution
        if hasattr(image, "size"):
            width, height = image.size
            print(f"  📐 Image dimensions: {width}x{height}")
        
        # Save the PIL Image directly without resizing
        image.save(filename)
        return True
    except Exception as e:
        print(f"  ⚠️ Error saving image: {e}")
        return False

# Function to extract label from URL
def get_label_from_url(url):
    if "fake" in url.lower() and "real" not in url.lower():
        return "AI"  # Deepfakes are AI-generated
    elif "real" in url.lower():
        return "Human"  # Real images are human-generated
    else:
        return "Unknown"

def download_and_process_dataset():
    print("🔄 Starting dataset processing...")
    
    try:
        # Generate random indices for each split independently
        print("🎲 Generating random indices for train and test splits")
        
        # Create random indices for train split
        train_indices = random.sample(range(TRAIN_SPLIT_SIZE), TRAIN_SAMPLES)
        print(f"🎲 Random train indices: {train_indices}")
        
        # Create random indices for test split
        test_indices = random.sample(range(TEST_SPLIT_SIZE), TEST_SAMPLES)
        print(f"🎲 Random test indices: {test_indices}")
        
        # Load dataset info
        print("📊 Loading dataset...")
        info = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR, 
                         trust_remote_code=True, split=None)
        
        splits = list(info.keys())
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
        
        # Counters for train/test sets
        train_ai = 0
        train_human = 0
        test_ai = 0
        test_human = 0
        
        # Process train samples
        print("\n⬇️ Processing train samples from 'train' split...")
        for i, idx in enumerate(train_indices):
            print(f"  🔍 Retrieving train example {i+1}/{len(train_indices)} (idx: {idx})")
            
            try:
                # Get the example from the train split
                example = info["train"][idx]
                
                # Extract label
                if "__url__" in example:
                    url = example["__url__"]
                    label = get_label_from_url(url)
                else:
                    label = "Unknown"
                
                # Skip unknown labels
                if label == "Unknown":
                    print(f"  ⚠️ Skipping example with unknown label")
                    continue
                
                # Track label distribution
                if label == "AI":
                    train_ai += 1
                elif label == "Human":
                    train_human += 1
                
                if 'png' in example:
                    image_filename = f"train_image_{i+1}_{label}.png"
                    image_path = os.path.join(TRAIN_FOLDER, image_filename)
                    
                    # Save image at full resolution
                    saved = save_pil_image(example['png'], image_path)
                    
                    if saved:
                        print(f"  ✅ Saved train image {i+1}: {label}")
                        
                        # Update CSV
                        with open(train_csv_path, 'a', newline='') as train_file:
                            train_writer = csv.writer(train_file)
                            rel_path = os.path.join("train_data", image_filename)
                            train_writer.writerow([rel_path, label])
            except Exception as e:
                print(f"  ⚠️ Error processing train example at index {idx}: {e}")
        
        # Process test samples
        print("\n⬇️ Processing test samples from 'test' split...")
        for i, idx in enumerate(test_indices):
            print(f"  🔍 Retrieving test example {i+1}/{len(test_indices)} (idx: {idx})")
            
            try:
                # Get the example from the test split
                example = info["test"][idx]
                
                # Extract label
                if "__url__" in example:
                    url = example["__url__"]
                    label = get_label_from_url(url)
                else:
                    label = "Unknown"
                
                # Skip unknown labels
                if label == "Unknown":
                    print(f"  ⚠️ Skipping example with unknown label")
                    continue
                
                # Track label distribution
                if label == "AI":
                    test_ai += 1
                elif label == "Human":
                    test_human += 1
                
                if 'png' in example:
                    image_filename = f"test_image_{i+1}_{label}.png"
                    image_path = os.path.join(TEST_FOLDER, image_filename)
                    
                    # Save image at full resolution
                    saved = save_pil_image(example['png'], image_path)
                    
                    if saved:
                        print(f"  ✅ Saved test image {i+1}: {label}")
                        
                        # Update CSV
                        with open(test_csv_path, 'a', newline='') as test_file:
                            test_writer = csv.writer(test_file)
                            rel_path = os.path.join("test_data_v2", image_filename)
                            test_writer.writerow([rel_path, label])
            except Exception as e:
                print(f"  ⚠️ Error processing test example at index {idx}: {e}")
        
        # Print final statistics
        print("\n📊 FINAL STATISTICS:")
        train_count = train_ai + train_human
        test_count = test_ai + test_human
        
        print(f"\n📊 TRAIN SET: {train_count} images")
        print(f"  - AI-generated: {train_ai} ({(train_ai/train_count)*100 if train_count > 0 else 0:.2f}%)")
        print(f"  - Human: {train_human} ({(train_human/train_count)*100 if train_count > 0 else 0:.2f}%)")
        
        print(f"\n📊 TEST SET: {test_count} images")
        print(f"  - AI-generated: {test_ai} ({(test_ai/test_count)*100 if test_count > 0 else 0:.2f}%)")
        print(f"  - Human: {test_human} ({(test_human/test_count)*100 if test_count > 0 else 0:.2f}%)")
        
        print(f"\n✅ Dataset processing completed: {train_count} train images, {test_count} test images")
        print(f"📂 Train CSV saved to: {train_csv_path}")
        print(f"📂 Test CSV saved to: {test_csv_path}")
        
        # Print absolute paths for easy access
        print(f"📂 Output folder: {os.path.abspath(OUTPUT_FOLDER)}")
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")

# Run the download and processing function
download_and_process_dataset()