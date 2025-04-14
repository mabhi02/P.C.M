import requests
from datasets import load_dataset
import time
import os
import json
import csv
from collections import defaultdict, Counter

# Constants
DATASET_NAME = "xingjunm/WildDeepfake"
CACHE_DIR = "D:/huggingface_datasets"

# Monkey patch requests to increase timeout globally
original_request = requests.Session.request
def timeout_request(self, *args, **kwargs):
    kwargs.setdefault("timeout", 120)
    return original_request(self, *args, **kwargs)
requests.Session.request = timeout_request

def explore_dataset():
    print("🔍 Starting dataset exploration...")
    
    try:
        # Load dataset info
        print("📊 Loading dataset info (non-streaming mode)...")
        info = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR, 
                         trust_remote_code=True, split=None)
        
        # Print total size and split sizes
        total_size = sum(split.num_rows for split_name, split in info.items())
        print("\n📊 DATASET SIZE INFORMATION:")
        print(f"📊 TOTAL DATASET SIZE: {total_size:,} images")
        
        splits_info = {}
        for split_name, split in info.items():
            splits_info[split_name] = {"size": split.num_rows}
            print(f"  - {split_name} split: {split.num_rows:,} images")
        
        # Function to extract label from URL or path
        def get_label_from_path(path):
            if "fake" in path.lower() and "real" not in path.lower():
                return "AI"
            elif "real" in path.lower():
                return "Human"
            else:
                return "Unknown"
        
        # Explore dataset file structure
        print("\n📊 EXPLORING DATASET FILE STRUCTURE:")
        
        # Get first example from each split to analyze URL structure
        for split_name, split in info.items():
            if split.num_rows > 0:
                example = split[0]
                if "__url__" in example:
                    url = example["__url__"]
                    print(f"  - {split_name} first example URL: {url}")
        
        # Check for patterns in file paths
        print("\n📊 ANALYZING FILE PATHS FOR PATTERNS:")
        path_patterns = defaultdict(int)
        label_counts = defaultdict(lambda: defaultdict(int))
        
        for split_name, split in info.items():
            print(f"  - Analyzing {split_name} split file paths...")
            
            # Look at a larger sample to get better statistics
            sample_size = min(1000, split.num_rows)
            indices = list(range(0, split.num_rows, max(1, split.num_rows // sample_size)))[:sample_size]
            
            samples = split.select(indices)
            for example in samples:
                if "__url__" in example:
                    url = example["__url__"]
                    
                    # Extract key parts of the path
                    parts = url.split('/')
                    for part in parts:
                        if "fake" in part.lower() or "real" in part.lower():
                            path_patterns[part] += 1
                    
                    # Determine label
                    label = get_label_from_path(url)
                    label_counts[split_name][label] += 1
        
        # Print path patterns
        print("\n📊 FILE PATH PATTERNS (showing patterns with 'real' or 'fake'):")
        for pattern, count in sorted(path_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {pattern}: {count} occurrences")
        
        # Print label distribution based on URL analysis
        print("\n📊 LABEL DISTRIBUTION (based on URL/path analysis):")
        for split_name, counts in label_counts.items():
            total = sum(counts.values())
            print(f"  📊 {split_name.upper()} SPLIT (sampled {total} examples):")
            for label, count in counts.items():
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"    - {label}: {count} ({percentage:.2f}%)")
            
            # Store in our splits info dictionary
            splits_info[split_name].update({
                "sampled": total,
                "labels": dict(counts)
            })
        
        # Extrapolate to full dataset
        print("\n📊 EXTRAPOLATED LABEL DISTRIBUTION FOR FULL DATASET:")
        total_ai = 0
        total_human = 0
        total_unknown = 0
        
        for split_name, info_dict in splits_info.items():
            if "labels" in info_dict and "sampled" in info_dict and info_dict["sampled"] > 0:
                split_size = info_dict["size"]
                sampled = info_dict["sampled"]
                labels = info_dict["labels"]
                
                ai_ratio = labels.get("AI", 0) / sampled
                human_ratio = labels.get("Human", 0) / sampled
                unknown_ratio = labels.get("Unknown", 0) / sampled
                
                estimated_ai = int(split_size * ai_ratio)
                estimated_human = int(split_size * human_ratio)
                estimated_unknown = int(split_size * unknown_ratio)
                
                print(f"  📊 {split_name.upper()} SPLIT (extrapolated to {split_size:,} examples):")
                print(f"    - AI: ~{estimated_ai:,} ({ai_ratio*100:.2f}%)")
                print(f"    - Human: ~{estimated_human:,} ({human_ratio*100:.2f}%)")
                print(f"    - Unknown: ~{estimated_unknown:,} ({unknown_ratio*100:.2f}%)")
                
                total_ai += estimated_ai
                total_human += estimated_human
                total_unknown += estimated_unknown
        
        # Overall totals
        print(f"\n📊 OVERALL TOTALS (extrapolated):")
        print(f"  - AI: ~{total_ai:,} ({(total_ai/total_size)*100:.2f}%)")
        print(f"  - Human: ~{total_human:,} ({(total_human/total_size)*100:.2f}%)")
        print(f"  - Unknown: ~{total_unknown:,} ({(total_unknown/total_size)*100:.2f}%)")
        
        # Look for specific files or directories mentioning "real"
        print("\n🔍 SEARCHING FOR 'REAL' IMAGES IN DATASET...")
        
        real_found = False
        for split_name, split in info.items():
            if label_counts[split_name].get("Human", 0) > 0:
                real_found = True
                
                # Find examples of "real" images
                print(f"  - Found potential real images in {split_name} split")
                found_count = 0
                
                for i in range(min(split.num_rows, 10000)):
                    if i % 1000 == 0:
                        print(f"    Checking example {i}/{min(split.num_rows, 10000)}...")
                    
                    try:
                        example = split[i]
                        if "__url__" in example:
                            url = example["__url__"]
                            if "real" in url.lower():
                                print(f"    🔍 Found real image at index {i}: {url}")
                                found_count += 1
                                if found_count >= 5:  # Find at most 5 examples
                                    break
                    except:
                        continue
        
        if not real_found:
            print("  ⚠️ No 'real' images found in initial scan. The dataset might only contain fake images.")
            print("  🔍 Trying alternative search approaches...")
            
            # Check for other indicators that might distinguish real from fake
            print("\n🔍 EXPLORING ALTERNATIVE LABEL INDICATORS:")
            
            # Sample some examples to look for other potential indicators
            for split_name, split in info.items():
                print(f"  - Checking fields in {split_name} split...")
                
                if split.num_rows > 0:
                    example = split[0]
                    print(f"    Fields available: {list(example.keys())}")
                    
                    # Check for any metadata that might indicate real/fake
                    for key in example.keys():
                        if key not in ["png", "__url__", "__key__"]:
                            values = Counter()
                            for i in range(min(100, split.num_rows)):
                                try:
                                    sample = split[i]
                                    if key in sample:
                                        values[str(sample[key])] += 1
                                except:
                                    continue
                            
                            if len(values) > 0:
                                print(f"    Field '{key}' values: {dict(values.most_common(5))}")
        
        print("\n✅ Dataset exploration completed!")
        
    except Exception as e:
        print(f"❌ Error exploring dataset: {e}")

# Run the exploration function
explore_dataset()