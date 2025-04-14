import pandas as pd

def analyze_labels(csv_path):
    try:
        df = pd.read_csv(csv_path)

        if 'label' not in df.columns:
            print("The CSV does not contain a 'label' column.")
            return

        total_images = len(df)
        label_1_count = (df['label'] == 1).sum()
        label_0_count = (df['label'] == 0).sum()

        print(f"Total number of images: {total_images}")
        print(f"Number of images with label 1: {label_1_count}")
        print(f"Number of images with label 0: {label_0_count}")

    except Exception as e:
        print(f"Error reading CSV: {e}")

# File path
csv_file = r'images\ai_vs_human_generated_dataset\test.csv'
analyze_labels(csv_file)
