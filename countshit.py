import pandas as pd

# Function to count labels in a CSV file
def count_labels(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        print(df.head())
        print(df.iloc[:1,:])
        
        # Check if 'label' column exists
        if 'label' not in df.columns:
            print("Error: 'label' column not found in the CSV file.")
            return
        
        # Count occurrences of each label
        label_counts = df['label'].value_counts().to_dict()
        print(label_counts)
        
        # Get counts for label 0 and 1
        count_0 = label_counts.get(0, 0)
        count_1 = label_counts.get(1, 0)
        
        # Print results
        print(f"Count of label=0: {count_0}")
        print(f"Count of label=1: {count_1}")
        print(f"Total records: {len(df)}")
        
        # Return counts as a dictionary
        return {'label_0': count_0, 'label_1': count_1, 'total': len(df)}
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

# Run the function with your file path
if __name__ == "__main__":
    file_path = r"C:\Users\athar\Documents\GitHub\P.C.M\ai_vs_human_gener\test.csv"
    count_labels(file_path)