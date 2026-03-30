import os
import pandas as pd

def load_dataset():
    # Get the absolute path to this script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Go one level up (to the project root) and into the 'data' folder
    file_path = os.path.join(base_dir, '..', 'data', 'Dataset_English_Hindi.csv')

    # Normalize the path (works across Windows, macOS, Linux)
    file_path = os.path.normpath(file_path)

    # Read the CSV
    df = pd.read_csv(file_path)

    # Clean whitespace if the columns exist
    if 'english' in df.columns and 'hindi' in df.columns:
        df['english'] = df['english'].astype(str).str.strip()
        df['hindi'] = df['hindi'].astype(str).str.strip()

    return df
