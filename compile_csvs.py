import os
import glob
import pandas as pd

# Specify the folder where your CSV files are located
folder_path = 'attack/'  # <-- Update this path

# Find all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

if not csv_files:
    print("No CSV files found in the specified folder.")
    exit(1)

# Read header from the first CSV file using pandas (ensures proper parsing)
first_df = pd.read_csv(csv_files[0], nrows=0)
header_line = list(first_df.columns)
header_line.append('file_name')  # Append new column for file name

compiled_rows = []

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        
        last_row = df.iloc[-1].tolist()
        # If the number of columns in the row doesn't match,
        # adjust by trimming or padding
        expected_cols = len(header_line) - 1  # without the file name column
        if len(last_row) > expected_cols:
            last_row = last_row[:expected_cols]
        elif len(last_row) < expected_cols:
            last_row += [None] * (expected_cols - len(last_row))
        
        last_row.append(os.path.basename(csv_file))
        compiled_rows.append(last_row)
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

compiled_df = pd.DataFrame(compiled_rows, columns=header_line)
output_file = os.path.join(folder_path, 'compiled_last_rows.csv')
compiled_df.to_csv(output_file, index=False)
print(f"Compilation complete. Output saved to {output_file}")
