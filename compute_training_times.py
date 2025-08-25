import os
import pandas as pd

# Path to the folder containing your CSV files
folder_path = "training"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Check if 'time' column exists
        if "time" not in df.columns:
            print(f"{filename}: no time column")
            continue

        # Check if there are enough rows to compute differences
        if len(df["time"]) < 2:
            print(f"{filename}: not enough data to compute differences")
            continue

        # Convert the 'time' column to numeric (in case it is not)
        times = pd.to_numeric(df["time"], errors="coerce")
        if times.isnull().all():
            print(f"{filename}: time column could not be converted to numeric values")
            continue

        # Drop any rows where the conversion failed
        times = times.dropna()

        # If after cleaning there are less than 2 rows, skip
        if len(times) < 2:
            print(f"{filename}: not enough valid time data")
            continue

        # Calculate differences between consecutive times
        time_diff = times.diff().dropna()

        # Compute the average time difference
        avg_diff = time_diff.mean()

        print(f"{filename}: average time difference = {avg_diff}")
