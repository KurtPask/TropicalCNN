import os
import pandas as pd
import re

def extract_epoch(epoch_str):
    """Extracts the numeric epoch value from a string formatted as f"{epoch}_{batch}"."""
    match = re.match(r"(\d+)_\d+", str(epoch_str))
    return int(match.group(1)) if match else epoch_str  # Return extracted epoch or original value

def compute_avg_epoch_time(directory):
    results = []
    
    for file in os.listdir(directory):
        if file.endswith(".csv"):  # Ensure we only process CSV files
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            
            if "epoch" in df.columns and "time" in df.columns:
                df["epoch_numeric"] = df["epoch"].apply(extract_epoch)  # Extract numeric epoch
                df = df.sort_values(by=["epoch_numeric", "time"])  # Ensure epochs and times are sorted
                
                # Capture the last time recorded for each numeric epoch
                last_epoch_times = df.groupby("epoch_numeric")["time"].last()
                
                epoch_times = last_epoch_times.diff().dropna()  # Compute time differences
                avg_epoch_time = epoch_times.mean()  # Compute average time per epoch
                results.append({"file_name": file, "avg_epoch_time": avg_epoch_time})
    
    # Save results to a new CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv("average_epoch_times.csv", index=False)
    print("Results saved to average_epoch_times.csv")

# Specify the directory containing the CSV files
training_directory = "training"
compute_avg_epoch_time(training_directory)
