import csv
from pathlib import Path

# File path for the CSV file
results_folder_path = Path(".").joinpath("results")
best_makers = []
# Function to iterate over the CSV file
def iterate_csv(file_path):
    try:
        with open(file_path, mode='r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            # Iterate over the rows in the CSV file
            for idx, row in enumerate(csv_reader):
                if len(row) > 1 and idx > 1:
                    if float(row[-1]) > 0.85 and float(row[1]) > 0.8577:
                        best_makers.append(row[0])
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function
for results_folder in results_folder_path.iterdir():
    iterate_csv(results_folder.joinpath("results.csv").resolve())
print(set(best_makers))