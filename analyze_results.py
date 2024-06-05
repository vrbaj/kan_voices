import csv

# File path for the CSV file
file_path = 'results_var17+.csv'

# Function to iterate over the CSV file
def iterate_csv(file_path):
    try:
        with open(file_path, mode='r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            # Iterate over the rows in the CSV file
            for idx, row in enumerate(csv_reader):
                if len(row) > 1 and idx > 1:
                    if float(row[-1]) > 0.90 and float(row[-2]) > 0.91:
                        print(row)
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function
iterate_csv(file_path)