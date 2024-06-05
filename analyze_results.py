import csv

# File path for the CSV file
file_path = 'results.csv'

# Function to iterate over the CSV file
def iterate_csv(file_path):
    try:
        with open(file_path, mode='r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            # Iterate over the rows in the CSV file
            for row in csv_reader:
                if len(row) > 1:
                    if float(row[-1]) > 0.87 and float(row[-2]) > 0.87:
                        print(row)
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function
iterate_csv(file_path)