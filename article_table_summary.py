import pandas as pd
import numpy as np

# Load the CSV file
file_path = "results_summary.csv"
data = pd.read_csv(file_path)

# Compute the new standard deviation column
data["uar stdev"] = np.sqrt(data.iloc[:, 4]**2 + data.iloc[:, 6]**2)

# Sort the DataFrame by the last column in descending order
sorted_data = data.sort_values(by=data.columns[-2], ascending=False)

# Take the first 10 rows
top_10_rows = sorted_data.head(10)

# Format the numeric values as percentages with two decimal points
formatted_top_10_rows = top_10_rows.applymap(lambda x: f"{x * 100:.2f}%" if isinstance(x, (int, float)) else x)

# Save the top 10 rows as a LaTeX table
latex_table = formatted_top_10_rows.to_latex(index=False)

# Save the LaTeX table to a .tex file
latex_file_path = "latex_table.tex"
with open(latex_file_path, 'w') as file:
    file.write(latex_table)

print("LaTeX table saved successfully.")