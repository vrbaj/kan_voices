import pandas as pd

# Load the CSV file
file_path = "results_summary.csv"
data = pd.read_csv(file_path)

# Sort the DataFrame by the last column
sorted_data = data.sort_values(by=data.columns[-1], ascending=False)

# Take the first 10 rows
top_10_rows = sorted_data.head(10)

# Save the top 10 rows as a LaTeX table
latex_table = top_10_rows.to_latex(index=False)

# Save the LaTeX table to a .tex file
latex_file_path = "latex_table_summary.tex"
with open(latex_file_path, 'w') as file:
    file.write(latex_table)

print("LaTeX table saved successfully.")