from pathlib import Path

import pandas as pd


def compute_stats(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Exclude the first column
    df_excluded = df.iloc[:, 1:]

    # Compute mean and standard deviation for each column
    means = df_excluded.mean()
    std_devs = df_excluded.std()
    results = {"name": file_path.parents[0].name}
    # Print the results
    for col in df_excluded.columns:
        results[f"{col}_mean"] = [means[col]]
        results[f"{col}_std"] = [std_devs[col]]
    return results


if __name__ == "__main__":
    path_to_results = Path(".", "results")
    result_summary = Path("results_summary.csv")
    do_header = True
    for result_dir in path_to_results.iterdir():
        result_file = result_dir.joinpath("results.csv")
        exp_stats = compute_stats(result_file)

        pd.DataFrame(exp_stats).to_csv(result_summary, header=do_header, index=False, mode="a")
        do_header = False
