from pathlib import Path
import pandas as pd
import json

min_acc = 0.8  # 0.8577
min_recall = 0.83  # 0.8759
min_spec = 0.83  # 0.8394
results_dir = Path("results")
temp_df_list = []
for path in results_dir.glob("**/results.csv"):
    temp_df_list.append(pd.read_csv(path))
    temp_df_list[-1]["data"] = path.parent.name

table_results = pd.concat(temp_df_list)
table_results = table_results[(table_results.mean_test_accuracy >= min_acc) &
                              (table_results.mean_test_recall >= min_recall) &
                              (table_results.mean_test_specificity >= min_spec)].sort_values("mean_test_accuracy")

for _, row in table_results.iterrows():
    str_params = ""
    for key, val in json.loads(row["params"].replace("'", '"')).items():
        if key == "classifier__C":
            str_params += f"{key.split('__')[1]}: {val:>5} - "
        elif key == "classifier__kernel":
            str_params += f"{key.split('__')[1]}: {val:>4} - "
        else:
            str_params += f"{key.split('__')[1]}: {val} - "

    print(
        f'{row["data"]} | {str_params[:-2]}| Acc: {row["mean_test_accuracy"]:.5f} - Sen: {row["mean_test_recall"]:.5f}'
        f' - Spe: {row["mean_test_specificity"]:.5f}')