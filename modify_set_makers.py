from pathlib import Path

set_maker_folder = Path(".").joinpath("training_data")
dst_folder = Path(".")
for idx, set_maker in enumerate(set_maker_folder.iterdir()):
    text = set_maker.joinpath("set_maker.py").read_text()
    # new_file = text.replace("not np.isnan(np.array(features)).any() and", "")
    # new_file = text.replace("discard_samples=discard_time)\n", "discard_samples=discard_time)\n        features = np.nan_to_num(np.array(features), copy=True, nan=0)\n")
    new_file = text.replace("features = np.nan_to_num(np.array(features), copy=True, nan=0)", "features.append(1 if np.isnan(features).any() else 0)\n        features = np.nan_to_num(np.array(features), copy=True, nan=0)\n")
    with open(dst_folder.joinpath(f"setmaker{idx}.py"), "w") as file:
        file.write(new_file)

