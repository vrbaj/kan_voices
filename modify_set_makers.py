from pathlib import Path
import subprocess
from multiprocessing import Process, Pool, Lock

def load_and_execute(file_path):
    print(f"Executing {file_path.resolve()}")
    proc = subprocess.Popen(f"C:\\Users\\extis\\PycharmProjects\\kan_voices\\.venv\\Scripts\\python.exe {file_path.resolve()}", bufsize=1,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # with open(file_path, "r") as file:
    #     exec(file.read())
    #     print("done")
    (output, err) = proc.communicate()

    # This makes the wait possible
    p_status = proc.wait()
    print(p_status)
set_maker_folder = Path(".").joinpath("evaluated_datasets")
dst_folder = Path(".")
file_paths = []
for idx, set_maker in enumerate(set_maker_folder.iterdir()):
    text = set_maker.joinpath("set_maker.py").read_text()
    # new_file = text.replace("not np.isnan(np.array(features)).any() and", "")
    # new_file = text.replace("discard_samples=discard_time)\n", "discard_samples=discard_time)\n        features = np.nan_to_num(np.array(features), copy=True, nan=0)\n")
    # new_file = text.replace("features = np.nan_to_num(np.array(features), copy=True, nan=0)", "features.append(1 if np.isnan(features).any() else 0)\n        features = np.nan_to_num(np.array(features), copy=True, nan=0)\n")
    # new_file = text.replace("MinMaxScaler", "StandardScaler")
    new_file = text.replace("StandardScaler", "MinMaxScaler")
    with open(dst_folder.joinpath(f"setmaker{idx}.py"), "w") as file:
        file.write(new_file)
    file_paths.append(dst_folder.joinpath(f"setmaker{idx}.py"))

if __name__ == "__main__":
    with Pool(16) as p:
        p.map(load_and_execute, file_paths)



