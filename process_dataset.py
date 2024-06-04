import parselmouth
from parselmouth.praat import call
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm


def dump_to_json(data, file_path):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
        print(f"Data successfully dumped to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_features(voice_path, f0_min, f0_max, unit):
    raw_data = parselmouth.Sound(str(voice_path))  # read raw sound data
    pitch = call(raw_data, "To Pitch", 0.0, f0_min, f0_max)
    mean_f0 = call(pitch, "Get mean", 0, 0, unit)
    stdev_f0 = call(pitch, "Get standard deviation", 0, 0, unit)
    harmonicity = call(raw_data, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    point_process = call(raw_data, "To PointProcess (periodic, cc)", f0_min, f0_max)
    hnr = call(harmonicity, "Get mean", 0, 0)
    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    local_shimmer = call([raw_data, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return [mean_f0, stdev_f0, hnr, local_jitter, local_shimmer]

def load_svd(datasets_path: Path):
    labels = []
    file_paths = []
    dir_list = ["saarbruecken_m_n", "saarbruecken_m_p",
                "saarbruecken_w_n", "saarbruecken_m_p"]
    for directory in dir_list:
        files = list(datasets_path.joinpath(directory, "export").glob("*.wav"))
        file_paths += files
        if "_n" in directory:
            labels += len(files) * [0]
        else:
            labels += len(files) * [1]
    return file_paths, labels

if __name__ == "__main__":
    datasets_path = Path(__file__).parent.joinpath("datasets")
    file_paths, labels = load_svd(datasets_path)
    patients_train, patients_test, labels_train, labels_test = train_test_split(file_paths,
                                                                                labels,
                                                                                test_size=0.2,
                                                                                random_state=42)
    x_train = []
    x_test = []

    train_file = "train_set.json"
    test_file = "test_set.json"
    train_to_dump = {"data": x_train,
                     "labels": labels_train}
    test_to_dump = {"data": x_test,
                    "labels": labels_test}

    for patient in tqdm(patients_train, desc="Processing the train set..."):
        x_train.append(get_features(patient, 75, 500, "Hertz"))
    dump_to_json(train_to_dump, train_file)

    for patient in tqdm(patients_test, desc="Processing the test set..."):
        x_test.append(get_features(patient, 75, 500, "Hertz"))
    dump_to_json(test_to_dump, test_file)

