import parselmouth
import pandas
import librosa
from parselmouth.praat import call
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
from tqdm import tqdm


def dump_to_json(data, file_path):
    try:
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
        print(f"Data successfully dumped to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_features(voice_path, f0_min, f0_max, unit, discard_samples=0):
    session_id = int(voice_path.name.split("-")[0])
    table = pandas.read_csv(voice_path.parent.parent.parent.joinpath("file_information.csv"))
    age = table[table.sessionid==session_id]["talkerage"].values[0]
    gender = table[table.sessionid==session_id]["talkersex"].values[0]
    if gender == "w":
        sex = 1
    else:
        sex = 0
    raw_data = parselmouth.Sound(str(voice_path)) # read raw sound data
    total_duration = raw_data.get_total_duration()
    raw_data = raw_data.extract_part(from_time=discard_samples, to_time=total_duration-0.1, preserve_times=True)
    pitch = call(raw_data, "To Pitch", 0.0, f0_min, f0_max)
    mean_f0 = call(pitch, "Get mean", 0, 0, unit)
    stdev_f0 = call(pitch, "Get standard deviation", 0, 0, unit)
    harmonicity = call(raw_data, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    point_process = call(raw_data, "To PointProcess (periodic, cc)", f0_min, f0_max)
    hnr = call(harmonicity, "Get mean", 0, 0)
    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    local_shimmer = call([raw_data, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    # mfcc_data = raw_data.to_mfcc(number_of_coefficients=13).to_array()
    # mfcc = np.mean(mfcc_data, axis=1)
    signal, sr = librosa.load(str(voice_path))
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
    mfcc = np.mean(mfccs, axis=1)
    mfcc_var = np.var(mfccs, axis=1)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mfcc = np.mean(delta_mfccs, axis=1)
    delta_mfcc_var = np.var(delta_mfccs, axis=1)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    delta2_mfcc = np.mean(delta2_mfccs, axis=1)
    return ([age, sex, mean_f0, stdev_f0, hnr, local_jitter, local_shimmer ] +
            list(mfcc) + list(delta_mfcc) + list(delta2_mfcc) + list(mfcc_var))

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


def remove_items_by_indices(lst, indices):
    # Sort the indices in descending order
    indices = sorted(indices, reverse=True)
    # Remove items from the list using the sorted indices
    for index in indices:
        if 0 <= index < len(lst):
            lst.pop(index)
    return lst


if __name__ == "__main__":
    datasets_path = Path(__file__).parent.joinpath("datasets")
    file_paths, labels = load_svd(datasets_path)
    patients_train, patients_test, labels_train, labels_test = train_test_split(file_paths,
                                                                                labels,
                                                                                test_size=0.2,
                                                                                random_state=42)
    x_train = []
    x_test = []

    train_file = "train_set.pk"
    test_file = "test_set.pk"
    train_to_dump = {"data": x_train,
                     "labels": labels_train}
    test_to_dump = {"data": x_test,
                    "labels": labels_test}
    indices_to_remove = []
    discard_time = 0.3
    for idx, patient in enumerate(tqdm(patients_train, desc="Processing the train set...")):
        features = get_features(patient, 50, 500, "Hertz", discard_samples=discard_time)
        if not np.isnan(np.array(features)).any() and features[0] > 16:
            x_train.append(features)
        else:
            indices_to_remove.append(idx)
    train_to_dump["data"] = MinMaxScaler().fit_transform(np.array(train_to_dump["data"]))
    print(train_to_dump["data"].shape)
    train_to_dump["labels"] = remove_items_by_indices(train_to_dump["labels"], indices_to_remove)
    dump_to_json(train_to_dump, train_file)

    indices_to_remove = []
    for idx, patient in enumerate(tqdm(patients_test, desc="Processing the test set...")):
        features = get_features(patient, 50, 500, "Hertz", discard_samples=discard_time)
        if not np.isnan(np.array(features)).any() and features[0] > 16:
            x_test.append(features)
        else:
            indices_to_remove.append(idx)
    test_to_dump["data"] = MinMaxScaler().fit_transform(np.array(test_to_dump["data"]))
    test_to_dump["labels"] = remove_items_by_indices(test_to_dump["labels"], indices_to_remove)
    dump_to_json(test_to_dump, test_file)

