import shutil
import uuid
import pickle
import json
import itertools
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import parselmouth
from parselmouth.praat import call
import numpy as np
import librosa
from tqdm import tqdm

@dataclass
class AudioFeaturesParams:
    f0_min: Optional[int] = 50
    f0_max: Optional[int] = 500
    age: Optional[bool] = True
    gender: Optional[bool] = True
    diff_pitch: Optional[bool] = True
    mean_f0: Optional[bool] = True
    stdev_f0: Optional[bool] = True
    hnr: Optional[bool] = True
    jitter: Optional[bool] = True
    shimmer: Optional[bool] = True
    mfcc: Optional[int] = 13
    delta_mfcc: Optional[bool] = True
    delta2_mfcc: Optional[bool] = True
    var_mfcc: Optional[bool] = False
    var_delta_mfcc: Optional[bool] = False
    var_delta2_mfcc: Optional[bool] = False
    spectral_centroid: Optional[bool] = True
    spectral_contrast: Optional[bool] = True
    spectral_flatness: Optional[bool] = True
    spectral_rolloff: Optional[bool] = True


def dataclass_to_json(dataclass_instance, file_path: Path):
    """Convert a dataclass instance to a JSON string."""
    dataclass_dict = asdict(dataclass_instance)
    json_str = json.dumps(dataclass_dict, indent=4)
    print(f"json file_path {file_path}")
    with file_path.open("w") as f:
        f.write(json_str)


def dump_to_pickle(data, file_path):
    try:
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
        print(f"Data successfully dumped to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_audio_features(voice_path: Path, params: AudioFeaturesParams) -> list:
    session_id = int(voice_path.name.split("-")[0])
    table = pd.read_csv(voice_path.parent.parent.joinpath("file_information.csv"))
    feature_list = [session_id]

    if params.age:
        age = table[table.sessionid == session_id]["talkerage"].values[0]
        feature_list.append(age)

    if params.gender:
        gender = table[table.sessionid==session_id]["talkersex"].values[0]
        if gender == "w":
            sex = 1
        else:
            sex = 0
        feature_list.append(sex)

    raw_data = parselmouth.Sound(str(voice_path)) # read raw sound data
    pitch = call(raw_data, "To Pitch", 0.0, params.f0_min, params.f0_max)
    point_process = call(raw_data, "To PointProcess (periodic, cc)", params.f0_min, params.f0_max)
    signal, sr = librosa.load(str(voice_path))

    if params.diff_pitch:
        pitch_data = pitch.selected_array['frequency']
        pitch_data[pitch_data == 0] = np.nan
        diff_pitch = (max(pitch_data) - min(pitch_data)) / min(pitch_data)
        feature_list.append(diff_pitch)

    if params.mean_f0:
        mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
        feature_list.append(mean_f0)

    if params.stdev_f0:
        stdev_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        feature_list.append(stdev_f0)



    if params.hnr:
        harmonicity = call(raw_data, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        feature_list.append(hnr)

    if params.jitter:
        local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        feature_list.append(local_jitter)

    if params.shimmer:
        local_shimmer = call([raw_data, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        feature_list.append(local_shimmer)

    if params.mfcc:
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=params.mfcc)
        mfcc = np.mean(mfccs, axis=1)
        feature_list = feature_list + list(mfcc)
        if params.var_mfcc:
            mfcc_var = np.var(mfccs, axis=1)
            feature_list = feature_list + list(mfcc_var)

        if params.delta_mfcc:
            delta_mfccs = librosa.feature.delta(mfccs)
            delta_mfcc = np.mean(delta_mfccs, axis=1)
            feature_list = feature_list + list(delta_mfcc)
            if params.var_delta_mfcc:
                delta_mfcc_var = np.var(delta_mfccs, axis=1)
                feature_list = feature_list + list(delta_mfcc_var)

        if params.delta2_mfcc:
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            delta2_mfcc = np.mean(delta2_mfccs, axis=1)
            feature_list = feature_list + list(delta2_mfcc)
            if params.var_delta2_mfcc:
                delta2_mfcc_var = np.var(delta2_mfccs, axis=1)
                feature_list = feature_list + list(delta2_mfcc_var)

    if params.spectral_centroid:
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr), axis=1)
        feature_list = feature_list + list(spectral_centroid)

    if params.spectral_contrast:
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=signal, sr=sr), axis=1)
        feature_list = feature_list + list(spectral_contrast)

    if params.spectral_flatness:
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=signal), axis=1)
        feature_list = feature_list + list(spectral_flatness)

    if params.spectral_rolloff:
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr), axis=1)
        feature_list = feature_list + list(spectral_rolloff)

    return feature_list

def load_svd(datasets_path: Path):
    labels = []
    file_paths = []
    dir_list = ["saarbruecken_m_n", "saarbruecken_m_p",
                "saarbruecken_w_n", "saarbruecken_w_p"]
    for directory in dir_list:
        files = list(datasets_path.joinpath(directory).glob("*.wav"))
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
    diff_pitch = [True, False]
    stdev_f0 = [True, False]
    spectral_centroid = [True, False]
    spectral_contrast = [True, False]
    spectral_flatness = [True, False]
    spectral_rolloff = [True, False]

    for diff, stdev, centroid, contrast, flatness, rolloff in itertools.product(diff_pitch, stdev_f0, spectral_centroid, spectral_contrast, spectral_flatness, spectral_rolloff):

        experiment_parameters = AudioFeaturesParams(
            f0_min=50,
            f0_max=500,
            age=True,
            gender=True,
            diff_pitch=diff,
            mean_f0=True,
            stdev_f0=stdev,
            hnr=True,
            jitter=True,
            shimmer=True,
            mfcc=13,
            delta_mfcc=True,
            delta2_mfcc=True,
            spectral_centroid=centroid,
            spectral_contrast=contrast,
            spectral_flatness=flatness,
            spectral_rolloff=rolloff
        )

        datasets_path = Path(".").joinpath("trimmed_files")
        file_paths, labels = load_svd(datasets_path)
        idx_dataset = []
        X = []
        uuid_dir = uuid.uuid4()
        dataset_path = Path(".").joinpath("training_data", str(uuid_dir))
        dataset_path.mkdir(parents=True)

        dataset_file = dataset_path.joinpath(f"dataset.pk")
        set_maker = dataset_path.joinpath(f"set_maker.py")
        shutil.copy(__file__, set_maker)
        dataset_to_dump = {"index": idx_dataset,
                         "data": X,
                         "labels": labels}

        indices_to_remove = []
        discard_time = 0
        for idx, patient in enumerate(tqdm(file_paths, desc="Processing the train set...")):
            features = get_audio_features(patient, experiment_parameters)
            features.append(1 if np.isnan(features).any() else 0)
            features = np.nan_to_num(np.array(features), copy=True, nan=0)
            if features[1] > 16:

                idx_dataset.append(features[0])
                X.append(features[1:])
            else:
                indices_to_remove.append(idx)

        dataset_to_dump["data"] = MinMaxScaler().fit_transform(np.array(dataset_to_dump["data"]))
        print(dataset_to_dump["data"].shape)
        dataset_to_dump["labels"] = remove_items_by_indices(dataset_to_dump["labels"], indices_to_remove)
        dump_to_pickle(dataset_to_dump, dataset_file)
        dataclass_to_json(experiment_parameters, dataset_path.joinpath("config.json"))