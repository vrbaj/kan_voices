"""
Script to generate all evaluated datasets, with various features combinations.
"""
import csv
from pathlib import Path
import itertools
import pickle
import uuid
import math
import json

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm.contrib.concurrent import process_map  # or thread_map


def dataset_config_to_json(experiment_config, file_path: Path):
    """Convert a dataclass instance to a JSON string."""
    with file_path.open("w") as f:
        json.dump(experiment_config, f)


def dump_to_pickle(data, file_path):
    try:
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
        #print(f"Data successfully dumped to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def compose_dataset(dataset_params: dict) -> None:
    X = []
    y = []
    data_to_dump = {"data": X, "labels": y}
    with open("features.csv", newline="") as csv_file:
        dataset = csv.DictReader(csv_file)
        patient: dict

        for idx, patient in enumerate(dataset):
            if idx > 0:
                patient_features = []
                nan_in_data = False
                if int(patient["age"]) > dataset_params["min_age"] and int(patient["gender"]) == dataset_params["gender"]:
                    y.append(int(patient["diagnosis"]))
                    patient_features.append(int(patient["age"]))
                    if dataset_params["diff_pitch"]:
                        if patient["diff_pitch"] == "nan":
                            patient_features.append(0.0)
                            nan_in_data = True
                        else:
                            patient_features.append(float(patient["diff_pitch"]))

                    if patient["mean_f0"] == "nan":
                        patient_features.append(0.0)
                        nan_in_data = True
                    else:
                        patient_features.append(float(patient["mean_f0"]))

                    if dataset_params["stdev_f0"]:
                        if patient["stdev_f0"] == "nan":
                            patient_features.append(0.0)
                            nan_in_data = True
                        else:
                            patient_features.append(float(patient["stdev_f0"]))

                    patient_features.append(float(patient["hnr"]))

                    if patient["jitter"] == "nan":
                        patient_features.append(0.0)
                        nan_in_data = True
                    else:
                        patient_features.append(float(patient["jitter"]))

                    if patient["shimmer"] == "nan":
                        patient_features.append(0.0)
                        nan_in_data = True
                    else:
                        patient_features.append(float(patient["shimmer"]))

                    all_mfcc = eval(patient["mfcc"])
                    patient_features += all_mfcc[:dataset_params["mfcc"]]

                    all_delta_mfcc = eval(patient["delta_mfcc"])
                    patient_features += all_delta_mfcc[:dataset_params["mfcc"]]

                    all_delta2_mfcc = eval(patient["delta2_mfcc"])
                    patient_features += all_delta2_mfcc[:dataset_params["mfcc"]]

                    if dataset_params["var_mfcc"]:
                        all_var_mfcc = eval(patient["var_mfcc"])
                        patient_features += all_var_mfcc[:dataset_params["mfcc"]]

                        all_var_delta_mfcc = eval(patient["var_delta_mfcc"])
                        patient_features += all_var_delta_mfcc[:dataset_params["mfcc"]]

                        all_var_delta2_mfcc = eval(patient["var_delta2_mfcc"])
                        patient_features += all_var_delta2_mfcc[:dataset_params["mfcc"]]

                    if dataset_params["spectral_centroid"]:
                        patient_features.append(float(patient["spectral_centroid"]))

                    if dataset_params["spectral_contrast"]:
                        all_contrasts = eval(patient["spectral_contrast"])
                        patient_features += all_contrasts

                    if dataset_params["spectral_flatness"]:
                        patient_features.append(float(patient["spectral_flatness"]))

                    if dataset_params["spectral_rolloff"]:
                        patient_features.append(float(patient["spectral_rolloff"]))

                    if dataset_params["zero_crossing_rate"]:
                        patient_features.append(float(patient["zero_crossing_rate"]))

                    if dataset_params["formants"]:
                        all_formants = eval(patient["formants"])
                        patient_features += all_formants

                    if dataset_params["shannon_entropy"]:
                        patient_features.append(float(patient["shannon_entropy"]))

                    if dataset_params["lfcc"]:
                        all_lfcc = eval(patient["lfcc"])
                        patient_features += all_lfcc

                    if dataset_params["skewness"]:
                        patient_features.append(float(patient["skewness"]))

                    if nan_in_data:
                        patient_features.append(1)
                    else:
                        patient_features.append(0)

                    X.append(patient_features)

    data_to_dump["data"] = MinMaxScaler().fit_transform(np.array(data_to_dump["data"]))
    uuid_dir = uuid.uuid4()
    dataset_path = Path(".").joinpath("training_data", str(uuid_dir))
    dataset_path.mkdir(parents=True)
    dataset_file = dataset_path.joinpath(f"dataset.pk")
    dump_to_pickle(data_to_dump, dataset_file)
    dataset_config_to_json(dataset_params, dataset_path.joinpath("config.json"))

if __name__ == "__main__":
    diff_pitch = [True, False]
    stdev_f0 = [True, False]
    spectral_centroid = [True, False]
    spectral_contrast = [True, False]
    spectral_flatness = [True, False]
    spectral_rolloff = [True, False]
    zcr = [True, False]
    mfccs = [13, 20]
    var_mfccs = [True, False]
    formants = [True, False]
    lfccs = [True, False]
    skewness = [True, False]
    shannon_entropy = [True, False]

    sex_of_interest = 0
    age_of_interest = 20
    configurations = []
    for diff, stdev, centroid, contrast, flatness, rolloff, zrc_param, mfcc, var_mfcc, \
            formant, lfcc, skew, shannon in itertools.product(diff_pitch, stdev_f0,
                                                              spectral_centroid,
                                                              spectral_contrast,
                                                              spectral_flatness,
                                                              spectral_rolloff,
                                                              zcr, mfccs, var_mfccs,
                                                              formants, lfccs,
                                                              skewness, shannon_entropy):
        dataset_config = {"gender": sex_of_interest,
                          "min_age": age_of_interest,
                          "diff_pitch": diff,
                          "stdev_f0": stdev,
                          "mfcc": mfcc,
                          "var_mfcc": var_mfcc,
                          "spectral_centroid": centroid,
                          "spectral_contrast": contrast,
                          "spectral_flatness": flatness,
                          "spectral_rolloff": rolloff,
                          "zero_crossing_rate": zrc_param,
                          "formants": formant,
                          "lfcc": lfcc,
                          "skewness": skew,
                          "shannon_entropy": shannon}
        configurations.append(dataset_config)

    print(f"Totally {len(configurations)} datasets will be created")
    chunk_size = 900
    chunks = math.ceil(len(configurations) / chunk_size)
    processed = 0
    while configurations:
        print(f"chunks {processed} / {chunks - 1}")
        chunk, configurations = configurations[:chunk_size], configurations[chunk_size:]
        r = process_map(compose_dataset, chunk, max_workers=1)
        processed += 1