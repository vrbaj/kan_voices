import math
from multiprocessing import Process, Pool, Lock
import itertools
import pickle
import shutil
import uuid
from pathlib import Path
import numpy as np

from tqdm.contrib.concurrent import process_map  # or thread_map
from sklearn.preprocessing import MinMaxScaler
from dataset_generator import AudioFeaturesParams, load_svd, get_audio_features
from dataset_generator import remove_items_by_indices, dump_to_pickle, dataclass_to_json
def create_dataset(dataset_params: AudioFeaturesParams):
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
    for idx, patient in enumerate(file_paths):
        features = get_audio_features(patient, dataset_params, SEX_OF_INTEREST)
        features.append(1 if np.isnan(features).any() else 0)
        features = np.nan_to_num(np.array(features), copy=True, nan=0)
        if features[2] > 20 and features[1] == SEX_OF_INTEREST:

            idx_dataset.append(features[0])
            X.append(features[2:])
        else:
            indices_to_remove.append(idx)

    dataset_to_dump["data"] = MinMaxScaler().fit_transform(np.array(dataset_to_dump["data"]))
    dataset_to_dump["labels"] = remove_items_by_indices(dataset_to_dump["labels"], indices_to_remove)
    dump_to_pickle(dataset_to_dump, dataset_file)
    dataclass_to_json(dataset_params, dataset_path.joinpath("config.json"))

SEX_OF_INTEREST = 0
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

    configurations = []
    for diff, stdev, centroid, contrast, flatness, rollof, zrc_param, mfcc, var_mfcc, \
            formant, lfcc, skew, shannon in itertools.product(diff_pitch, stdev_f0,
                                                                               spectral_centroid,
                                                                               spectral_contrast,
                                                                               spectral_flatness,
                                                                               spectral_rolloff,
                                                                               zcr, mfccs, var_mfccs,
                                                                               formants, lfccs,
                                                                               skewness, shannon_entropy):
        experiment_params = AudioFeaturesParams(
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
            mfcc=mfcc,
            delta_mfcc=True,
            delta2_mfcc=True,
            var_mfcc=var_mfcc,
            var_delta_mfcc=False,
            var_delta2_mfcc=False,
            spectral_centroid=centroid,
            spectral_contrast=contrast,
            spectral_flatness=flatness,
            spectral_rolloff=rollof,
            zero_crossing_rate=zrc_param,
            shannon=shannon,
            lfcc=lfcc,
            skew=skew
        )
        configurations.append(experiment_params)
    print(f"Totally {len(configurations)} datasets will be created")
    chunk_size = 900
    chunks = math.ceil(len(configurations) / chunk_size)
    processed = 0
    while configurations:
        print(f"chunks {processed} / {chunks - 1}")
        chunk, configurations = configurations[:chunk_size], configurations[chunk_size:]
        r = process_map(create_dataset, chunk, max_workers=11)
        processed += 1