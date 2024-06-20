from pathlib import Path
import csv

import pandas as pd
from tqdm import tqdm
import parselmouth
from parselmouth.praat import call
import librosa
import numpy as np
import spkit as sp
import formantfeatures as ff
import torch
import torchaudio
from torchaudio import transforms as T
from scipy.stats import skew

def extract_features(voice_path):
    session_id = int(voice_path.name.split("-")[0])
    table = pd.read_csv(voice_path.parent.parent.joinpath("file_information.csv"))
    features = {}
    features["session_id"] = session_id

    # gender of the patient
    gender = table[table.sessionid == session_id]["talkersex"].values[0]
    if gender == "w":
        sex = 1
    else:
        sex = 0
    features["gender"] = sex


    # age of the patient
    age = table[table.sessionid == session_id]["talkerage"].values[0]
    features["age"] = age

    raw_data = parselmouth.Sound(str(voice_path))  # read raw sound data
    pitch = call(raw_data, "To Pitch", 0.0, 50, 500)
    point_process = call(raw_data, "To PointProcess (periodic, cc)", 50, 500)
    signal, sr = librosa.load(str(voice_path))

    pitch_data = pitch.selected_array['frequency']
    pitch_data[pitch_data == 0] = np.nan
    diff_pitch = (max(pitch_data) - min(pitch_data)) / min(pitch_data)
    features["diff_pitch"] = diff_pitch

    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    features["mean_f0"] = mean_f0

    stdev_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    features["stdev_f0"] = stdev_f0

    harmonicity = call(raw_data, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    features["hnr"] = hnr

    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    features["jitter"] = local_jitter

    local_shimmer = call([raw_data, point_process],
                         "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    features["shimmer"] = local_shimmer

    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=30)
    mfcc = np.mean(mfccs, axis=1)
    features["mfcc"] = mfcc.tolist()

    mfcc_var = np.var(mfccs, axis=1)
    features["var_mfcc"] = mfcc_var.tolist()

    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mfcc = np.mean(delta_mfccs, axis=1)
    features["delta_mfcc"] = delta_mfcc.tolist()

    delta_mfcc_var = np.var(delta_mfccs, axis=1)
    features["var_delta_mfcc"] = delta_mfcc_var.tolist()

    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    delta2_mfcc = np.mean(delta2_mfccs, axis=1)
    features["delta2_mfcc"] = delta2_mfcc.tolist()

    delta2_mfcc_var = np.var(delta2_mfccs, axis=1)
    features["var_delta2_mfcc"] = delta2_mfcc_var.tolist()

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr), axis=1)
    features["spectral_centroid"] = spectral_centroid[0]

    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=signal, sr=sr), axis=1)
    features["spectral_contrast"] = spectral_contrast.tolist()

    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=signal), axis=1)
    features["spectral_flatness"] = spectral_flatness[0]

    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr), axis=1)
    features["spectral_folloff"] = spectral_rolloff[0]

    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y=signal)[0])
    features["zero_crossing_rate"] = zero_crossing

    window_step = 0.010
    window_length = 0.025
    emphasize_ratio = 0.65
    formants_f0_min = 30
    formants_f0_max = 4000
    max_frames = 500
    max_formants = 3
    formants_features, frame_count, _, _ = ff.Extract_wav_file_formants(str(voice_path), window_length,
                                                                                                 window_step,
                                                                                                 emphasize_ratio,
                                                                                                 norm=0,
                                                                                                 f0_min=formants_f0_min,
                                                                                                 f0_max=formants_f0_max,
                                                                                                 max_frames=max_frames,
                                                                                                 formants=max_formants)

    formants_list = []
    for formant in range(max_formants):
        formants_list.append(np.mean(formants_features[0:frame_count, (formant * 4) + 0]))
    features["formants"] = formants_list

    shannon_entropy = sp.entropy(signal, alpha=1)
    features["shannon_entropy"] = shannon_entropy


    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(voice_path)
    lfcc_transform = T.LFCC(
        sample_rate=SAMPLE_RATE,
        n_lfcc=20,
        speckwargs={
            "n_fft": 2048,
            "win_length": None,
            "hop_length": 512,
        },
    )
    lfccs = lfcc_transform(SPEECH_WAVEFORM)
    lfcc = torch.mean(lfccs, dim=2)[0]
    features["lfcc"] = [tensor.item() for tensor in lfcc]

    features["skewness"] = skew(signal)
    return features

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


wav_data_path = Path(".", "trimmed_files")


file_paths, labels = load_svd(wav_data_path)
data_to_dump = []
for idx, patient in enumerate(tqdm(file_paths,  desc="Extracting features...")):
    features = extract_features(patient)
    data_to_dump.append(features)

with open("features.csv", "w", encoding="utf8", newline="") as output_file:
    fc = csv.DictWriter(output_file,
                        fieldnames=data_to_dump[0].keys())
    fc.writeheader()
    fc.writerows(data_to_dump)
