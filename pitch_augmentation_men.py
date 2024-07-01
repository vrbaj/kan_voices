import random
import shutil
from pathlib import Path
import librosa
import soundfile as sf
import tqdm


if __name__ == "__main__":
    fld = 9
    dataset_path = Path(".", "folds_dataset", f"fold_{fld}")
    augmented_dataset_path = Path(".", "folds_dataset", f"fold_{fld}")
    folders_list = ["healthy",
                   "unhealty"]

    healthy_men_paths = list(dataset_path.joinpath("healthy").glob("*.wav"))
    healthy_men_total = len(healthy_men_paths)

    unhealty_men_paths = list(dataset_path.joinpath("unhealthy").glob("*.wav"))
    unhealthy_men_total = len(unhealty_men_paths)


    print(f"healthy men: {healthy_men_total} unhealthy men: {unhealthy_men_total}")
    # select randomly 185 samples and pitch them up
    healthy_to_up = random.sample(healthy_men_paths, 18)
    # select randomly another 185 samples and pitch them down
    healthy_to_down = random.sample(healthy_men_paths, 19)

    print("pitching up")
    for sample in tqdm.tqdm(healthy_to_up):
        y, sr = librosa.load(str(sample), sr=None)

        y_pitched = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
        splitted_name = sample.name.split("_")
        new_name = splitted_name[0] + "u_" + splitted_name[1] + "_" + splitted_name[2]
        path_to_save = augmented_dataset_path.joinpath("healthy", new_name)
        sf.write(path_to_save, y_pitched, sr)

    print("pitching down")
    for sample in tqdm.tqdm(healthy_to_down):
        y, sr = librosa.load(str(sample), sr=None)
        y_pitched = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)
        splitted_name = sample.name.split("_")
        new_name = splitted_name[0] + "d_" + splitted_name[1] + "_" + splitted_name[2]
        path_to_save = augmented_dataset_path.joinpath("healthy", new_name)
        sf.write(path_to_save, y_pitched, sr)
