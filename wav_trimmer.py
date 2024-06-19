"""
Script that removes silence from the Saarbruecken Voice Database recordings.
"""
from pathlib import Path
import librosa
import soundfile as sf


def trim_silence(input_path, output_path):
    """
    Function that removes the silence from the WAV file specified by input_path.
    :param input_path: path to file that will be trimmed.
    :param output_path: path to output file, where the trimmed file will be written.
    :return: None
    """
    # Load the audio file
    y, sr = librosa.load(input_path, sr=None)
    # Get the recording length
    original_length = librosa.get_duration(y=y, sr=sr)
    # Trim silence from the beginning and end
    yt, _ = librosa.effects.trim(y, top_db=15, frame_length=256, hop_length=65)
    # Create the directory to keep consistence with the original dataset
    output_path.parent.mkdir(exist_ok=True)
    # Get the length of trimmed recording
    trimmed_length = librosa.get_duration(y=yt, sr=sr)
    # Print information about the total trimmed time
    print(f" Diff: {trimmed_length - original_length} {input_path}")
    # Save the trimmed audio file
    sf.write(output_path, yt, sr)

if __name__ == "__main__":
    # Path to dataset
    dataset_path = Path(".").joinpath("datasets")
    # Output directory for the trimmed files
    output_dir = Path(".").joinpath("trimmed_files")
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Iterate over all directories and process .wav files
    for input_dir in dataset_path.iterdir():
        for file in input_dir.rglob('*.wav'):
            trimmed_file_path = output_dir.joinpath(file.parent.parent.name).joinpath(file.stem +
                                                                                '_trim.wav')
            trim_silence(file, trimmed_file_path)
            # print(f'Trimmed and saved: {output_path}')
