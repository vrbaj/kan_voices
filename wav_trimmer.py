from pathlib import Path
import librosa
import soundfile as sf

# Path to dataset
input_path = Path(".").joinpath("datasets")

# Output directory for the trimmed files
output_dir = Path(".").joinpath("trimmed_files")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

def trim_silence(input_path, output_path):
    # Load the audio file
    y, sr = librosa.load(input_path, sr=None)
    original_length = librosa.get_duration(y=y, sr=sr)
    # Trim silence from the beginning and end
    yt, index = librosa.effects.trim(y, top_db=15, frame_length=256, hop_length=65)
    output_path.parent.mkdir(exist_ok=True)
    # Save the trimmed audio file
    trimmed_length = librosa.get_duration(y=yt, sr=sr)
    print(f" Diff: {trimmed_length - original_length} {input_path}")
    sf.write(output_path, yt, sr)

# Iterate over all directories and process .wav files
for input_dir in input_path.iterdir():
    for file in input_dir.rglob('*.wav'):
        output_path = output_dir.joinpath(file.parent.parent.name).joinpath(file.stem + '_trim.wav')
        trim_silence(file, output_path)
        # print(f'Trimmed and saved: {output_path}')
