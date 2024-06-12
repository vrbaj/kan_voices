from pathlib import Path
import soundfile as sf
import numpy as np

# Input directory containing the .wav files
input_dir = Path('trimmed_files')

# Output file for the joined audio
output_file = Path('joined_audio.wav')

# List to store audio data and sampling rates
audio_data = []
sampling_rates = []

# Iterate over all .wav files in the input directory
for file_path in input_dir.glob('*.wav'):
    # Load each .wav file and store audio data and sampling rate
    data, sr = sf.read(file_path)
    audio_data.append(data)
    sampling_rates.append(sr)

# Concatenate audio data
joined_audio = np.concatenate(audio_data)

# Get the minimum sampling rate
min_sr = min(sampling_rates)

# Write the joined audio to the output file with the minimum sampling rate
sf.write(output_file, joined_audio, min_sr)

print(f"Joined audio saved to '{output_file}'")