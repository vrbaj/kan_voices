from pathlib import Path
import librosa
import soundfile as sf

path_to_wav = Path(".", "datasets", "saarbruecken_m_n", "export", "32-a_n.wav")
y, sr = librosa.load(str(path_to_wav), sr=None)
y_third = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
sf.write("pitch_1up.wav", y_third, sr)
