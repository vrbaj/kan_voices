from pyAudioAnalysis import audioTrainTest as aT
import matplotlib.pyplot as plt
import numpy as np
import wave

# Load audio file
print(aT.extract_features_and_train(["pyaudio_data/p","pyaudio_data/n"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False))
