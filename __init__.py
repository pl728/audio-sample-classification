import librosa
import numpy as np
import scipy.io.wavfile
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout

d, sr = librosa.load("sample_kick/VEH1 Hard Kick - 002.wav", sr=44100,
                     res_type='kaiser_fast')
mel = librosa.stft(d)
mels = np.mean(librosa.feature.melspectrogram(y=d, sr=sr).T, axis=0)
print(mel.shape)
wav = librosa.istft(mel)
print(wav)

fake_data = np.zeros((1024, 36))
print(fake_data.shape)
wav = librosa.istft(fake_data)
print(wav.shape)

print(wav)

