import librosa
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("kick-crash-classifier")
d, sr = librosa.load("VEH3 Crash 14.wav", sr=44100,
                     res_type='kaiser_fast')
mel = np.mean(librosa.feature.melspectrogram(y=d, sr=sr).T, axis=0)
mel = mel.reshape((1, 16, 8, 1))

print(model.predict(mel))
