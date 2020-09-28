import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import scipy.io.wavfile as wvf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout, LeakyReLU, Reshape, \
    Conv2DTranspose, Lambda


def define_discriminator(shape=(1024, 36, 1)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="tanh",
                     input_shape=shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="tanh"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def define_generator(latent_dimension):
    n_nodes = 128 * 256 * 9
    model = Sequential()
    model.add(Dense(n_nodes, input_dim=latent_dimension))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((9, 256, 128)))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (16, 8), activation='sigmoid', padding='same'))
    model.add(Reshape((1024, 36, 1)))
    return model


# latent_dim = 100
# model = define_generator(latent_dim)
# model.summary()


def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

latent_dim = 100
gen_model = define_generator(latent_dim)
disc_model = define_discriminator()
model = define_gan(gen_model, disc_model)
model.summary()


def load_real_samples():
    """loads all hard kick samples and returns melspectrogram data"""
    feature = []
    for i in range(1, 202):
        d, sr = librosa.load("sample_kick/VEH1 Hard Kick - " + str(i).zfill(3) + ".wav", sr=44100,
                             res_type='kaiser_fast')
        m = librosa.stft(d)
        z = np.zeros((1024, 36), dtype=float)
        for j, array in enumerate(m):
            for k, num in enumerate(array):
                try:
                    z[j][k] = num
                except IndexError:
                    pass
        feature.append(z)

    X = np.array(feature)
    return X


def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    X = X.reshape((12, 1024, 36, 1))
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y


def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples):
    X = g_model.predict(generate_latent_points(latent_dim, n_samples))
    y = np.zeros((n_samples, 1))
    return X, y


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=3, n_batch=24):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            print(X_real.shape, X_fake.shape, y_real.shape, y_fake.shape)

            # create training set for the discriminator
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            print(X_gan.shape, y_gan.shape)
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss, g_loss))
        # evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            # summarize_performance(i, g_model, d_model, dataset, latent_dim)
            print(str(i+1) + " epoch")


# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

fake_sample, _ = generate_fake_samples(g_model, latent_dim, 1)
# print(fake_sample, fake_sample.shape, fake_sample[0], fake_sample[0].shape)
fake_sample = librosa.istft(fake_sample.reshape((1024, 36)))
wvf.write("output_sampple.wav", 44100, fake_sample)


