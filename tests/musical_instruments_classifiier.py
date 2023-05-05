import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd

from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import librosa
import librosa.feature
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
from keras import optimizers
from keras.callbacks import EarlyStopping
import random

DATA_PATH = "/content/instr_data/"

# this function returns the folder name as label for given data path
# also returns label indices and one hot coded variables
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

# this function will return mfcc coefficients
def wav2mfcc(file_path, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    # downsampling
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    # if maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # else cutoff the remaning parts
    else:
        mfcc = mfcc[:, :max_len]
    # print(mfcc.shape)
    return mfcc

def get_feature(path=DATA_PATH, max_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        # init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            if wavfile.endswith('wav'):
                mfcc = wav2mfcc(wavfile, max_len=max_len)
                mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)

def get_train_test(split_ratio=0.8, random_state=42):
    # get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)

# second dimension of the feature is dim2
feature_dim_2 = 64

# save data to array file first
get_feature(max_len=feature_dim_2)

# loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# feature dimension
feature_dim_1 = 20

# reshaping to perform 2D convolution
X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

# model related variables
channel = 1
epochs = 100
batch_size = 100
verbose = 1
num_classes = 4

def get_model():
    model = Sequential()
    # layer 1
    model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2)))

    # layer 2
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.54))

    # layer 3
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # flatten
    model.add(Flatten())
    model.add(Dropout(0.4))

    # full layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    # output layer
    model.add(Dense(num_classes, activation='softmax'))
    Ada = optimizers.SGD(lr=0.01)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Ada, metrics=['accuracy'])

    return model

model = get_model()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
hist = model.fit(X_train, y_train_hot, batch_size=None, epochs=250, steps_per_epoch=150, validation_steps=200, verbose=verbose, validation_data=(X_test, y_test_hot), callbacks=[es])

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(1, len(loss) + 1)
fig = plt.figure(figsize=(14, 5))
plt.plot(epochs, loss, 'r+', label='Training loss')
plt.plot(epochs, val_loss, 'b+', label="'Validation loss")
plt.title('Training and validation loss for MFCC {}'.format(feature_dim_2))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(hist.history['val_acc'], label='Accuracy')
plt.title("Final accuracy on validation data for epochs 250 with early stopping for MFCC {}: {:.3%}".format(feature_dim_2))
plt.legend(loc=2)
plt.show()