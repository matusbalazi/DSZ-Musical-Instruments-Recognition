import librosa
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.utils.np_utils import to_categorical

# definovanie hudobnych nastrojov, ktore je mozne rozpoznat
instruments = ["pia", "sax"]

# funkcia na vyextrahovanie audio priznakov zo zvukoveho suboru
def extract_features(file_path):
    # nacitanie zvukoveho suboru
    y, sr = librosa.load(file_path, mono=True, duration=5)

    # vyextrahovanie MFCC kepstralnych koeficientov
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # orezanie MFCC kepstralnych koeficientov na fixnu dlzku
    if mfccs.shape[1] < 40:
        mfccs = np.pad(mfccs, ((0, 0), (0, 40 - mfccs.shape[1])), mode="constant")
    else:
        mfccs = mfccs[:, :40]
    return mfccs

def main():
    # zadefinovanie CNN modelu
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(20, 40, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(instruments), activation="softmax"))

    # kompilacia modelu
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # natrenovanie modelu na datasete so zvukovymi nahravkami
    X_train = []
    y_train = []

    for instrument in instruments:
        for file_name in os.listdir(f'C:\DSZ\dataset_irmas\{instrument}'):
            file_path = f'C:\DSZ\dataset_irmas\{instrument}/{file_name}'
            features = extract_features(file_path)
            X_train.append(features)
            y_train.append(instrument)

    X_train = np.array(X_train).reshape(-1, 20, 40, 1)
    y_train = np.array(y_train)
    encoder = LabelEncoder()
    y_train = to_categorical(encoder.fit_transform(y_train))
    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2)

    # otestovanie modelu na novej zvukovej nahravke
    file_path = "C:\DSZ\samples\saxophone.mp3"
    features = extract_features(file_path)
    X_test = np.array(features).reshape(-1, 20, 40, 1)
    y_pred = model.predict(X_test)
    prediction = instruments[np.argmax(y_pred)]

    print(f"The predicted instrument is: {prediction}")

if __name__ == '__main__':
    main()
