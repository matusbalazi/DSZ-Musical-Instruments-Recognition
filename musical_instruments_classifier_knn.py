import os
import numpy as np
import librosa
from colorama import Fore, Style
from sklearn.neighbors import KNeighborsClassifier

INSTRUMENTS = ["guitar", "piano", "saxophone", "violin"]
RECORDINGS_DIR = "C:\DSZ\samples\downloaded_recordings"
DATASET_PATH = "C:\DSZ\dataset_irmas"
AUDIO_FILE_PATH = "C:\DSZ\samples\my_recordings_wav\my_recording_1.wav"

def extract_features(file_path):
    with open(file_path, 'rb') as file:
        audio_data, sample_rate = librosa.load(file)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)

def main():
    instruments_counter = [0, 0, 0, 0]
    dataset = []
    labels = []

    for instrument in INSTRUMENTS:
        for file_name in os.listdir(f'{DATASET_PATH}\{instrument}'):
            file_path = f'{DATASET_PATH}\{instrument}/{file_name}'
            features = extract_features(file_path)
            dataset.append(features)
            labels.append(instrument)

    knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
    knn.fit(dataset, labels)

    for file_name in os.listdir(f'{RECORDINGS_DIR}'):
        file_path = f'{RECORDINGS_DIR}/{file_name}'
        features = extract_features(file_path)
        instrument = knn.predict([features])[0]

        print(f"\nRozpoznany hudobny nastroj v nahravke {Style.BRIGHT}{file_name}{Style.RESET_ALL} je: {Fore.GREEN}{Style.BRIGHT}{instrument}{Fore.RESET}{Style.RESET_ALL}")

        if instrument == INSTRUMENTS[0]:
            instruments_counter[0] += 1
        elif instrument == INSTRUMENTS[1]:
            instruments_counter[1] += 1
        elif instrument == INSTRUMENTS[2]:
            instruments_counter[2] += 1
        elif instrument == INSTRUMENTS[3]:
            instruments_counter[3] += 1

    print("\n")

    i = 0
    while(i < len(INSTRUMENTS)):
        print(f"Pocet rozpoznanych hudobnych nastrojov {Fore.RED}{INSTRUMENTS[i]}{Fore.RESET}: {instruments_counter[i]}")
        i += 1

    # audio_file_path = AUDIO_FILE_PATH
    # features = extract_features(audio_file_path)
    # instrument = knn.predict([features])[0]

    # print(f"\nRozpoznany hudobny nastroj v nahravke {Style.BRIGHT}{audio_file_path}{Style.RESET_ALL} je: {Fore.GREEN}{Style.BRIGHT}{instrument}{Fore.RESET}{Style.RESET_ALL}")

if __name__ == '__main__':
    main()

