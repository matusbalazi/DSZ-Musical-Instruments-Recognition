import os
import numpy as np
from colorama import Fore, Style
from tensorflow.python.keras.models import load_model
from musical_instruments_classifier_cnn_model_training import get_model_filename
from musical_instruments_classifier_cnn_model_training import extract_features
from musical_instruments_classifier_cnn_model_training import get_instruments

RECORDINGS_DIR = "C:\DSZ\samples\my_recordings"
AUDIO_FILE_PATH = "C:\DSZ\samples\my_recordings\my_recording_9_4.wav"

def main():
    instruments_counter = [0, 0, 0, 0]

    # nacitanie natrenovaneho modelu
    model = load_model(get_model_filename())

    # otestovanie modelu na VIACERYCH zvukovych nahravkach
    for file_name in os.listdir(f'{RECORDINGS_DIR}'):
        file_path = f'{RECORDINGS_DIR}/{file_name}'
        features = extract_features(file_path)
        X_test = np.array(features).reshape(-1, 20, 40, 1)
        y_pred = model.predict(X_test)
        prediction = get_instruments()[np.argmax(y_pred)]

        print(f"\nRozpoznany hudobny nastroj v nahravke {Style.BRIGHT}{file_name}{Style.RESET_ALL} je: {Fore.GREEN}{Style.BRIGHT}{prediction}{Fore.RESET}{Style.RESET_ALL}")

        if prediction == get_instruments()[0]:
            instruments_counter[0] += 1
        elif prediction == get_instruments()[1]:
            instruments_counter[1] += 1
        elif prediction == get_instruments()[2]:
            instruments_counter[2] += 1
        elif prediction == get_instruments()[3]:
            instruments_counter[3] += 1

    print("\n")

    i = 0
    while(i < len(get_instruments())):
        print(f"Pocet rozpoznanych hudobnych nastrojov {Fore.RED}{get_instruments()[i]}{Fore.RESET}: {instruments_counter[i]}")
        i += 1

    # otestovanie modelu na JEDNEJ zvukovej nahravke
    # file_path = AUDIO_FILE_PATH
    # features = extract_features(file_path)
    # X_test = np.array(features).reshape(-1, 20, 40, 1)
    # y_pred = model.predict(X_test)
    # prediction = get_instruments()[np.argmax(y_pred)]

    # print(f"\nRozpoznany hudobny nastroj je:{Fore.GREEN}{Style.BRIGHT} {prediction}")

if __name__ == '__main__':
    main()