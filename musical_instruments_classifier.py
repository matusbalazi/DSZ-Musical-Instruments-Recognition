import numpy as np
from colorama import Fore, Style
from tensorflow.python.keras.models import load_model
from musical_instruments_classifier_model_training import get_model_filename
from musical_instruments_classifier_model_training import extract_features
from musical_instruments_classifier_model_training import get_instruments

AUDIO_FILE_PATH = "C:\DSZ\samples\guitar4.mp3"

def main():
    # nacitanie natrenovaneho modelu
    model = load_model(get_model_filename())

    # otestovanie modelu na novej zvukovej nahravke
    file_path = AUDIO_FILE_PATH
    features = extract_features(file_path)
    X_test = np.array(features).reshape(-1, 20, 40, 1)
    y_pred = model.predict(X_test)
    prediction = get_instruments()[np.argmax(y_pred)]

    print(f"\nRozpoznany hudobny nastroj je:{Fore.GREEN}{Style.BRIGHT} {prediction}")

if __name__ == '__main__':
    main()