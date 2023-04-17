import pandas as pd
import numpy as np
import os
import sys
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense , GRU , LSTM, Bidirectional



def process_audioFiles():
    ravdess_directory_list = os.listdir("Ravdess")

    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        actor = os.listdir("Ravdess/" + dir)
        for file in actor:
            part = file.split('.')
            if int(part[0][7]) != 2:
                file_emotion.append(int(part[0][7]))
                file_path.append("Ravdess/" + dir + '/' + file)

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)


    Ravdess_df.Emotions.replace(
        {1: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)

    Ravdess_df.to_csv('Ravdess_df.csv', index=False)
def visualize_data(targets):
    y_axis = []
    df = pd.read_csv('Ravdess_df.csv')
    value_counts = df['Emotions'].value_counts()
    for target in targets:
        count_target_value = value_counts[target]
        y_axis.append(count_target_value)
    x_axis = targets
    plt.bar(x_axis, y_axis)
    plt.show()
# visualize_data(["happy", "sad", "surprise", "angry", "fear","disgust","neutral"])

def extract_features(file):
    result = ([])
    y, sr = librosa.load(file)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0)
    result = np.hstack((result,mfcc))
    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally
    # Chroma_stft
    # stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally
    return result
    print("***")
    # print(chroma_stft)
    # print("***")
    # print(mel)


def process_filess(files):
    X, Y = [], []
    for path, emotion in zip(files.Path, files.Emotions):
        print(path)
        feature = extract_features(path)
        X.append(feature)
        Y.append(emotion)
    return(X,Y)

def import_data():
    dftrain = pd.read_csv('features_sound.csv')
    X_train = dftrain.drop('labels', axis='columns')
    y_train = dftrain.labels
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42)
    return X_train,X_test,y_train,y_test,dftrain

def mlp_accuracy(X_train,X_test,y_train,y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000, random_state=42)

    # Train the MLP classifier on the training data
    mlp.fit(X_train, y_train)

    # Evaluate the performance of the MLP classifier on the test data
    accuracy = mlp.score(X_test, y_test)
    print(accuracy)

def test_deep( X_train , X_test, y_train, y_test,data):
    X_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    y_train = np.asarray(pd.get_dummies(y_train), dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True),
                            input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    history = model.fit(X_train, y_train, epochs=110, batch_size=264, verbose=1)
    plt.plot(np.array(history.history['loss']), "r--")
    plt.plot(np.array(history.history['accuracy']), "g--")
    plt.title("Training session's progress over iterations")
    # plt.legend(loc='lower left')
    plt.ylabel('Training Progress (Loss/Accuracy)')
    plt.xlabel('Training Epoch')
    plt.ylim(0)


def main():
    # files = pd.read_csv("Ravdess_df.csv")
    # # files = files.values.tolist()
    # X,Y=process_filess(files)
    # Features = pd.DataFrame(X)
    # print(Features.head())
    # Features['labels'] = Y
    # Features.to_csv('features_temp.csv', index=False)
    # Features.head()
    X_train , X_test, y_train, y_test,data = import_data()
    mlp_accuracy(X_train,X_test,y_train,y_test)
    test_deep( X_train , X_test, y_train, y_test,data)

main()