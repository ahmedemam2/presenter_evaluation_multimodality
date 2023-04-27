import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
from keras.optimizers import Adam
from keras.losses import KLDivergence
from keras.losses import Huber
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense , GRU , LSTM, Bidirectional
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    dftrain = pd.read_csv('features+aug_voice.csv')
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

def build_BiLSTM(data,classes):
    X_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    y_train = np.asarray(pd.get_dummies(y_train), dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True),
                            input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))  # Add dropout regularization
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.1))  # Add dropout regularization
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

    model.summary()
    history = model.fit(X_train, y_train, epochs=300, batch_size=264, verbose=1)
    loss, accuracy = model.evaluate(X_test, y_test)

    # Print the test set loss and accuracy
    print('Test set loss:', loss)
    print('Test set accuracy:', accuracy)
    # plt.plot(np.array(history.history['loss']), "r--")
    # plt.plot(np.array(history.history['accuracy']), "g--")
    # plt.title("Training session's progress over iterations")
    # # plt.legend(loc='lower left')
    # plt.ylabel('Training Progress (Loss/Accuracy)')
    # plt.xlabel('Training Epoch')
    # plt.ylim(0)
    # Get predictions on test set
    y_pred = model.predict(X_test)

    y_pred_classes = np.argmax(y_pred, axis=1)

    y_test_classes = np.argmax(y_test, axis=1)

    # Create confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)

    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)

    plt.title('Confusion matrix')

    plt.xlabel('Predicted')

    plt.ylabel('True')

    plt.show()

    from sklearn.metrics import f1_score, accuracy_score
def BiLSTM(data):


    # Split the data into features and labels
    X_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values

    # Reshape the input features
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    # One-hot encode the labels
    y_train = np.asarray(pd.get_dummies(y_train), dtype=np.float32)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Define the model
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.1))  # Add dropout regularization
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.2))  # Add dropout regularization
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.1))  # Add dropout regularization
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    # Define a learning rate scheduler
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Train the model
    history = model.fit(X_train, y_train, epochs=300, batch_size=264, verbose=1, validation_data=(X_test, y_test),
                        callbacks=[reduce_lr])

    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


def build_LSTM(data, classes, d1, d2, u1, u2):
    print("dense_1: ", d1)
    print("dense_2: ", d2)
    print("units_1: ", u1)
    print("units_2: ", u2)

    from keras.layers import Reshape

    # Reshape input to remove extra dimensions
    X_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    y_train = np.asarray(pd.get_dummies(y_train), dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Define model
    from keras.regularizers import l2

    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True
                   # , kernel_regularizer=l2(0.001)
                   ))
    # model.add(Dropout(0.1))
    # model.add(LSTM(64, return_sequences=True  # , kernel_regularizer=l2(0.01)
    #                ))

    model.add(Dropout(0.1))
    model.add(LSTM(128, return_sequences=True  # , kernel_regularizer=l2(0.01)
                   ))

    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.1))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, epochs=300, batch_size=256, verbose=1, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)

    # Print the test set loss and accuracy
    print('Test set loss:', loss)
    print('Test set accuracy:', accuracy)

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    # Get predictions on test set
    y_pred = model.predict(X_test)

    y_pred_classes = np.argmax(y_pred, axis=1)

    y_test_classes = np.argmax(y_test, axis=1)

    # Create confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)

    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)

    plt.title('Confusion matrix')

    plt.xlabel('Predicted')

    plt.ylabel('True')

    plt.show()


def build_Gru(data,classes):
    from keras.layers import GRU, Dense, Dropout
    from keras.models import Sequential
    X_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    y_train = np.asarray(pd.get_dummies(y_train), dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # sc = StandardScaler()
    #
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    # pca = PCA(n_components=2)
    #
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    #

    # Define the model architecture
    model = Sequential()
    model.add(GRU(units=32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.1))
    model.add(GRU(units=128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(units=128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(units=256))
    model.add(Dropout(0.2))
    model.add(Dense(units=y_train.shape[1], activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=300, batch_size=64, validation_split=0.2)
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test set loss:', loss)
    print('Test set accuracy:', accuracy)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    # Get predictions on test set
    y_pred = model.predict(X_test)

    y_pred_classes = np.argmax(y_pred, axis=1)

    y_test_classes = np.argmax(y_test, axis=1)

    # Create confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)

    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)

    plt.title('Confusion matrix')

    plt.xlabel('Predicted')

    plt.ylabel('True')

    plt.show()


def import_data():
    dftrain = pd.read_csv('features+augmentation(3).csv')
    X_train = dftrain.drop('labels', axis='columns')
    y_train = dftrain.labels
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42)
    return X_train,X_test,y_train,y_test,dftrain


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
    classes = ['happy','sad','disgust','angry','disgust','surprise','neutral','fear']

    # mlp_accuracy(X_train,X_test,y_train,y_test)
    # BiLSTM(data)
    build_BiLSTM(data,classes)
    # X_train = X_train.values.tolist()
    # X_test = X_test.values.tolist()
    # y_train = y_train.values.tolist()
    # y_test = y_test.values.tolist()
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(X_train.shape)
    build_LSTM(data,classes,0.1,0.1,64,128)
    build_Gru(data,classes)

main()