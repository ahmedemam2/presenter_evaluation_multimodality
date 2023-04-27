import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from keras.losses import Huber
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
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense , GRU , LSTM, Bidirectional
import seaborn as sns
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense , GRU , LSTM
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def BiLSTM(data,classes):


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
    history = model.fit(X_train, y_train, epochs=120, batch_size=264, verbose=1, validation_data=(X_test, y_test),
                        callbacks=[reduce_lr])

    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
def build_BiLSTM(data,classes):
    X_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    y_train = np.asarray(pd.get_dummies(y_train), dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True),
                            input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer='RMSprop', metrics=['accuracy'])

    model.summary()
    history = model.fit(X_train, y_train, epochs=120, batch_size=264, verbose=1)
    loss, accuracy = model.evaluate(X_test, y_test)

    # Print the test set loss and accuracy
    print('Test set loss:', loss)
    print('Test set accuracy:', accuracy)
    plt.plot(np.array(history.history['loss']), "r--")
    plt.plot(np.array(history.history['accuracy']), "g--")
    plt.title("Training session's progress over iterations")
    # plt.legend(loc='lower left')
    plt.ylabel('Training Progress (Loss/Accuracy)')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    from sklearn.metrics import f1_score, accuracy_score
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
    model.add(Dropout(0.1))
    model.add(LSTM(64, return_sequences=True  # , kernel_regularizer=l2(0.01)
                   ))

    # model.add(Dropout(0.1))
    # model.add(LSTM(128, return_sequences=True  # , kernel_regularizer=l2(0.01)
    #                ))

    model.add(Dropout(0.1))
    model.add(LSTM(128))
    model.add(Dropout(0.1))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, epochs=120, batch_size=256, verbose=1, validation_data=(X_test, y_test))
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
    # model.add(GRU(units=128, return_sequences=True))
    # model.add(Dropout(0.3))
    # model.add(GRU(units=128, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(GRU(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(units=y_train.shape[1], activation='softmax'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=120, batch_size=64, validation_split=0.2)
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
    dftrain = pd.read_csv('MasscomDatasetLandmarks.csv')
    X_train = dftrain.drop('label', axis='columns')
    y_train = dftrain.label
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
    classes = ['Good','Bad']

    X_train , X_test, y_train, y_test,data = import_data()
    # BiLSTM(data,classes)
    # mlp_accuracy(X_train,X_test,y_train,y_test)
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
    # build_Gru(data,classes)

main()
dftrain = pd.read_csv('MasscomDatasettemp2.csv')
dftest = pd.read_csv('testDataset3.csv')
y=dftrain.label
X_train = dftrain.drop('label',axis='columns')
y_train = dftrain.label
X_test = dftest.drop('label',axis='columns')
y_test = dftest.label

def train_featurebased(model,modelname):
    print(modelname, 'prediction')
    model.fit(X_train, y_train)
    model_prediction = model.predict(X_test)
    print(model.predict(X_test))
    print(model.score(X_test, y_test))
    return model_prediction
knn = KNeighborsClassifier(n_neighbors=3)
knn_pred=train_featurebased(knn,'knn')
clfRE = RandomForestClassifier(max_depth=2, random_state=0)
rfe_pred=train_featurebased(clfRE,'rfe')
clf = make_pipeline(StandardScaler(), SVC(kernel = 'rbf' ,gamma='auto',probability=True))
svm_pred=train_featurebased(clf,'svm')

y_true=dftest["label"]

data = pd.read_csv('MasscomDatasetLandmarks.csv')
data.info()

def resize_data(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X,y
def train_deep(X,y,model):
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=1)


# X,y = resize_data(data)
# model = Sequential()
# model.add(LSTM(64, input_shape=(1, X.shape[2])))
# train_deep(X,y, model)
# model = Sequential()
# model.add(GRU(64, input_shape=(1, X.shape[2])))
# train_deep(X,y, model)
# model = Sequential()
# model.add(SimpleRNN(64, input_shape=(1, X.shape[2])))
# train_deep(X,y, model)


knn=confusion_matrix(y_true,knn_pred)
svm=confusion_matrix(y_true,svm_pred)
rfe=confusion_matrix(y_true,rfe_pred)

actual = y_true
def confusion_matrix(actual,predicted,modelname):
    cm = metrics.confusion_matrix(actual, predicted)

    class_names = np.unique(dftrain['label'])
    fig = plt.figure(figsize=(16, 14))
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g');  # annot=True to annotate cells
    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize=10)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(class_names, fontsize=10)
    plt.yticks(rotation=0)

    plt.title(modelname, fontsize=20)


    plt.show()

confusion_matrix(actual, knn_pred,'knn confusion matrix')
confusion_matrix(actual, svm_pred,'svm confusion matrix')
confusion_matrix(actual, rfe_pred,'rfe confusion matrix')
