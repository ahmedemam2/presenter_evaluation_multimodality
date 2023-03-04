import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense , GRU , LSTM
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


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


X,y = resize_data(data)
model = Sequential()
model.add(LSTM(64, input_shape=(1, X.shape[2])))
train_deep(X,y, model)
model = Sequential()
model.add(GRU(64, input_shape=(1, X.shape[2])))
train_deep(X,y, model)
model = Sequential()
model.add(SimpleRNN(64, input_shape=(1, X.shape[2])))
train_deep(X,y, model)


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
