import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Dropout, LSTM
from keras.optimizers import Adam

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


activities = ['Good','Bad']
def getlandmarks(activity,path):
        ct=0
        cap = cv2.VideoCapture(path)
        landmarks = []
        print(path)
        while True:
            ct+=1
            success, img = cap.read()
            success, frames = cap.read()
            try:
                imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
            except:
                break
            results = pose.process(imgRGB)
            if results.pose_landmarks:
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    pass
            # cv2.imshow("Image", img)
            # cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()
        print(ct)

def extracfeatures():
    pass

for i in range(len(activities)):
    for file in os.listdir("videosNew" + "/" + activities[i]):
        path = "videosNew" + '/' + activities[i] + "/" + file
        landmarks = []
        getlandmarks(activities[i],path)



classifier = Sequential()

# #Adding the input LSTM network layer
# classifier.add(LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
# classifier.add(Dropout(0.2))
#
# #Adding a second LSTM network layer
# classifier.add(CuDNNLSTM(128))
#
# #Adding a dense hidden layer
# classifier.add(Dense(64, activation='relu'))
# classifier.add(Dropout(0.2))
#
# #Adding the output layer
# classifier.add(Dense(10, activation='softmax'))
#
#
# #Compiling the network
# classifier.compile( loss='sparse_categorical_crossentropy',
#               optimizer=Adam(lr=0.001, decay=1e-6),
#               metrics=['accuracy'] )
#
# #Fitting the data to the model
# classifier.fit(X_train,
#          y_train,
#           epochs=3,
#           validation_data=(X_test, y_test))