import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from scipy.stats import kurtosis
import os

mpPose = mp.solutions.pose
pose = mpPose.Pose()
labellist = []
landmarks11x = []
landmarks11y = []
landmarks12x = []
landmarks12y = []
landmarks14x = []
landmarks14y = []
landmarks16x = []
landmarks16y = []
activities = ['Good','Bad']
def getlandmarks(activity,path):
        cap = cv2.VideoCapture(path)
        print(path)
        while True:
            success, img = cap.read()
            success, frames = cap.read()
            try:
                imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
            except:
                break
            results = pose.process(imgRGB)
            if results.pose_landmarks:
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    if id == 11:
                        landmarks11x.append(lm.x)
                        landmarks11y.append(lm.y)
                        if activity == 'Good':
                            labellist.append(1)
                        else:
                            labellist.append(0)

                    if id == 12:
                        landmarks12x.append(lm.x)
                        landmarks12y.append(lm.y)
                    if id == 0:
                        landmarks14x.append(lm.x)
                        landmarks14y.append(lm.y)
                    if id == 16:
                        landmarks16x.append(lm.x)
                        landmarks16y.append(lm.y)
        cap.release()
        cv2.destroyAllWindows()


for i in range(len(activities)):
    for file in os.listdir("videosNew" + "/" + activities[i]):
        path = "videosNew" + '/' + activities[i] + "/" + file
        getlandmarks(activities[i],path)

print(len(landmarks16x))
print(len(labellist))
df = pd.DataFrame({
    'left_shoulderX' : landmarks11x,
    'left_shoulderY': landmarks11y,
    'right_shoulderX': landmarks12x,
    'right_shoulderY': landmarks12y,
    'right_elbowX': landmarks14x,
    'right_elbowY': landmarks14y,
    'right_wristX': landmarks16x,
    'right_wristY': landmarks16x,
    'label': labellist
})
df.to_csv('MasscomDatasetLandmarks.csv',index=False)
