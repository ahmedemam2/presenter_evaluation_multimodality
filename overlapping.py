import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from scipy.stats import kurtosis
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

import statistics
#
label=[]
landmarks11xS = []
landmarks11yS = []
landmarks12xS = []
landmarks12yS = []
landmarks14xS = []
landmarks14yS = []
landmarks16xS = []
landmarks16yS = []
maxlistxS = []
minlistxS = []
meanlistxS = []
medianlistxS = []
stdevlistxS = []
modelistxS = []
coefvarxS = []
peakpeakxS =[]
skewxS = []
interquartilexS = []
kurtosisxS = []
sqrtxS = []
powerxS = []

maxlistyS = []
minlistyS = []
meanlistyS = []
medianlistyS = []
stdevlistyS = []
modelistyS = []
coefvaryS = []
peakpeakyS =[]
skewyS = []
interquartileyS = []
kurtosisyS = []
sqrtyS = []
poweryS = []

labellist = []
activities = ['Good','Bad']
def getlandmarks(frames):
        ct=0
        flag = False
        flag_start= True
        step=0
        landmarks11xS = []
        landmarks11yS = []
        landmarks12xS = []
        landmarks12yS = []
        landmarks14xS = []
        landmarks14yS = []
        landmarks16xS = []
        landmarks16yS = []
        while flag_start:
            if step not in range(len(frames)):
                flag_start = False
                break
            frame = frames[step]
            results = pose.process(frame)
            ct = 1
            step=1
            print(ct)
            if results.pose_landmarks:
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    if id == 11:
                        landmarks11xS.append(lm.x)
                        landmarks11yS.append(lm.y)
                    if id == 12:
                        landmarks12xS.append(lm.x)
                        landmarks12yS.append(lm.y)
                    if id == 0:
                        landmarks14xS.append(lm.x)
                        landmarks14yS.append(lm.y)
                    if id == 16:
                        landmarks16xS.append(lm.x)
                        landmarks16yS.append(lm.y)
            if ct%25==0:
                flag = True
            # cv2.imshow("Image", img)
            # cv2.waitKey(1)
            if flag==True:
                # labellist.extend([activity, activity, activity, activity])
                maxlistxS.extend([max(landmarks11xS), max(landmarks12xS), max(landmarks14xS), max(landmarks16xS)])
                maxlistyS.extend([max(landmarks11yS), max(landmarks12yS), max(landmarks14yS), max(landmarks16yS)])
                meanlistxS.extend(
                    [statistics.mean(landmarks11xS), statistics.mean(landmarks12xS), statistics.mean(landmarks14xS),
                     statistics.mean(landmarks16xS)])
                meanlistyS.extend(
                    [statistics.mean(landmarks11yS), statistics.mean(landmarks12yS), statistics.mean(landmarks14yS),
                     statistics.mean(landmarks16yS)])
                minlistxS.extend([min(landmarks11xS), min(landmarks12xS), min(landmarks14xS), min(landmarks16xS)])
                minlistyS.extend([min(landmarks11yS), min(landmarks12yS), min(landmarks14yS), min(landmarks16yS)])
                medianlistxS.extend(
                    [statistics.median(landmarks11xS), statistics.median(landmarks12xS), statistics.median(landmarks14xS),
                     statistics.median(landmarks16xS)])
                medianlistyS.extend(
                    [statistics.median(landmarks11yS), statistics.median(landmarks12yS), statistics.median(landmarks14yS),
                     statistics.median(landmarks16yS)])
                stdevlistxS.extend(
                    [statistics.stdev(landmarks11xS), statistics.stdev(landmarks12xS), statistics.stdev(landmarks14xS),
                     statistics.stdev(landmarks16xS)])
                stdevlistyS.extend(
                    [statistics.stdev(landmarks11yS), statistics.stdev(landmarks12yS), statistics.stdev(landmarks14yS),
                     statistics.stdev(landmarks16yS)])
                modelistxS.extend(
                    [statistics.mode(landmarks11xS), statistics.mode(landmarks12xS), statistics.mode(landmarks14xS),
                     statistics.mode(landmarks16xS)])
                modelistyS.extend(
                    [statistics.mode(landmarks11yS), statistics.mode(landmarks12yS), statistics.mode(landmarks14yS),
                     statistics.mode(landmarks16yS)])
                interquartilexS.extend([np.percentile(landmarks11xS, 75) - np.percentile(landmarks11xS, 25),
                                        np.percentile(landmarks12xS, 75) - np.percentile(landmarks12xS, 25),
                                        np.percentile(landmarks14xS, 75) - np.percentile(landmarks14xS, 25),
                                        np.percentile(landmarks16xS, 75) - np.percentile(landmarks16xS, 25)])
                interquartileyS.extend([np.percentile(landmarks11yS, 75) - np.percentile(landmarks11yS, 25),
                                        np.percentile(landmarks12yS, 75) - np.percentile(landmarks12yS, 25),
                                        np.percentile(landmarks14yS, 75) - np.percentile(landmarks14yS, 25),
                                        np.percentile(landmarks16yS, 75) - np.percentile(landmarks16yS, 25)])
                peakpeakxS.extend([max(landmarks11xS) - min(landmarks11xS), max(landmarks12xS) - min(landmarks12xS),
                                   max(landmarks14xS) - min(landmarks14xS), max(landmarks16xS) - min(landmarks16xS)])
                peakpeakyS.extend([max(landmarks11yS) - min(landmarks11yS), max(landmarks12yS) - min(landmarks12yS),
                                   max(landmarks14yS) - min(landmarks14yS), max(landmarks16yS) - min(landmarks16yS)])
                kurtosisxS.extend(
                    [kurtosis(landmarks11xS), kurtosis(landmarks12xS), kurtosis(landmarks14xS), kurtosis(landmarks16xS)])
                kurtosisyS.extend(
                    [kurtosis(landmarks11yS), kurtosis(landmarks12yS), kurtosis(landmarks14yS), kurtosis(landmarks16yS)])
                labellist.extend([1,1,1,1])
                print(labellist)
                landmarks11xS = []
                landmarks11yS = []
                landmarks12xS = []
                landmarks12yS = []
                landmarks14xS = []
                landmarks14yS = []
                landmarks16xS = []
                landmarks16yS = []
                flag=False
                step-=3

def get_frames(path):
    cap = cv2.VideoCapture(path)
    list_frames = []
    print(path)
    while True:
        success, frames = cap.read()
        # print(success)
        try:
            imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
        except:
            break
        list_frames.append(imgRGB)
    return list_frames

def extract_frames():
    path= "testoverlap.mp4"
    # print(path)
    frames = get_frames(path)
    return frames
def main():
    frames = extract_frames()
    print(len(frames))
    # print (range(len(frames)))
    # print(frames)
    getlandmarks(frames)

main()
df = pd.DataFrame({
    'max-xS':maxlistxS,
    'max-yS':maxlistyS,
    'min-xS':minlistxS,
    'min-yS': minlistyS,
    'mean-xS':meanlistxS,
    'mean-yS':meanlistyS,
    'median-xS':medianlistxS,
    'median-yS':medianlistyS,
    'mode-xS':modelistxS,
    'mode-yS':modelistyS,
    'stdv-xS':stdevlistxS,
    'stdv-yS':stdevlistyS,
    'peaktopeak-xS': peakpeakxS,
    'peaktopeak-yS': peakpeakyS,
    'interquart-xS': interquartilexS,
    'interquart-yS': interquartileyS,
    'kurtosis-xS': kurtosisxS,
    'kurtosis-yS': kurtosisyS,
    'label': labellist
})
df.to_csv('overlapping_dataset.csv',index=False)