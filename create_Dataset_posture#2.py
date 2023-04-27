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
maxlistx0 = []
minlistx0 = []
meanlistx0 = []
medianlistx0 = []
stdevlistx0 = []
modelistx0 = []
coefvarx0 = []
peakpeakx0 =[]
skewx0 = []
interquartilex0 = []
kurtosisx0 = []
sqrtx0 = []
power0 = []

maxlisty0 = []
minlisty0 = []
meanlisty0 = []
medianlisty0 = []
stdevlisty0 = []
modelisty0 = []
coefvary0 = []
peakpeaky0 =[]
skewy0 = []
interquartiley0 = []
kurtosisy0 = []
sqrty0 = []
powery0 = []


maxlistx11 = []
minlistx11 = []
meanlistx11 = []
medianlistx11 = []
stdevlistx11 = []
modelistx11 = []
coefvarx11 = []
peakpeakx11 =[]
skewx11 = []
interquartilex11 = []
kurtosisx11 = []
sqrtx11 = []
powerx11 = []

maxlisty11 = []
minlisty11 = []
meanlisty11 = []
medianlisty11 = []
stdevlisty11 = []
modelisty11 = []
coefvary11 = []
peakpeaky11 =[]
skewy11 = []
interquartiley11 = []
kurtosisy11 = []
sqrty11 = []
powery11 = []


maxlistx12 = []
minlistx12 = []
meanlistx12 = []
medianlistx12 = []
stdevlistx12 = []
modelistx12 = []
coefvarx12 = []
peakpeakx12 =[]
skewx12 = []
interquartilex12 = []
kurtosisx12 = []
sqrtx12 = []
powerx12 = []

maxlisty12 = []
minlisty12 = []
meanlisty12 = []
medianlisty12 = []
stdevlisty12 = []
modelisty12 = []
coefvary12 = []
peakpeaky12 =[]
skewy12 = []
interquartiley12 = []
kurtosisy12 = []
sqrty12 = []
powery12 = []

maxlistx16 = []
minlistx16 = []
meanlistx16 = []
medianlistx16 = []
stdevlistx16 = []
modelistx16 = []
coefvarx16 = []
peakpeakx16 =[]
skewx16 = []
interquartilex16 = []
kurtosisx16 = []
sqrtx16 = []
powerx16 = []

maxlisty16 = []
minlisty16 = []
meanlisty16 = []
medianlisty16 = []
stdevlisty16 = []
modelisty16 = []
coefvary16 = []
peakpeaky16 =[]
skewy16 = []
interquartiley16 = []
kurtosisy16 = []
sqrty16 = []
poweryS16 = []

labellist = []
activities = ['Good','Bad']
def getlandmarks(activity,path):
        cap = cv2.VideoCapture(path)
        landmarks11xS = []
        landmarks11yS = []
        landmarks12xS = []
        landmarks12yS = []
        landmarks0xS = []
        landmarks0yS = []
        landmarks16xS = []
        landmarks16yS = []
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
                    # if activity=="Good":
                    #     label = 1
                    # else:
                    #     label = 0
                    if id == 11:
                        landmarks11xS.append(lm.x)
                        landmarks11yS.append(lm.y)
                    if id == 12:
                        landmarks12xS.append(lm.x)
                        landmarks12yS.append(lm.y)
                    if id == 0:
                        landmarks0xS.append(lm.x)
                        landmarks0yS.append(lm.y)
                    if id == 16:
                        landmarks16xS.append(lm.x)
                        landmarks16yS.append(lm.y)
            # cv2.imshow("Image", img)
            # cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()
        labellist.append(activity)
        maxlistx11.append(max(landmarks11xS))
        maxlisty11.append(max(landmarks11yS))
        meanlistx11.append(statistics.mean(landmarks11xS))
        meanlisty11.append(statistics.mean(landmarks11yS))
        minlistx11.append(min(landmarks11xS))
        minlisty11.append(min(landmarks11yS))
        medianlistx11.append(statistics.median(landmarks11xS))
        medianlisty11.append(statistics.median(landmarks11yS))
        stdevlistx11.append(statistics.stdev(landmarks11xS))
        stdevlisty11.append(statistics.stdev(landmarks11yS))
        modelistx11.append(statistics.mode(landmarks11xS))
        modelisty11.append(statistics.mode(landmarks11yS))
        interquartilex11.append(np.percentile(landmarks11xS, 75) - np.percentile(landmarks11xS, 25))
        interquartiley11.append(np.percentile(landmarks11yS, 75) - np.percentile(landmarks11yS, 25))
        peakpeakx11.append(max(landmarks11xS) - min(landmarks11xS))
        peakpeaky11.append(max(landmarks11yS) - min(landmarks11yS))
        kurtosisx11.append(kurtosis(landmarks11xS))
        kurtosisy11.append(kurtosis(landmarks11yS))

        maxlistx0.append(max(landmarks0xS))
        maxlisty0.append(max(landmarks0yS))
        meanlistx0.append(statistics.mean(landmarks0xS))
        meanlisty0.append(statistics.mean(landmarks0yS))
        minlistx0.append(min(landmarks0xS))
        minlisty0.append(min(landmarks0yS))
        medianlistx0.append(statistics.median(landmarks0xS))
        medianlisty0.append(statistics.median(landmarks0yS))
        stdevlistx0.append(statistics.stdev(landmarks0xS))
        stdevlisty0.append(statistics.stdev(landmarks0yS))
        modelistx0.append(statistics.mode(landmarks0xS))
        modelisty0.append(statistics.mode(landmarks0yS))
        interquartilex0.append(np.percentile(landmarks0xS, 75) - np.percentile(landmarks0xS, 25))
        interquartiley0.append(np.percentile(landmarks0yS, 75) - np.percentile(landmarks0yS, 25))
        peakpeakx0.append(max(landmarks0xS) - min(landmarks0xS))
        peakpeaky0.append(max(landmarks0yS) - min(landmarks0yS))
        kurtosisx0.append(kurtosis(landmarks0xS))
        kurtosisy0.append(kurtosis(landmarks0yS))

        maxlistx12.append(max(landmarks12xS))
        maxlisty12.append(max(landmarks12yS))
        meanlistx12.append(statistics.mean(landmarks12xS))
        meanlisty12.append(statistics.mean(landmarks12yS))
        minlistx12.append(min(landmarks12xS))
        minlisty12.append(min(landmarks12yS))
        medianlistx12.append(statistics.median(landmarks12xS))
        medianlisty12.append(statistics.median(landmarks11yS))
        stdevlistx12.append(statistics.stdev(landmarks12xS))
        stdevlisty12.append(statistics.stdev(landmarks12yS))
        modelistx12.append(statistics.mode(landmarks12xS))
        modelisty12.append(statistics.mode(landmarks12yS))
        interquartilex12.append(np.percentile(landmarks12xS, 75) - np.percentile(landmarks12xS, 25))
        interquartiley12.append(np.percentile(landmarks12yS, 75) - np.percentile(landmarks12yS, 25))
        peakpeakx12.append(max(landmarks12xS) - min(landmarks12xS))
        peakpeaky12.append(max(landmarks12yS) - min(landmarks12yS))
        kurtosisx12.append(kurtosis(landmarks12xS))
        kurtosisy12.append(kurtosis(landmarks12yS))

        maxlistx16.append(max(landmarks16xS))
        maxlisty16.append(max(landmarks16yS))
        meanlistx16.append(statistics.mean(landmarks16xS))
        meanlisty16.append(statistics.mean(landmarks16yS))
        minlistx16.append(min(landmarks16xS))
        minlisty16.append(min(landmarks16yS))
        medianlistx16.append(statistics.median(landmarks16xS))
        medianlisty16.append(statistics.median(landmarks16yS))
        stdevlistx16.append(statistics.stdev(landmarks16xS))
        stdevlisty16.append(statistics.stdev(landmarks16yS))
        modelistx16.append(statistics.mode(landmarks16xS))
        modelisty16.append(statistics.mode(landmarks16yS))
        interquartilex16.append(np.percentile(landmarks16xS, 75) - np.percentile(landmarks16xS, 25))
        interquartiley16.append(np.percentile(landmarks16yS, 75) - np.percentile(landmarks16yS, 25))
        peakpeakx16.append(max(landmarks16xS) - min(landmarks16xS))
        peakpeaky16.append(max(landmarks16yS) - min(landmarks16yS))
        kurtosisx16.append(kurtosis(landmarks16xS))
        kurtosisy16.append(kurtosis(landmarks16yS))

def extracfeatures():
    pass

for i in range(len(activities)):
    for file in os.listdir("videosNew" + "/" + activities[i]):
        path = "videosNEw" + '/' + activities[i] + "/" + file
        landmarks11xS = []
        landmarks11yS = []
        landmarks12xS = []
        landmarks12yS = []
        landmarks14xS = []
        landmarks14yS = []
        landmarks16xS = []
        landmarks16yS = []
        getlandmarks(activities[i],path)
print(len(maxlistx11))
print(len(interquartilex0))
print(len(peakpeakx16))
print(len(kurtosisx11))

df = pd.DataFrame({
    'max-x0':maxlistx0,
    'max-y0':maxlisty0,
    'min-x0':minlistx0,
    'min-y0': minlisty0,
    'mean-x0':meanlistx0,
    'mean-y0':meanlisty0,
    'median-x0':medianlistx0,
    'median-y0':medianlisty0,
    'mode-x0':modelistx0,
    'mode-y0':modelisty0,
    'stdv-x0':stdevlistx0,
    'stdv-y0':stdevlisty0,
    'peaktopeak-x0': peakpeakx0,
    'peaktopeak-y0': peakpeaky0,
    'interquart-x0': interquartilex0,
    'interquart-y0': interquartiley0,
    'kurtosis-x0': kurtosisx0,
    'kurtosis-y0': kurtosisy0,
    'max-x11':maxlistx11,
    'max-y11':maxlisty11,
    'min-x11':minlistx11,
    'min-y11': minlisty11,
    'mean-x11':meanlistx11,
    'mean-y11':meanlisty11,
    'median-x11':medianlistx11,
    'median-y11':medianlisty11,
    'mode-x11':modelistx11,
    'mode-y11':modelisty11,
    'stdv-x11':stdevlistx11,
    'stdv-y11':stdevlisty11,
    'peaktopeak-x11': peakpeakx11,
    'peaktopeak-y11': peakpeaky11,
    'interquart-x11': interquartilex11,
    'interquart-y11': interquartiley11,
    'kurtosis-x11': kurtosisx11,
    'kurtosis-y11': kurtosisy11,
    'max-x12':maxlistx12,
    'max-y12':maxlisty12,
    'min-x12':minlistx12,
    'min-y12': minlisty12,
    'mean-x12':meanlistx12,
    'mean-y12':meanlisty12,
    'median-x12':medianlistx12,
    'median-y12':medianlisty12,
    'mode-x12':modelistx12,
    'mode-y12':modelisty12,
    'stdv-x12':stdevlistx12,
    'stdv-y12':stdevlisty12,
    'peaktopeak-x12': peakpeakx12,
    'peaktopeak-y12': peakpeaky12,
    'interquart-x12': interquartilex12,
    'interquart-y12': interquartiley12,
    'kurtosis-x12': kurtosisx12,
    'kurtosis-y12': kurtosisy12,
    'max-x16':maxlistx16,
    'max-y16':maxlisty16,
    'min-x16':minlistx16,
    'min-y16': minlisty16,
    'mean-x16':meanlistx16,
    'mean-y16':meanlisty16,
    'median-x16':medianlistx16,
    'median-y16':medianlisty16,
    'mode-x16':modelistx16,
    'mode-y16':modelisty16,
    'stdv-x16':stdevlistx16,
    'stdv-y16':stdevlisty16,
    'peaktopeak-x16': peakpeakx16,
    'peaktopeak-y16': peakpeaky16,
    'interquart-x16': interquartilex16,
    'interquart-y16': interquartiley16,
    'kurtosis-x16': kurtosisx16,
    'kurtosis-y16': kurtosisy16,
    'label': labellist
})
df.to_csv('Dataset_Features_Train.csv',index=False)
