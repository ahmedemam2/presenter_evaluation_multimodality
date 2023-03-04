from deepface import DeepFace

import cv2
from deepface import DeepFace
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np
import matplotlib.pyplot as plt
leftct,centerct,rightct = 0,0,0
eyeposition = ['Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'right', 'right', 'right', 'right', 'right', 'Center', 'Center', 'Center', 'right', 'right', 'right', 'right', 'Center', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'right', 'Center', 'right', 'right', 'Center', 'right', 'right', 'Center', 'Center', 'right', 'right', 'Center', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', ' left', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center', 'Center'
]
ct=0
posture = 'bad posture'
ctflag = 0
statelist = []
happy = 0
sad = 0
fear = 0
disgust = 0
surprised = 0
angry = 0
neutral=0
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 0, 0)
fontScale = 1
gazelist = []

cap = cv2.VideoCapture('videos/BadPosture/2.mp4')

print("1")
while True:
    ret, frame = cap.read()

    if ret:
        print("2")
        #result = DeepFace.analyze(frame, actions=['emotion'])
        #result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        print("3")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if ctflag > 0:
            statecheck = temp
            # gazecheck = gazetemp
            # print(statecheck, temp)
            # frameres = cv2.resize(frame,(0,0),None,0.25,0.25)
        frameres = frame
        predict = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        #predict = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        state = predict['dominant_emotion']
        temp = state
        # gazetemp = eyeposition[ct]
        stategaaze = eyeposition[ct]
        # zip to iterate on both in same loop
        if ctflag > 0:
            if statecheck != state:
                statelist = statecheck
                if statelist == 'happy':
                    happy += 1
                if statelist == 'sad':
                    sad += 1
                if statelist == 'angry':
                    angry += 1
                if statelist == 'surprised':
                    surprised += 1
                if statelist == 'fear':
                    fear += 1
                if statelist == 'disgust':
                    disgust += 1
                if statelist == 'neutral':
                    neutral += 1
            frame = cv2.putText(frame, "happy" + ":" + str(happy), (10, 50), font,
                                fontScale, color, 1, cv2.LINE_AA)
            frame = cv2.putText(frame, "surprised" + ":" + str(surprised), (160, 50), font,
                                fontScale, color, 1, cv2.LINE_AA)
            frame = cv2.putText(frame, "angry" + ":" + str(angry), (10, 100), font,
                                fontScale, color, 1, cv2.LINE_AA)
            frame = cv2.putText(frame, "fear" + ":" + str(fear), (160, 100), font,
                                fontScale, color, 1, cv2.LINE_AA)
            frame = cv2.putText(frame, "disgust" + ":" + str(disgust), (10, 150), font,
                                fontScale, color, 1, cv2.LINE_AA)
            frame = cv2.putText(frame, "sad" + ":" + str(sad), (160, 150), font,
                                fontScale, color, 1, cv2.LINE_AA)
            frame = cv2.putText(frame, "neutral" + ":" + str(neutral), (10, 200), font,
                                fontScale, color, 1, cv2.LINE_AA)
        # if ctflag==0:
        #     if eyeposition[0] == 'right':
        #         rightct += 1
        #     if eyeposition[0] == 'left':
        #         print('left')
        #         leftct += 1
        #     if eyeposition[0] == 'Center':
        #         centerct += 1


        # cv2.putText(frame,
        #             result['dominant_emotion'],
        #             (50, 50),
        #             font, 3,
        #             (0, 255, 0),
        #             cv2.LINE_4);
        # cv2.putText(frame,
        #             'Right Eye:' + eyeposition[ct],
        #             (50, 150),
        #             font, 1,
        #             (0, 255, 0),
        #             cv2.LINE_4);
        cv2.putText(frame,
                    'left:' + str(leftct),
                    (300, 200),
                    font, 1,
                    (0, 255, 0),
                    cv2.LINE_4);
        cv2.putText(frame,
                    'Right:' + str(rightct),
                    (300, 250),
                    font, 1,
                    (0, 255, 0),
                    cv2.LINE_4);
        cv2.putText(frame,
                    'Center:' + str(centerct),
                    (300, 300),
                    font, 1,
                    (0, 255, 0),
                    cv2.LINE_4);
        cv2.putText(frame,
                    'posture:' ,
                    (10, 450),
                    font, 1,
                    (0, 255, 0),
                    cv2.LINE_4);
        if ct > 60:
            cv2.putText(frame,
                        'bad',
                        (150, 450),
                        font, 1,
                        (0, 255, 0),
                        cv2.LINE_4);
        cv2.imshow('Demo', frame)
        ct+=1
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
        ctflag += 1
    else:
        break
cap.release()
cv2.destroyAllWindows
# Xtest = np.array([[0.5929877758],[0.4933790863],	[0.5826571584],[0.4799264371],[0.5898668227],[0.4882222669],[0.590601027],[0.4890303016],[0.5826571584],[0.4799264371],[0.003028683858],[0.003916791679],[0.01033061743],[0.01345264912],	[0.003884673119],	[0.002979189157],	[0.02163675846],[-0.3203185927]])
# Xtemp=np.array([[0.5871326923],[0.4775428474],[0.5784017444],[0.4658283591],[0.584083569],[0.4726485384],[0.5843223929],[0.4734421372],[0.5784017444],[0.466676712],[0.002497697],[0.004414497238],	[0.008730947971],	[0.01171448827],	[0.004393041134],[0.009506523609],[-0.7598333528],[-1.396687218]])
# distance, path = fastdtw(Xtemp, Xtest, dist=euclidean)
# ct=0
# gazelist = []
# for i in eyeposition:
#     gazetemp = i
#     if ctflag > 0:
#         gazecheck = gazetemp
#     if ctflag > 0:
#         if gazecheck != gazetemp:
#             gazelist = gazecheck
#             if gazelist == 'right':
#                 rightct += 1
#             if gazelist == 'left':
#                 leftct += 1
#             if gazelist == 'Center':
#                 centerct += 1
#     ctflag += 1
# if ct>10:
#     print('BadEyeContact')
# else:
#     print('GoodEyeContact')
# #
#
# y = [happy, sad, neutral, angry, surprised, fear]
# x = ["Happy", "Sad", "Neutral", "Angry", "Surprised", "Fear"]
# print(rightct, leftct, centerct)
# plt.plot(x, y)
# plt.show()
#
# y = [rightct, leftct, centerct]
# x = ["Right", "Left", "Center"]
# print(rightct, leftct, centerct)
# plt.plot(x, y)
# plt.show()