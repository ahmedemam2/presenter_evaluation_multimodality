import cv2
import mediapipe as mp
import numpy as np
FONTS = cv2.FONT_HERSHEY_COMPLEX

map_face_mesh = mp.solutions.face_mesh
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

RightEyeRight = [33]
RightEyeLeft = [133]
LeftEyeRight = [362]
LeftEyeLeft = [263]
LeftIris = [474,475,476,477]
RightIris = [469,470,471,472]
pos_faceEm,pos_posture,pos_voiceEm = 0,0,0
eye_position=  []
voiceEmotion = ""
postureConstant = ""
ct = 1
def get_eye_position(path,prediction_list):
    cap = cv2.VideoCapture(path)
    with map_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,
                                min_tracking_confidence=0.5) as face_mesh:
        flag, ctright, ctleft, ct = 0, 0, 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = face_mesh.process(rgb_frame)
            h,w,c = frame.shape
            if results.multi_face_landmarks:
                facepoints =np.array([np.multiply([p.x,p.y],[w,h]).astype(int)
                                     for p in results.multi_face_landmarks[0].landmark])
                (cx,cy),radiusl = cv2.minEnclosingCircle(facepoints[LeftIris])
                (rx,ry), radiusr = cv2.minEnclosingCircle(facepoints[RightIris])
                centerright = np.array([rx,ry],dtype=np.int32)
                distanceHalf = np.linalg.norm(centerright - facepoints[RightEyeRight])
                distanceAll = np.linalg.norm(facepoints[RightEyeLeft] - facepoints[RightEyeRight])
                ratio = distanceHalf / distanceAll
                if ratio <= 0.44:
                    position = 'right'
                    flag=1
                    ctright+=1
                if ratio > 0.44 and ratio <= 0.56:
                    position= 'Center'
                    flag=1
                if ratio > 0.56:
                    position = ' left'
                    flag=1
                    ctleft+=1
                # if flag!=1:
                #     position = 'Eye closed'
                flag = 0
                prediction_list.append(position)

        cv2.destroyAllWindows()
        cap.release()
        print(prediction_list)
    with open('eyeGaze.txt', 'w') as f:
        for item in prediction_list:
            f.write("%s\n" % item)

# get_eye_position("00026.tts",[])