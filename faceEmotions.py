import cv2
from deepface import DeepFace

def get_face_emotions(path,prediction_list):
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if ret:
            predict = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            state = predict['dominant_emotion']
            print(state)
            prediction_list.append(state)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
        else:
            break

    print(prediction_list)
    with open('faceEmotion.txt', 'w') as f:
        for item in prediction_list:
            f.write("%s\n" % item)
