import os
import cv2
import mediapipe as mp
from dollarpy import Recognizer, Template, Point



def extract_landmarks_from_videos(video_path):
    mp_pose = mp.solutions.pose
    points = []
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)
                if results is None:
                    print('tmm')

                landmarks = results.pose_landmarks
                for id, landmark in enumerate(landmarks.landmark):
                    if id == 16:
                        points.append(Point(landmark.x, landmark.y, id))

            cap.release()
            cv2.destroyAllWindows()
            print('points',points)
            return points

def find_template_videos(label):
    Templates = []
    for filename in os.listdir('videosNew/' + label):
        video = 'videosNew/'+ label + '/' + filename
        print(video)
        # print(folder)
        Templates.append(Template(label ,extract_landmarks_from_videos(video)))
    return Templates
def test_video(Templates,label):
    for filename in os.listdir('test/' + label):
        video = 'test/'+ label + '/' + filename
        print(video)
        testingpoints = extract_landmarks_from_videos(video)
        recognizer = Recognizer(Templates)
        result = recognizer.recognize(testingpoints)
        print(result)

def main():
    Templates = find_template_videos('Good')
    Templates.extend(find_template_videos('Bad'))
    test_video(Templates,'Good')
    test_video(Templates,'Bad')


main()

# id 0 Good,Bad,Good,Bad,Bad,Bad
# id 11 Good,Good,Good,Bad,Bad,Bad
# id 12 Good,Good,Bad,Bad,Good,Bad
# id 16 Good,Good,Bad,Bad,Good,Good