import os
import cv2
import mediapipe as mp
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def extract_landmarks_from_videos(video_path):
    mp_pose = mp.solutions.pose
    points = []
    ct=0
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ct=1
                ret, frame = cap.read()
                if not ret:
                    break
                print(ct)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)
                if results is None:
                    print('tmm')

                landmarks = results.pose_landmarks
                for id, landmark in enumerate(landmarks.landmark):
                    if id == 12:
                        points.append((landmark.x,landmark.y))

            cap.release()
            cv2.destroyAllWindows()
            return points

def create_ftw_templates(label):
    template = []
    ct=0
    print(label)
    for filename in os.listdir('videosNew/'  label):
        video = 'videosNew/' label  '/'  filename
        template.append(extract_landmarks_from_videos(video))
    return template

# template_16_Bad = extract_landmarks_from_videos("videosNew/Bad/Untitled video - Made with Clipchamp (1).mp4")
# print(template_16_Bad)
def test_ftw_templates(label):
    test_template = []
    ct=0
    print(label)
    for filename in os.listdir('test/' + label):
        video = 'test/' + label + '/' + filename
        print(video)
        test_template = extract_landmarks_from_videos(video)
        print(test_template)
        for i in range(8):
            distance, path = fastdtw(good_templates[i], test_template,dist=euclidean)
            print("Good test")
            print(distance)
            print('*')
            print('*')
            print('*')
            distance2, path = fastdtw(bad_templates[i], test_template, dist=euclidean)
            print("Bad test")
            print(distance2)
            print('*')
            print('*')
            print('*')
            if label == "Good":
                if distance < distance2:
                    print("Correct")
                    ct=1
            else:
                if distance2 < distance:
                    print("Correct")
                    ct=1
    print(ct)

good_templates = create_ftw_templates("Good")
print(good_templates)
bad_templates = create_ftw_templates("Bad")
print(bad_templates)

test_ftw_templates("Good")
test_ftw_templates("Bad")

# 14/24 correct classifications for good, 18/24 correct classifications for bad using nose.
# 11/24 for good, 14/24 for bad in left shoulder.
# 6/24 for good, 14/24 for bad in right shoulder.