import cv2
import mediapipe as mp
from scipy.stats import kurtosis
import pickle
import pandas as pd
import numpy as np
import statistics

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

activities = ['Good','Bad']

def getlandmarks(frames):
    landmark_ids = [11, 12, 14, 16]
    stats = {i: {'x': [], 'y': []} for i in landmark_ids}
    stat_results = {
        'max': {'x': [], 'y': []},
        'min': {'x': [], 'y': []},
        'mean': {'x': [], 'y': []},
        'median': {'x': [], 'y': []},
        'mode': {'x': [], 'y': []},
        'std': {'x': [], 'y': []},
        'interquart': {'x': [], 'y': []},
        'peaktopeak': {'x': [], 'y': []},
        'kurtosis': {'x': [], 'y': []}
    }
    step = 1
    ct = 0
    while step < len(frames):
        frame = frames[step]
        results = pose.process(frame)
        step += 1
        if results.pose_landmarks:
            ct += 1
            for id, lm in enumerate(results.pose_landmarks.landmark):
                if id in landmark_ids:
                    stats[id]['x'].append(lm.x)
                    stats[id]['y'].append(lm.y)
        if ct % 25 == 0:
            for id in landmark_ids:
                for coord in ['x', 'y']:
                    data = stats[id][coord]
                    if len(data) > 0:
                        stat_results['max'][coord].append(max(data))
                        stat_results['min'][coord].append(min(data))
                        stat_results['mean'][coord].append(statistics.mean(data))
                        stat_results['median'][coord].append(statistics.median(data))
                        stat_results['mode'][coord].append(statistics.mode(data))
                        stat_results['std'][coord].append(statistics.stdev(data))
                        stat_results['interquart'][coord].append(np.percentile(data, 75) - np.percentile(data, 25))
                        stat_results['peaktopeak'][coord].append(max(data) - min(data))
                        stat_results['kurtosis'][coord].append(kurtosis(data))
                    stats[id][coord] = []
            step -= 3

    df_data = {}
    for stat, values in stat_results.items():
        for coord, data in values.items():
            df_data[f"{stat}-{coord}S"] = data
    df = pd.DataFrame(df_data)
    return df

    df.to_csv('overlapping_test_set.csv', index=False)
    main_machine(df)

def get_frames(path):
    cap = cv2.VideoCapture(path)
    list_frames = []
    print(path)
    while True:
        success, frames = cap.read()
        try:
            imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
        except:
            break
        list_frames.append(imgRGB)
    return list_frames

def extract_frames(path):
    # print(path)
    frames = get_frames(path)
    return frames
def main(path):
    frames = extract_frames(path)
    print(len(frames))
    getlandmarks(frames)


def test_featurebased(X_test):
    with open('models/svm_model.sav', 'rb') as f:
        loaded_classifier = pickle.load(f)
    results = loaded_classifier.predict(X_test)
    print(results)
    w = 0
    predictions = []
    for i in range(len(results)):
        if i!=0:
            if i % 4 == 0:
                unique_strings, counts = np.unique(results[w:i], return_counts=True)
                if len(unique_strings) == 1:
                    most_frequent_string = unique_strings[0]
                    predictions.append(most_frequent_string)
                elif counts[0] == counts[1]:
                    most_frequent_string = "Good"
                    predictions.append(most_frequent_string)
                else:
                    most_frequent_index = np.argmax(counts)
                    most_frequent_string = unique_strings[most_frequent_index]
                    predictions.append(most_frequent_string)
                w += 4
    with open('posture.txt', 'w') as f:
        for item in results:
            f.write("%s\n" % item)


def main_machine(df):
    test_featurebased(df)


