import faceEmotions
import eyeGaze
import textSimilarity
import posture
import testing
import os
import tkinter as tk
from tkinter import filedialog


def read_predictions(file_name):
    with open(file_name, "r") as file:
        return [line.strip() for line in file.readlines()]

def calculate_good_percentage(lst, good_values):
    return sum(1 for i in lst if i in good_values) / len(lst)

def read_score_from_file(file_name):
    with open(file_name, 'r') as file:
        return float(file.readline().strip())

def write_scores_to_file(scores, file_name):
    try:
        with open(file_name, "w") as file:
            for key, value in scores.items():
                file.write(f"{key}: {value}\n")
        print(f"Scores written to {file_name}")
    except Exception as e:
        print(f"Error writing scores to file: {e}")

def process_video():
    file_path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
    if file_path:
        video_name = os.path.splitext(os.path.basename(file_path))[0]
        print("Selected Video Name:", video_name)
    pronunciation_score = textSimilarity.main(file_path)
    posture_data = posture.main(file_path)
    eye_position_list = []
    eye_gaze = eyeGaze.get_eye_position(file_path, eye_position_list)
    face_emotion_list = []
    face_emotions = faceEmotions.get_face_emotions(file_path, face_emotion_list)
    voice_emotions = testing.main(file_path)

    # posture_data = read_predictions("posture.txt")
    # voice_emotions = read_predictions("voiceEmotions.txt")
    # face_emotions = read_predictions("faceEmotion.txt")
    # eye_gaze = read_predictions("eyeGaze.txt")

    posture_score = calculate_good_percentage(posture_data, ["Good"])
    eye_gaze_score = calculate_good_percentage(eye_gaze, ["Center"])
    #
    face_emotion_score = calculate_good_percentage(face_emotions, ["neutral", "happy"])
    voice_emotion_score = calculate_good_percentage(voice_emotions, ["neutral", "happy"])



    scores = {
        "face emotion skills": face_emotion_score,
        "voice emotion skills": voice_emotion_score,
        "Body language": posture_score,
        "eye gaze skills": eye_gaze_score,
        "Pronunciation": pronunciation_score
    }
    print(scores)
    write_scores_to_file(scores, f"{video_name}_scores.txt")

root = tk.Tk()
button = tk.Button(root, text="Select Video", command=process_video)
button.pack()

root.mainloop()

