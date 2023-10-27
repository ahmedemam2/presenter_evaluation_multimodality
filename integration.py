import faceEmotions
import eyeGaze
import textSimilarity
import posture
import testing
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
    with open(file_name, "w") as file:
        for key, value in scores.items():
            file.write(f"{key}: {value}\n")
    print("Scores written to file.")

def process_video():
    file_path = filedialog.askopenfilename()

    textSimilarity.main(file_path)
    posture.main(file_path)
    eye_position_list = []
    eyeGaze.get_eye_position(file_path, eye_position_list)
    face_emotion_list = []
    faceEmotions.get_face_emotions(file_path, face_emotion_list)
    testing.main(file_path)

    posture_data = read_predictions("posture.txt")
    voice_emotions = read_predictions("voiceEmotions.txt")
    face_emotions = read_predictions("faceEmotion.txt")
    eye_gaze = read_predictions("eyeGaze.txt")

    posture_score = calculate_good_percentage(posture_data, ["Good"])
    eye_gaze_score = calculate_good_percentage(eye_gaze, ["Center"])
    body_language_score = ((eye_gaze_score + posture_score) / 2) * 4

    face_emotion_score = calculate_good_percentage(face_emotions, ["neutral", "happy"])
    voice_emotion_score = calculate_good_percentage(voice_emotions, ["neutral", "happy"])
    story_telling_score = ((face_emotion_score) / 2) * 6

    # s = read_score_from_file('Pronounciation.txt')
    # pronunciation_score = 3 - (s - 0.1) * 3

    scores = {
        "Story telling skills": story_telling_score,
        "Body language": body_language_score
        # "Pronunciation": pronunciation_score
    }
    print(scores)
    write_scores_to_file(scores, "Report.txt")

root = tk.Tk()
button = tk.Button(root, text="Select Video", command=process_video)
button.pack()

root.mainloop()

