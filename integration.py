import cv2
import faceEmotions
import eyeGaze
import textSimilarity
import posture
import testing

path = "00026.tts"
textSimilarity.main(path)
eye_position_list = []
eyeGaze.get_eye_position(path,eye_position_list)
face_emotion_list = []
faceEmotions.get_face_emotions(path,face_emotion_list)
posture.main(path)
testing.main(path)


def read_predictions(file_name):
    with open(file_name, "r") as file:
        mylist = file.readlines()
        predictions = [line.strip() for line in mylist]
        print(predictions)
    return predictions
p=read_predictions("posture.txt")
v=read_predictions("voiceEmotions.txt")
f=read_predictions("faceEmotion.txt")
e=read_predictions("eyeGaze.txt")

with open('Pronounciation.txt', 'r') as file:
    s = float(file.readline().strip())
ct=0
for i in p:
    if i == "Good":
        ct+=1
posture_score = ct/len(p)
print(posture_score)
ct=0
for i in e:
    if i=="Center":
        ct+=1
eye_gaze_score = ct/len(e)
body_language = (eye_gaze_score+posture_score)/2 * 4
print("Body language Score" , body_language,"/4")
ct=0
for i in f:
    if i == "neutral" or i== "happy":
        ct+=1
face_emotion_score = ct/len(f)
ct=0
for i in v:
    if i=="neutral" or i =="happy":
        ct+=1
voice_emotion_score = ct/len(v)
story_telling = (face_emotion_score+voice_emotion_score)/2 * 6
print("Story telling skills Score" , story_telling, "/6")
pronounciation = 3 - (s-0.1)*3
print("Pronounciation Score" , pronounciation,"/3")
variable1 = "Hello"
variable2 = "World"

# Open a file for writing
with open("Report.txt", "w") as file:
    # Write the variables to the file
    file.write(f"Story telling skills: {story_telling}\n")
    file.write(f"Body language: {body_language}\n")
    file.write(f"Pronounciation: {pronounciation}\n")

print("Variables written to file.")

