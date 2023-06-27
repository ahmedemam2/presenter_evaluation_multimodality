from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.metrics import jaccard_distance
from moviepy.editor import VideoFileClip
import speech_recognition as sr
import os

def preprocess_text(text):
    tokens = word_tokenize(text.lower())

    stop_words = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

def calculate_similarity(paragraph1, paragraph2):
    tokens1 = preprocess_text(paragraph1)
    tokens2 = preprocess_text(paragraph2)

    distance = jaccard_distance(set(tokens1), set(tokens2))
    similarity = 1 - distance
    print(similarity)
    return similarity

def main(path):

    base_name = os.path.splitext(os.path.basename(path))[0]
    audio_file = base_name + ".wav"
    video = VideoFileClip(path)
    audio = video.audio
    audio.write_audiofile(audio_file)
    print('Audio extracted:', audio_file)

    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        print("Processing audio...")
        audio = r.record(source)

    try:
        text = r.recognize_google(audio)
        print("Text:", text)
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    paragraph1 = text
    paragraph2 = "Hello and welcome to MSA news I'm Sohaila Farid and here are the headlines president" \
                 " el sisi directs to exert more effort to control market and now the details" \
                 "president abdelfatah el sisi held a meeting on saturday with premier minister" \
                 "mostafa madboly governer of the central bank of egypt tarek amer and major general" \
                 "mahmoud zaki minister of defense and military production the spokseman foud that " \
                 "presidency said that the meeting dealed with the governments efforts to provide the " \
                 "necessary commits for citizen during the holy month of ramadan in quantities and" \
                 "appropriate prices" \

    similarity = calculate_similarity(paragraph1, paragraph2)
    print(f"The similarity between the paragraphs is: {similarity}")
    with open('Pronounciation.txt', 'w') as f:
        f.write("%s\n" % similarity)

