from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.metrics import jaccard_distance
from moviepy.editor import VideoFileClip
import speech_recognition as sr
import os

def preprocess_text(text):
    tokens = word_tokenize(text.lower())

    stop_words = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]

    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

def calculate_similarity(paragraph1, paragraph2):
    tokens1 = preprocess_text(paragraph1)
    tokens2 = preprocess_text(paragraph2)
    # Jaccard
    # Similarity = (Size of Intersection) / (Size of Union)
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
    paragraph2 = "hello and welcome to msa news I am Ahmed Hany and here are the headlines president" \
                 " el Sisi directs to exert more effort to control the market and now the details"

    similarity = calculate_similarity(paragraph1, paragraph2)
    print(f"The similarity between the paragraphs is: {similarity}")
    return similarity

