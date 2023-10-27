import pandas as pd
import numpy as np
import os
import librosa
from keras.models import load_model
from moviepy.editor import VideoFileClip
import math


def ensemble(model1, model2, X_test, classes):
    X_test = np.expand_dims(X_test, axis=1)
    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    avg_confidences = []
    for pred1, pred2 in zip(y_pred1, y_pred2):
        avg_confidence = (pred1 + pred2) / 2
        avg_confidences.append(avg_confidence)
    avg_confidences = np.array(avg_confidences)
    avg_confidences = np.squeeze(avg_confidences)
    predicted_classes = np.argmax(avg_confidences, axis=1)
    predicted_labels = [classes[pred_class] for pred_class in predicted_classes]
    print("Predicted classes:", predicted_labels)
    return predicted_labels

def test(X_test,classes,model):
    X_test = np.expand_dims(X_test, axis=1)
    prediction = model.predict(X_test)
    print(prediction)
    predicted_classes = np.argmax(prediction, axis=1)
    print(predicted_classes)
    predicted_labels = [classes[i] for i in predicted_classes]
    print("Predicted classes:", predicted_labels)
    return predicted_labels


def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    res1 = extract_features(data,sample_rate)
    result = np.array(res1)

    return result

def extract_features(data,sample_rate):
    result = np.array([])

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result



def cut_video_into_voice_records(video_path, output_directory):
    video = VideoFileClip(video_path)
    os.makedirs(output_directory, exist_ok=True)
    duration = video.duration
    overlap = 0.2
    segment_length = 1.5
    start_time = 0.0

    while start_time + segment_length <= duration:
        end_time = start_time + segment_length
        audio = video.audio.subclip(start_time, end_time)
        output_file = os.path.join(output_directory, f"second_{math.floor(start_time)}.mp3")
        audio.write_audiofile(output_file, codec="mp3")
        start_time += (segment_length - overlap)
    video.close()



def process_audioFiles(path):
    X = []
    for filename in os.listdir(path):
        print(filename)
        if filename.endswith(".mp3"):
            video_path = os.path.join(path, filename)
            feature = get_features(video_path)
            X.append(feature)
    Features = pd.DataFrame(X)
    Features.to_csv(path + '.csv', index=False)
    test_data = Features
    return test_data
    print('ok')


def main_test(test_data):


    # model = load_model("models/final_bilstm85.h5")
    # model2 = load_model("models/final_RNN84.h5")
    # X_test = test_data
    # sc = StandardScaler()
    # X_test = sc.fit_transform(X_test)

    classes = pd.read_csv('Ravdess_standard_nofear.csv').labels
    # class_labels = np.unique(classes)
    # print(class_labels)
    # model = load_model("models/smote_bilstm_nofear.h5")
    # model2 = load_model("models/smote_RNN_nofear.h5")
    # test(X_test,class_labels,model)
    # test(X_test,class_labels,model2)
    # predictions = ensemble(model,model2,X_test,class_labels)
    # model = load_model("models/smote_bilstm_nofear.h5")
    model2 = load_model("models/smote_RNN_nofear.h5")
    X_test = test_data
    class_labels = np.unique(classes)
    print(class_labels)
    # test(X_test, class_labels, model)
    test(X_test, class_labels, model2)
    # ensemble(model, model2, X_test, class_labels)
    predictions = test(X_test,class_labels,model2)
    with open('voiceEmotions.txt', 'w') as f:
        for item in predictions:
            f.write("%s\n" % item)


def main(presenter_id):
    video_path = presenter_id
    output_directory = "".join([presenter_id, "presentation"])
    cut_video_into_voice_records(video_path, output_directory)
    test_data = process_audioFiles(output_directory)
    main_test(test_data)

