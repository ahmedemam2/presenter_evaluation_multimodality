import os
import cv2
import numpy as np
from keras.applications import VGG16
from keras.models import Model

# Define function for feature extraction using VGG16
def extract_features(frame, model):
    # Preprocess the image
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    img = img / 255.0
    # Extract features using VGG16
    features = model.predict(img)
    return features

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Define path to video file
video_path = 'videosNew/Good/Untitled video - Made with Clipchamp.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Loop through each frame of the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # Extract features from the frame using VGG16
        features = extract_features(frame, model)
        print(features)
        # Do something with the features (e.g., save them to disk)
        # ...
    else:
        break

# Release the video capture object
cap.release()
