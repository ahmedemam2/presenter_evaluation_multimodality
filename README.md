Evaluation of Multimodality in Presentation Delivery

This project endeavored to quantitatively evaluate a presenter's performance by scrutinizing four integral aspects: vocal attributes, physical posture, ocular movement (eye gaze), and facial expressions.

Here is a demo for the project.


https://github.com/ahmedemam2/presenter_evaluation_multimodality/assets/88343933/f75c6c63-9e62-4b88-9e58-bdb36b3c97e1




Libraries used:
Mediapipe, NLTK, Keras, Sklearn, pandas, numpy.

To try the system, run the python file integrate.py upload your video and you will receive a result like this <img width="394" alt="Screenshot 2023-06-20 192844" src="https://github.com/ahmedemam2/presenter_evaluation_multimodality/assets/88343933/25a6cd80-32b2-4a84-910e-aad25ff14257">
, which will be saved locally as a text file.

In assessing vocal attributes:
Both voice emotion and pronunciation were taken into consideration. The Ravdess Voice Emotion Dataset served as a valuable resource for emotional evaluation. For pronunciation analysis, Google Cloud's Speech-to-Text API was employed. The resultant output was tokenized and lemmatized before the Jaccard similarity index was applied to compare the presenter's script with the actual delivery.

For posture evaluation:
A dataset was generously supplied by the Mass Communication faculty at MSA, comprising multiple videos categorized into 'good' and 'poor' postural habits. This allowed for model training and testing, even enabling the use of ensemble methods for optimally performative models.

Regarding eye gaze:
This aspect was scrutinized by utilizing Euclidean distance calculations to derive the ratio between the iris size and one side of the eye. This method enabled an understanding of the presenter's focus, as the script's optimal position is centralized.

In relation to facial expression:
The DeepFace library was employed for the face emotion detection.

Future prospects include the implementation of Convolutional Neural Networks (CNN) on a facial emotion dataset to enhance the evaluation process.

Results: 
![image](https://github.com/ahmedemam2/presenter_evaluation_multimodality/assets/88343933/ea694539-f5a9-4bed-b696-9946c4c6bc2a)


In summary, this quantitative approach allows for a comprehensive evaluation of a presenter's performance, encapsulating voice, posture, eye gaze, and facial expressions. However, the presenter's appearance, a variable not yet explored in this project, also contribute to the overall evaluation.

