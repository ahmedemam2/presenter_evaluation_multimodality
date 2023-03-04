import librosa

# Load the audio as a waveform `y`
# Sample rate is the number of samples of audio carried per second
# If unspecified, defaults to 22,050 Hz
y, sr = librosa.load("test/Good.mp4", sr=None)

# Get Mel-frequency cepstral coefficients
mfccs = librosa.feature.mfcc(y=y, sr=sr)