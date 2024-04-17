from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
import numpy as np
import librosa
import librosa.display
import sounddevice as sd
from tensorflow.keras.models import load_model
import streamlit as st
import speech_recognition as sr

def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    max_sentiment = max(sentiment, key=sentiment.get)
    return sentiment

# Load the model (make sure the path is correct)
cnn_model = load_model('emotion_cnn_model.h5')

def record_audio(duration, sr):
    """ Record audio for a given duration and sampling rate. """
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return audio.flatten()  # Return a 1D numpy array

def preprocess_audio(audio, sr, target_length=575, n_mfcc=13):
    """ Preprocess the audio for the model prediction. """
    audio = audio / np.max(np.abs(audio), axis=0)  # Normalize audio
    audio, _ = librosa.effects.trim(audio)  # Trim silence
    
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, max(0, target_length - len(audio))), 'constant')
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    rms = librosa.feature.rms(y=audio)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features = np.vstack([mfcc, rms, zcr])
    
    if features.shape[1] < target_length:
        features = np.pad(features, ((0, 0), (0, target_length - features.shape[1])), 'constant')
    else:
        features = features[:, :target_length]

    return features.T[np.newaxis, :, :]

# Sidebar with options
with st.sidebar:
    st.markdown("### Options")
    option = st.radio("Choose an option:", ("Analyze sentiment from audio", "Enter text manually"))

# Main content
st.title("Emotion Detection and Sentiment Analysis")

if option == "Analyze sentiment from audio":
    st.header("Emotion Detection from Audio")
    st.write("This application predicts your emotional state based on a short audio recording.")
    
    duration = 3  # seconds
    sr = 22050  # sampling rate

    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None

    if st.button('Record Audio'):
        st.write(f"Recording for {duration} seconds...")
        st.session_state.audio_data = record_audio(duration, sr)
        st.success("Recording finished!")
        st.audio(st.session_state.audio_data, format='audio/wav', start_time=0, sample_rate=sr)
        

    if st.session_state.audio_data is not None:
        if st.button('Predict Emotion'):
            st.write("Processing...")
            features = preprocess_audio(st.session_state.audio_data, sr)
            prediction = cnn_model.predict(features)
            emotion_index = np.argmax(prediction)
            emotions = ['😐 Neutral', '😌 Calm', '😄 Happy', '😔 Sad', '😠 Angry', '😨 Fearful', '😖 Disgust', '😲 Surprised']
            predicted_emotion = emotions[emotion_index]
            
            st.success(f"Predicted Emotion: **{predicted_emotion}**")
else:
    st.header("Sentiment Analysis from Text")
    st.write("Enter text below:")
    text = st.text_area("Enter text here")
    if st.button("Analyze Sentiment"):
        if text.strip() == "":
            st.warning("Please enter some text.")
        else:
            st.write("Processing...")
            sentiment = get_sentiment(text)
            st.success("Sentiment analysis completed!")
            for key, value in sentiment.items():
                if key == 'compound':
                    st.write(f"Compond emotion: {'😐 Neutral' if value == 0 else '😄 Positive' if value > 0 else '😔 Negative'} ({value*100}%)")
                elif key == 'neg':
                    st.write(f"Negative emotion: {'😔 Negative'} ({value*100}%)")
                elif key == 'neu':
                    st.write(f"Neutral emotion: {'😐 Neutral'} ({value*100}%)")
                elif key == 'pos':
                    st.write(f"Positive emotion: {'😄 Positive'} ({value*100}%)")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Harsh")
st.markdown("&copy; All rights are reserved 2024")


# Load the model (make sure the path is correct)
# cnn_model = load_model('emotion_cnn_model.h5')

# def record_audio(duration, sr):
#     """ Record audio for a given duration and sampling rate. """
#     audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
#     sd.wait()  # Wait until recording is finished
#     return audio.flatten()  # Return a 1D numpy array

# def preprocess_audio(audio, sr, target_length=575, n_mfcc=13):
#     """ Preprocess the audio for the model prediction. """
#     audio = audio / np.max(np.abs(audio), axis=0)  # Normalize audio
#     audio, _ = librosa.effects.trim(audio)  # Trim silence
    
#     if len(audio) > target_length:
#         audio = audio[:target_length]
#     else:
#         audio = np.pad(audio, (0, max(0, target_length - len(audio))), 'constant')
    
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
#     rms = librosa.feature.rms(y=audio)
#     zcr = librosa.feature.zero_crossing_rate(y=audio)
#     features = np.vstack([mfcc, rms, zcr])
    
#     if features.shape[1] < target_length:
#         features = np.pad(features, ((0, 0), (0, target_length - features.shape[1])), 'constant')
#     else:
#         features = features[:, :target_length]

#     return features.T[np.newaxis, :, :]

# # Sidebar with options
# with st.sidebar:
#     st.markdown("### Options")
#     option = st.radio("Choose an option:", ("Analyze sentiment from audio", "Enter text manually"))

# # Main content
# if option == "Analyze sentiment from audio":
#     st.title('Emotion Detection from Audio')
#     st.write("This application predicts your emotional state based on a short audio recording.")
    
#     duration = 3  # seconds
#     sr = 22050  # sampling rate

#     if 'audio_data' not in st.session_state:
#         st.session_state.audio_data = None

#     if st.button('Record Audio'):
#         st.write(f"Recording for {duration} seconds...")
#         st.session_state.audio_data = record_audio(duration, sr)
#         st.success("Recording finished!")
#         st.audio(st.session_state.audio_data, format='audio/wav', start_time=0, sample_rate=sr)

#     if st.session_state.audio_data is not None:
#         text = get_text_from_audio()
#         st.write(text)
#         if st.button('Predict Emotion'):
#             features = preprocess_audio(st.session_state.audio_data, sr)
#             prediction = cnn_model.predict(features)
#             emotion_index = np.argmax(prediction)
#             emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
#             predicted_emotion = emotions[emotion_index]
            
#             st.write(f"Predicted Emotion: **{predicted_emotion}**")
# else:
#     st.title("Sentiment Analysis App using Text typed by user")
#     st.markdown("---")
#     st.write("Enter text below:")
#     text = st.text_area("Enter text here")
#     if st.button("Analyze Sentiment"):
#         if text.strip() == "":
#             st.write("Please enter some text.")
#         else:
#             sentiment = get_sentiment(text)
#             st.write("Sentiment:", sentiment)

# # Footer
# st.markdown("---")
# st.markdown("Made with ❤️ by Harsh")


    
# # Make streamlit app
# import streamlit as st
# st.title("Sentiment Analysis App")
# if st.button("Analyze sentiment from audio"):
#     text = get_text_from_audio()
#     st.write(text)
#     sentiment = get_sentiment(text)
#     st.write(sentiment)

# else:
#     text = st.text_area("Enter text here")
#     sentiment = get_sentiment(text)
#     st.write(sentiment)