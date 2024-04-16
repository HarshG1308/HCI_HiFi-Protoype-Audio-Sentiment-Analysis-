from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr

def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    max_sentiment = max(sentiment, key=sentiment.get)
    return sentiment

# def get_text_from_audio():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Say something:")
#         audio = recognizer.listen(source)
#     try:
#         text = recognizer.recognize_google(audio)
#         return text
#     except sr.UnknownValueError:
#         return "Could not understand audio"
#     except sr.RequestError as e:
#         return "Could not request results; {0}".format(e)
    
import streamlit as st
from textblob import TextBlob
import speech_recognition as sr

# Function to perform sentiment analysis on text
# def get_sentiment(text):
#     blob = TextBlob(text)
#     sentiment_score = blob.sentiment.polarity
#     if sentiment_score > 0:
#         return "Positive"
#     elif sentiment_score < 0:
#         return "Negative"
#     else:
#         return "Neutral"

# Function to get text from audio using Speech Recognition
def get_text_from_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Please speak something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        st.write("Sorry, could not understand audio. Please try again.")
        return ""

# Set title and page config
st.title("Sentiment Analysis App")

# Sidebar with options
with st.sidebar:
    st.markdown("### Options")
    option = st.radio("Choose an option:", ("Analyze sentiment from audio", "Enter text manually"))

# Main content
if option == "Analyze sentiment from audio":
    if st.button("Start Recording"):
        text = get_text_from_audio()
        if text:
            st.write("You said:")
            st.write(text)
            sentiment = get_sentiment(text)
            st.write("Sentiment:", sentiment)
else:
    st.markdown("---")
    st.write("Enter text below:")
    text = st.text_area("Enter text here")
    if st.button("Analyze Sentiment"):
        if text.strip() == "":
            st.write("Please enter some text.")
        else:
            sentiment = get_sentiment(text)
            st.write("Sentiment:", sentiment)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Harsh")


    
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