# App for predicting emotion real-time audio using a pre-trained CNN model

# Importing necessary libraries
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

# Defining the app
app = Flask(__name__)

# Loading the pre-trained model
model = load_model('emotion_cnn_model.h5')

# Function to extract features from audio

