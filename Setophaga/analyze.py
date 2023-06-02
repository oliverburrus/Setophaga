import sounddevice as sd
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
import wave
import pylab
import io
import os
import requests
from pydub import AudioSegment
import numpy as np
import pandas as pd
import pkg_resources
import urllib

class Analyze:
    def __init__(self, filename, sound_info=None, frame_rate=None, audio_length=None, prediction=None):
        self.filename = filename
        self.sound_info = sound_info
        self.frame_rate = frame_rate
        self.audio_length = audio_length
        self.prediction = prediction
        self.binary_model = None
        self.warbler_model = None
        self.confidence = 0.7
        self.get_models()
        self.get_wav_info()
        self.analyze()

    def get_models(self):
        print("Getting models ready...")
        package_path = pkg_resources.resource_filename(__name__, '')
        models_dir = os.path.join(package_path, 'models')
        os.makedirs(models_dir, exist_ok=True)
        if not(os.path.exists(os.path.join(models_dir, 'binary.h5'))):
            model_url = 'https://drive.google.com/uc?export=download&id=14igHOLLg74WiM-eTHPVA9sKs_hAmiuVr'
            model_path = os.path.join(models_dir, 'binary.h5')
            # Download model file
            urllib.request.urlretrieve(model_url, model_path)
        else:
            model_path = os.path.join(models_dir, 'binary.h5')
        # Load model from file
        self.binary_model = tf.keras.models.load_model(model_path)

        if not(os.path.exists(os.path.join(models_dir, 'warbler.h5'))):
            model_url = 'https://drive.google.com/uc?export=download&id=1cFwNVpCaMacM9fDv_2qIEOB70XkwKfKs'
            model_path = os.path.join(models_dir, 'warbler.h5')
            # Download model file
            urllib.request.urlretrieve(model_url, model_path)
        else:
            model_path = os.path.join(models_dir, 'warbler.h5')
        # Load model from file
        self.warbler_model = tf.keras.models.load_model(model_path)

    def get_wav_info(self):
        with wave.open(self.filename, 'rb') as wav:
            self.frame_rate = wav.getframerate()
            self.audio_length = wav.getnframes() / float(self.frame_rate)
            self.sound_info = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)

    def analyze(self):
        audio_file = AudioSegment.from_file(self.filename)
        for i in range(0, round(self.audio_length)):
            start_time = i * 1000
            end_time = (i + 1) * 1000
            samples = audio_file[start_time:end_time]

            plt.figure(figsize=(6.4, 4.8))
            plt.specgram(samples, Fs=self.frame_rate, cmap='gray_r', scale='dB', mode='magnitude')
            plt.axis('off')
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)
            img_buf.seek(0)
            image = tf.keras.preprocessing.image.load_img(img_buf, target_size=(256, 256))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image /= 255.0
            image = np.expand_dims(image, axis=0)

            plt.close()
            img_buf.close()

            predictions = self.binary_model.predict(image)
            if predictions[0][0] > self.confidence:
                self.prediction = "bird call detected"
            else:
                self.prediction = "No bird calls detected"

    def get_prediction(self):
        return self.prediction
