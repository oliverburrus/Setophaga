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


# Set up audio parameters
#sample_rate = 16000  # Sample rate in Hz
#duration = 0.5  # Duration of each audio frame in seconds
#n_fft = 2048  # Number of FFT points


class Analyze:
    def __init__(self, filename, sound_info, frame_rate, audio_length, prediction, 
                binary_model=None, warbler_model=None, confidence=0.7):
        self.filename = filename
        self.sound_info = sound_info
        self.frame_rate = frame_rate
        self.audio_length = audio_length
        self.prediction = prediction
        self.binary_model = binary_model
        self.warbler_model = warbler_model
        self.confidence = confidence
        self.load_models()

    def load_models(self):
        print("Loading models...")
        binary_model_path = 'models/binary.h5'
        self.binary_model = tf.keras.models.load_model(binary_model_path)

        warbler_model_path = 'models/warbler.h5'
        self.warbler_model = tf.keras.models.load_model(warbler_model_path)

    def get_wav_info(self):
        wav = wave.open(self.filename, 'r')
        frames = wav.readframes(-1)
        self.sound_info = pylab.frombuffer(frames, 'int16')
        self.frame_rate = wav.getframerate()
        self.audio_length = len(self.sound_info) / self.frame_rate
        wav.close()
        #return self.sound_info, self.frame_rate, self.audio_length

    # def save_spectrogram(spectrogram):
    #     # Convert spectrogram to a plot
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     im = ax.imshow(spectrogram, aspect='auto', origin='lower')
    #     plt.colorbar(im, ax=ax)

    #     # Save the plot as a PNG image
    #     plt.savefig('sample.png')
    #     plt.close(fig)

    def analyze(self):
        # Load the audio file
        self.load_models()
        audio_file = AudioSegment.from_file(self.filename)
        for i in range(0, round(self.audio_length)):
            # Set the start and end time for the segment in milliseconds
            start_time = i*1000
            end_time = (i+1)*1000    

            samples = audio_file[start_time:end_time]

            #frequencies, times, spectrogram = signal.spectrogram(samples, self.sample_rate, scaling='spectrum', mode='magnitude', nfft=2048, window=('tukey', 0.25))

            # Plot spectrogram
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
            print(predictions[0][0])
            if predictions[0][0] > self.confidence:
                self.prediction = "bird call detected"
                # sound_info, frame_rate = get_wav_info("sample.wav")
                # print("Sample rate - Analyze:" + str(frame_rate))
                # print("Audio length - Analyze:" + str(len(sound_info)))
                # pylab.specgram(sound_info, Fs=frame_rate)
                # pylab.savefig('sample.png')
                # image = tf.keras.preprocessing.image.load_img('sample.png', target_size=(256, 256))
                # image = tf.keras.preprocessing.image.img_to_array(image)
                # image /= 255.0
                # image = np.expand_dims(image, axis=0)
                # predictions = warbler_model.predict(image)
                # df = pd.read_csv("https://raw.githubusercontent.com/oliverburrus/NFCPi/main/models/class_names.csv")
                # # Create dataframe with class names and predictions
                # df['Prediction'] = predictions[0]
                # df['percentage'] = df['Prediction'].apply(lambda x: f"{x:.2%}")
                # # Sort dataframe by prediction in descending order
                # df = df.sort_values(by='Prediction', ascending=False)
                # # Reset index
                # df = df.reset_index(drop=True)
                # if df.Prediction[0] > .7:
                #     # specify the source and destination file paths
                #     src_file = 'sample.png'
                #     dst_file = "/home/pi/NFCPi/flask_app/static/images/last_detect.png"
                #     # copy the file from source to destination
                #     shutil.copy(src_file, dst_file)
                #     file.export("/home/pi/NFCPi/flask_app/static/audio/"+ str(datetime.now().strftime('%Y%m%d%H%M%S'))+".wav", format="wav")
                #     prediction = "This audio is likely of a(n) " + str(df.Species[0]) + " with a probability of " + str(df.percentage[0])
                #     new_data = pd.DataFrame({'Species': [df.Species[0]], "Probability": [df.percentage[0]], "DT": [datetime.now()]})
                #     if os.path.exists("flask_app/detections.csv"):
                #         df1 = pd.read_csv("flask_app/detections.csv")
                #         df1 = pd.concat([df1, new_data], ignore_index=True)
                #     else:
                #         df2 = pd.DataFrame(columns=['Species', 'Probability', 'DT'])
                #         df1 = pd.concat([df2, new_data], ignore_index=True)
                #     df1.to_csv("flask_app/detections.csv", index=False)
                # else:
                #     print("3\n")
                #     prediction = "Not confident in my prediction, likely not a warbler call."
            else:
                self.prediction = "No bird calls detected"

    def get_prediction(self):
        return self.prediction
