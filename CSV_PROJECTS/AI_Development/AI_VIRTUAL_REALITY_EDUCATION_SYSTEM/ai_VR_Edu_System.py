#       ---- IMPORT REQUIRED LIBRARIES ----
import cv2
import torch
import numpy as np 
import json
import openai
import speech_recognition as sr 
import pyttsx3 
from deepface import DeepFace 
from transformers import pipeline 


# Class (EmotionAnalyzer:): Detects emotion 
# form facial expressions and voice input 
class EmotionAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
    # Function analyse_facial_expression(frame):
    # Uses DeepFace.analyze() to recognize emotions.
    # Returns the dominant emotion.
    # Fix: Wrapped in try-except to prevent errors 
    # if analysis fails.
    def analyse_facial_expression(self, frame):
        try:
            emotions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            return emotions[0]['dominant_emotion'] if emotions else 'neutral'
        except Exception as e:
            print(f"Error analyzing facial expression: {e}")
            return 'neutral'
    # Function analyze_voice_emotion(audio_path):
    # Uses speech recognition (recognizer.recognize_google()) 
    # to convert speech to text.
    # Uses Hugging Face's pipeline('text-classification') 
    # to analyze text emotions.
    def analyze_voice_emotion(self, audio_path):
        try:
            emotion_model = pipeline('text-classification', model='bhadresh-savani/distilbert-emotion')
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                audio_text = self.recognizer.recognize_google(audio)
                prediction = emotion_model(audio_text)
                return prediction[0]['label']
        except Exception as e:
            print(f"Error analyzing voice emotion: {e}")
            return "neutral"

# Class (AITutor:): AI tutor that responds using 
# OpenAI and provides text-to-speech output.
# respond(text)
# Calls OpenAI's GPT-4 API with a max_tokens=150.
# Uses "pyttsx3" to vocalize responses.
class AITutor:
    def __init__(self):
        self.engine = pyttsx3.init()
    
    # OPENAI_API_KEY: Authenticates 
    # the code opens the "config.json" file.
    # Reads and loads its contents as a Python dictionary 
    # using "json.load(config_file)".
    with open("config.json") as config_file:
    config = json.load(config_file)
    
    # Extracts the API key stored under "OPENAI_API_KEY" 
    # and assigns it to "openai.api_key".
    openai.api_key = config["OPENAI_API_KEY"]
    
    def respond(self, text):
        try:
            response = openai.Completion.create(
                model='gpt-4',
                prompt=text,
                max_tokens=500
            )
            self.speak(response.choices[0].text.strip())
        # If an error occurs while callling on the "OpenAI API", 
        # it catches the error and returns an error message, 
        # instead of crashing
        except Exception as e:
            print(f"Error generating AI response: {e}")
    
    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

# Class (VRAdaptiveLearning:): Adjusts the learning pace and 
# difficulty based on student emotions and performance.
class VRAdaptiveLearning:
    def __init__(self):
        self.user_state = {'emotion': 'neutral', 'performance': 0.5}
    # update_state(emotion, performance)
    # Stores the user's emotion and performance in self.user_state.
    def update_state(self, emotion, performance):
        self.user_state['emotion'] = emotion
        self.user_state['performance'] = performance
    # adapt_content()
    # Adjusts the learning strategy based on:
    # Negative emotion → Slows down learning.
    # High performance (>80%) → Increases difficulty.
    # Default case → Maintains the current strategy.
    def adapt_content(self):
        if self.user_state['emotion'] in ['sad', 'angry', 'frustrated']:
            return "Adjusting Content: Slowing down pace and providing additional examples."
        elif self.user_state['performance'] > 0.8:
            return "Increasing difficulty level to keep engagement high."
        else:
            return "Maintaining current learning strategy."


# Example Workflow
# Main Workflow (if __name__ == '__main__':)
# Emotion detection: Reads an image using 
# cv2.imread('sample_face.jpg').
# Handles missing image file.
# Performance simulation: Uses np.random.uniform(0, 1).
# Updates VR system state.
# Adjusts learning strategy.
# AI tutor responds with an adaptive message.
if __name__ == '__main__':
    analyzer = EmotionAnalyzer()
    tutor = AITutor()
    vr_system = VRAdaptiveLearning()
    
    # Simulated Data (Replace with actual VR camera feed)
    frame = cv2.imread('sample_face.jpg')  
    if frame is not None:
        emotion = analyzer.analyse_facial_expression(frame)
    else:
        print("Error: Image file not found.")
        emotion = "neutral"
    
    # Simulated Performance Score
    performance_score = np.random.uniform(0, 1)
    vr_system.update_state(emotion, performance_score)
    
    # Adapt Content and Respond
    adjustment = vr_system.adapt_content()
    tutor.respond(adjustment)
    
    print(f"Detected Emotion: {emotion}, Performance Score: {performance_score:.2f}")
    print(adjustment)