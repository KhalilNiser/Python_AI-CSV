
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# Used to interact with OpenAI's GPT-Models.
import openai
# Handles reading and writing from JSON files.
import json
# Used for numerical operations.
import numpy as np 
# Provides support for deep learning operations (PyTorch).
import torch
# Handles image processing.
from PIL import Image
# Converts speech to text.
import speech_recognition as sr 
# pyttsx3: Provides text-to-speech (TTS) capabilities.
import pyttsx3
# Transformers "(from huggingface)": Used for natural 
# language processing (NLP) Models. 
# CLIPProcessor, CLIPModel: Handles "image-text" 
# understanding via "OpenAI's CLIP Model".
from transformers import pipeline, CLIPProcessor, CLIPModel
# Performs facial analysis (e.g., emotion recognition).
from deepface import DeepFace
# Whisper: A speech-to-text model by "OpenAI"
import whisper 
# "moviepy.editor (mp)": Provides video files
import moviepy.editor as mp 


# ---- Load Pre-trained Models ----
# clip_model: Loads OpenAI's Clip model, 
# which can associate images with text
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor: Prepares images for CLIP processing
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# whisper_model: "Loads OpenAI's Whisper" 
# model for transcribing speech to text
whisper_model = whisper.load_model("base")



# ---- Initialize Text-to-Speech (TTS) and Speech Recognition ----
# speech_engine: Initializes the "pyttsx3" 
# text-to-speech engine
speech_engine = pyttsx3.init()
# recognizer: Initializes a speech recognition 
# engine from speech recognition
recognizer = sr.Recognizer()



# ---- Class MultiModal ----
# This class (MultiModal), encapsulates all the 
# functionalities: Text processing, Image 
# captioning/recognition, speech-to-text convertion, 
# sentiment analysis, and multimodal responses  
class MultiModalAI:
    
    
    def __init__(self):
        # Stores user feedback to refine AI responses
      self.user_feedback = {}
      
    
     # OPENAI_API_KEY: Authenticates the code 
    # opens the "config.json" file. 
    # Reads and loads its contennts as a python 
    # dictionary using "json.load(config_file)".
    with open("config.json") as config_file :
            config = json.load(config_file)
            
            
    # Extracts the API key stored under "OPEN_API_KEY" 
    # and assigns to "openai_api_key"
    openai.api_key = config["OPEN_API_KEY"]
      
      
    # ---- Text Processing ----
    def process_text(self, text):
        
        try:
            # "Sends a prompt to GPT-4": Uses OpenAI API 
            # to process text input 
            response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": text}]
            )
            
            return response['choices'][0]['message']['content']
    
        # If an error occurs while calling on the "OpenAI API", 
        # it handles the error, for me. Cathes the error and 
        # returns an error message, instead of crashing.
        except Exception as e:
            # Use "e" to show error details
            return f"Error processing text: {str(e)}"
        
        
    # ---- RETURN_TENSORS='PT' ----
    # ---- return_tensors="pt": In Python code, specifically within 
    # the context of libraries like "Hugging Face's Transformers", 
    # [return_tensors='pt'], is an argument used during tokenization. 
    # "Tokenization", is the process of converting text into numerical 
    # representations (tokens) that machine learning models can understand. 
    # The return_tensors argument specifies the format of these tokens.
    
    # When "return_tensors='pt'"" is used, it instructs the tokenizer to 
    # return the output as PyTorch tensors. "Tensors" are multi-dimensional 
    # arrays, similar to NumPy arrays, but optimized for use with "PyTorch", 
    # a popular deep learning framework. This is particularly useful when 
    # preparing input data for models that expect tensors, such as those 
    # built with "PyTorch".
    
    # Other possible values for return_tensors include: ('tf'): Returns 
    # TensorFlow tensors, ('np'): Returns NumPy arrays, and (None) or not 
    # specified: Returns Python lists.
    
    # The choice of return_tensors value depends on the framework being 
    # used for model training or inference. When working with PyTorch models, 
    # setting return_tensors='pt' ensures that the tokenized input is in the 
    # correct format for the model.
    # ---- Image Captioning ----
    def describe_image(self, image_path):
        # "Opens and image": Loads with PIL.Image
        image = Image.open(image_path)
        # "Processes the image": Converts it into a format usable by CLIP
        inputs = clip_processor(images=image, return_tensors="pt")
        # "Runs the CLIP Model": Extracts image 
        # features and returns them
        outputs = clip_model(**inputs)
        return f"Detected image features: {outputs.logits_per_image}"


    # ---- Speech-to-Text Convertion ----
    def transcribe_audio(self, audio_path):
        # "Loads an audio file": Converts it into 
        # a format usable by Whisper.
        audio = whisper.load_audio(audio_path)
        # "Transcribes the audio": Returns the recognized text
        result = whisper_model.transcribe(audio)
        return result["text"]


    # ---- Video Summarization ----
    def summarize_video(self, video_path):
        # "Loads a video file": Opens it using moviepy (mp)
        video = mp.VideoFileClip(video_path)
        # "Extracts audio": Saves it as "temp_audio.wav". 
        # Stores it into "audio_path"
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path)
        # "Transcribes the audio": Uses Whisper 
        # to convert speech to text
        transcript = self.transcribe_audio(audio_path)
        # "Return a short summary": Outputs the first 
        # 300 charaters of the transcript.
        return f"Summary: {transcript[:300]}..."  


    # ---- Sentiment Analysis Across Modalities ----
    def analyze_sentiment(self, input_data, data_type):
        # If (if) "Text", uses a "pre-trained sentiment 
        # model" to classify emotion (positive/negative)
        if data_type == "text":
            sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
            return sentiment_model(input_data)[0]['label']
        # Else-if (elif) "image", uses "DeepFace", to 
        # analyze facial emotions
        elif data_type == "image":
            return DeepFace.analyze(input_data, actions=['emotion'], enforce_detection=False)[0]['dominant_emotion']
        # Else-if (elif) "audio", Transcribes speech 
        # and analyzes sentiment from text
        elif data_type == "audio":
            transcript = self.transcribe_audio(input_data)
            return self.analyze_sentiment(transcript, "text")
        
        
    # ---- User Feedback Integration ----
    # Stores user feedback for future improvements
    def incorporate_feedback(self, input_data, correction):
        self.user_feedback[input_data] = correction
        return "Feedback recorded and AI will adapt accordingly."

    # ---- Multi-Modal Query Handling ----
    # Determines which function to call based on user "query"
    def respond_to_query(self, query):
        if "describe image" in query:
            return self.describe_image("sample.jpg")
        elif "summarize video" in query:
            return self.summarize_video("sample.mp4")
        elif "analyze sentiment" in query:
            return self.analyze_sentiment("sample.jpg", "image")
        else:
            return self.process_text(query)

# ---- Example Workflow ----
if __name__ == "__main__":
    # Describes an instance of the class "MultiModalAI()" 
    ai_system = MultiModalAI()

    # Describes an image (describe image)
    print("Query: Describe image")
    print(ai_system.respond_to_query("describe image"))

    # Summarizes a video (summarize video)
    print("Query: Summarize video")
    print(ai_system.respond_to_query("summarize video"))

    # Analyzes sentiment (analyze_sentiment)
    print("Query: Sentiment analysis on text")
    print(ai_system.analyze_sentiment("I am very happy today!", "text"))

    
    
    
        
      
      