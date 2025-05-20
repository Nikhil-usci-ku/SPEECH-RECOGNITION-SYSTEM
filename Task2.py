from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from deepmultilingualpunctuation import PunctuationModel
import torch
import librosa
import os

print("Initializing...")

# Taking a pre-trained Wav2Vec model from hugging face
model_name = "facebook/wav2vec2-base-960h"
wav2vec_processor = Wav2Vec2Processor.from_pretrained(model_name)
wav2vec_model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Taking a pre-trained fullstop punctuation model
punctuation_model = PunctuationModel()

# Taking user input for audio file location
os.system("clear || cls")
audio_file = str(input("Enter Audio File location (wav file only) : "))
if not os.path.exists(audio_file):
    print("File does not exist")
    exit()

# Resampling and preprocessing audio file
try:
    speech_array, original_sampling_rate = librosa.load(audio_file, sr=16000)
except Exception as e:
    print(f"Error loading audio file: {e}")
    exit()
input_values = wav2vec_processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values

# Transcribing audio to raw text
with torch.no_grad():
    logits = wav2vec_model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
raw_transcription = wav2vec_processor.batch_decode(predicted_ids)[0]

# Adding missing punctuation marks
clean_transcription = punctuation_model.restore_punctuation(raw_transcription)
Final_Transcription = ". ".join(i.capitalize() for i in clean_transcription.split(". "))

# Final Result
print("Audio Transcription : ",Final_Transcription)
