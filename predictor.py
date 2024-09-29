# from transformers import AlbertTokenizer, AlbertModel
# import pandas as pd
# import torch
# from pydub import AudioSegment
# import numpy as np
# import io

# # Load pre-trained ALBERT model and tokenizer
# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# model = AlbertModel.from_pretrained('albert-base-v2')

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# model_name = "VietAI/envit5-translation"
# tokenizer_vie = AutoTokenizer.from_pretrained(model_name)
# model_vie = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# from tensorflow.keras.models import load_model
# model_loaded = load_model('D:/Research/Eureka2024/Codedemoweb/Code/deep_learning.h5')

# import speech_recognition as sr
# recognizer = sr.Recognizer()

# def extract_features(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Extract features from the last layer
#     return features

# def predict_text_recording(inputs):
#   outputs = model_vie.generate(tokenizer_vie(inputs, return_tensors="pt", padding=True).input_ids.to('cpu'), max_length=512)
#   outputs = (tokenizer_vie.batch_decode(outputs, skip_special_tokens=True))
#   outputs = outputs[0][4:]
#   print(outputs)

#   text = outputs
#   feature = extract_features(text)
#   input_data = [feature]
#   input_data = np.array(input_data)

#   res = model_loaded.predict(input_data)
#   return res[0]

# def decode_audio(audio_bytes):
#   try:
#         print("decode_audio: ",audio_bytes[:20])
#         # Đọc dữ liệu âm thanh từ bytes
#         # audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
#         audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp4")
        
#         # Chuyển đổi âm thanh thành định dạng WAV (vì speech_recognition không hỗ trợ MP3)
#         wav_audio_bytes = audio_segment.export(format="wav").read()
        
#         # Sử dụng speech_recognition để nhận dạng văn bản từ dữ liệu âm thanh WAV
#         recognizer = sr.Recognizer()
#         with sr.AudioFile(io.BytesIO(wav_audio_bytes)) as source:
#             audio_data = recognizer.record(source)
        
#         return audio_data
#   except Exception as e:
#         print("Error decoding audio:", e)
#         return None
    
#   return audio_data

# def predict_recording(input):
#   try:
#     audio_data = decode_audio(input)
#     text = recognizer.recognize_google(audio_data, language='vi-VN')
#     res = predict_text_recording(text)
#     return res
#   except sr.UnknownValueError:
#     print("Could not understand audio")
#     return "error"
#   except sr.RequestError as e:
#     print("Could not request results from Google Speech Recognition service; {0}".format(e))
#     return "gg-error"

# from transformers import AlbertTokenizer, AlbertModel, AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
# import numpy as np
# import io
# import speech_recognition as sr
# from pydub import AudioSegment
# from tensorflow.keras.models import load_model

# # Load pre-trained ALBERT model and tokenizer
# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# model = AlbertModel.from_pretrained('albert-base-v2')

# # Load translation model and tokenizer
# model_name = "VietAI/envit5-translation"
# tokenizer_vie = AutoTokenizer.from_pretrained(model_name)
# model_vie = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # Load deep learning model
# model_loaded = load_model('D:/Research/Eureka2024/Codedemoweb/Code/deep_learning.h5')

# # Initialize speech recognizer
# recognizer = sr.Recognizer()

# def extract_features(text):
#     """Extract features from text using ALBERT model."""
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Extract features from the last layer
#     return features

# def predict_text_recording(inputs):
#     """Translate text and predict using the deep learning model."""
#     outputs = model_vie.generate(tokenizer_vie(inputs, return_tensors="pt", padding=True).input_ids.to('cpu'), max_length=512)
#     outputs = tokenizer_vie.batch_decode(outputs, skip_special_tokens=True)
    
#     # Check if outputs is non-empty
#     if len(outputs) == 0 or len(outputs[0]) < 5:  # Ensure there is enough data to slice
#         print("Error: Translated text is empty or too short.")
#         return "error"
    
#     outputs = outputs[0][4:]  # Extract translated content
#     print(f"Translated Text: {outputs}")

#     # Feature extraction and model prediction
#     feature = extract_features(outputs)
#     input_data = np.array([feature])
    
#     res = model_loaded.predict(input_data)
#     return res[0]


# def decode_audio(audio_bytes, file_format="mp4"):
#     """Decode audio bytes and convert to wav format."""
#     try:
#         print("decode_audio: ", audio_bytes[:20])  # Debug: Display the first 20 bytes of audio
        
#         # Ensure correct format is passed to the decoder
#         if file_format not in ["mp3", "mp4"]:
#             raise ValueError("Unsupported audio format")
        
#         audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=file_format)
        
#         if len(audio_segment) == 0:
#             raise ValueError("Decoded audio segment is empty")
        
#         wav_audio = io.BytesIO()
#         audio_segment.export(wav_audio, format="wav")  # Export to wav format
#         wav_audio.seek(0)  # Reset pointer to the start
#         return wav_audio
#     except Exception as e:
#         print(f"Error decoding audio: {e}")
#         return None
    
# def predict_recording(input, file_format="mp4"):
#     """Process the audio, recognize text, and predict vishing."""
#     try:
#         wav_audio = decode_audio(input, file_format)
#         if wav_audio is None:
#             return "error"
        
#         # Recognize speech
#         with sr.AudioFile(wav_audio) as source:
#             audio_data = recognizer.record(source)
#         text = recognizer.recognize_google(audio_data, language='vi-VN')
        
#         # Predict based on recognized text
#         res = predict_text_recording(text)
#         return res
#     except sr.UnknownValueError:
#         print("Could not understand audio")
#         return "error"
#     except sr.RequestError as e:
#         print(f"Error with Google Speech Recognition service: {e}")
#         return "gg-error"
#     except Exception as e:
#         print(f"General error: {e}")
#         return "error"

from transformers import AlbertTokenizer, AlbertModel, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
import io
import speech_recognition as sr
from pydub import AudioSegment
from tensorflow.keras.models import load_model

# Load pre-trained ALBERT model and tokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained('albert-base-v2')

# Load translation model and tokenizer
model_name = "VietAI/envit5-translation"
tokenizer_vie = AutoTokenizer.from_pretrained(model_name)
model_vie = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load deep learning model
model_loaded = load_model('D:/Research/Eureka2024/Codedemoweb/Code/deep_learning.h5')

# Initialize speech recognizer
recognizer = sr.Recognizer()

def extract_features(text):
    """Extract features from text using ALBERT model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Extract features from the last layer
    return features

def predict_text_recording(inputs):
    """Translate text and predict using the deep learning model."""
    outputs = model_vie.generate(tokenizer_vie(inputs, return_tensors="pt", padding=True).input_ids.to('cpu'), max_length=512)
    outputs = tokenizer_vie.batch_decode(outputs, skip_special_tokens=True)
    
    # Check if outputs is non-empty
    if len(outputs) == 0 or len(outputs[0]) < 5:  # Ensure there is enough data to slice
        print("Error: Translated text is empty or too short.")
        return "error"
    
    outputs = outputs[0][4:]  # Extract translated content
    print(f"Translated Text: {outputs}")

    # Feature extraction and model prediction
    feature = extract_features(outputs)
    input_data = np.array([feature])
    
    res = model_loaded.predict(input_data)
    return res[0]

def decode_audio(audio_bytes):
    """Decode audio bytes and convert to wav format."""
    try:
        print("decode_audio: ", audio_bytes[:20])  # Debug: Display the first 20 bytes of audio
        
        # Thử cả mp3 và mp4
        for file_format in ["mp3", "mp4"]:
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=file_format)
                if len(audio_segment) > 0:  # Kiểm tra nếu âm thanh không rỗng
                    # Chuyển đổi sang định dạng wav
                    wav_audio = io.BytesIO()
                    audio_segment.export(wav_audio, format="wav")
                    wav_audio.seek(0)
                    return wav_audio
            except Exception as e:
                print(f"Error decoding audio with format {file_format}: {e}")

        # Nếu không có định dạng nào thành công
        raise ValueError("Unsupported audio format or empty audio")

    except Exception as e:
        print(f"Error decoding audio: {e}")
        return None

def predict_recording(input):
    """Process the audio, recognize text, and predict vishing."""
    try:
        wav_audio = decode_audio(input)
        if wav_audio is None:
            return "error"
        
        # Recognize speech
        with sr.AudioFile(wav_audio) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='vi-VN')
        
        # Predict based on recognized text
        res = predict_text_recording(text)
        return res
    except sr.UnknownValueError:
        print("Could not understand audio")
        return "error"
    except sr.RequestError as e:
        print(f"Error with Google Speech Recognition service: {e}")
        return "gg-error"
    except Exception as e:
        print(f"General error: {e}")
        return "error"


