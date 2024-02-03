from vosk import Model, KaldiRecognizer, SetLogLevel
import os
import json
from pytube import YouTube
from pydub import AudioSegment
import subprocess
from language_tool_python import LanguageTool
import torchaudio

from flask import Flask, request, jsonify

from transformers import AutoTokenizer, MarianMTModel

import torch
from TTS.api import TTS
import re
from deepmultilingualpunctuation import PunctuationModel

SetLogLevel(0)

app = Flask(__name__)

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up Vosk model
model = Model('assets/models/vosk-model-en-us-0.22')


# Initialize TTS model
print("Model yükleniyor...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('cpu')
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-tr")
TranslateModel = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-tr")

punctuationModel = PunctuationModel()

tool = LanguageTool('en-US')

# API endpoint
@app.route('/reverse', methods=['POST'])
def reverse_text():
    # Get the request data
    data = request.get_json()
    # Get the YouTube URL and download the audio
    youtube_url = data.get('url')
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    stream.download(output_path='downloads', filename='audio.mp4')

    # Convert audio to WAV format
    audio_path = 'downloads/audio.mp4'
    sound = AudioSegment.from_file(audio_path, format='mp4')
    sound.export('downloads/audio.wav', format='wav')

    # Perform speech recognition and translation
    text = ''
    rec = KaldiRecognizer(model, 16000)
    with subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i", 'downloads/audio.wav', "-ar", str(16000) , "-ac", "1", "-f", "s16le", "-"], stdout=subprocess.PIPE) as process:
        outputCounter = 1
        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text += res['text'] + " "
    tool_results = tool.correct(text)
    result = punctuationModel.restore_punctuation(tool_results)
    print('****--'+result+'--****')
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', result)
    output_counter=1
    respond = ''
    maksimum_karakter_sayisi = 220

    split_sentences = []

    gecici_metin = ""
    for sentence in sentences:
    # Eğer geçici metin ile bu cümlenin toplam karakter sayısı maksimum sınırdan küçükse
        if len(gecici_metin) + len(sentence) <= maksimum_karakter_sayisi:
        # Geçici metne cümleyi ekle
            gecici_metin += sentence + " "
        else:
        # Geçici metni parçalara böl ve listeye ekle
            split_sentences.append(gecici_metin.strip())
        # Yeni bir geçici metin başlat
            gecici_metin = sentence + " "
    
    split_sentences.append(gecici_metin.strip())

    for sentence in split_sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        translation = TranslateModel.generate(**inputs)
        tr = tokenizer.batch_decode(translation, skip_special_tokens=True)
        respond = respond+' '+tr[0]
        print("Ses üretiliyor... "+f"output{output_counter}.wav")
        tts.tts_to_file(text=tr[0], speaker_wav="assets/r.wav", language="tr", file_path=f"output{output_counter}.wav")
        output_counter+=1
    return jsonify({'text': respond})

if __name__ == '__main__':
    app.run(debug=True)
