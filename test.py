import os
# Flask paketini kurun ve bir app nesnesi oluşturun
from flask import Flask, request, send_file
app = Flask(__name__)

# Bir route fonksiyonu tanımlayın ve isteği argüman olarak alın
@app.route("/dub", methods=["GET"])
def dub():
    # İsteğin bir YouTube video linki olduğunu doğrulayın
    url = request.args.get("url")
    if not url or not url.startswith("https://www.youtube.com/"):
        return "Geçersiz URL"

    # İsteği youtube_dl paketi ile işleyin ve videoyu indirin
    from pytube import YouTube

    yt = YouTube(url)
    stream = yt.streams.first()
    stream.download(output_path='downloads', filename='video.mp4')

    # Videoyu ses ve görüntü olarak ayırın
    # ffmpeg -i video.mp4 -c:v copy -an video_only.mp4
    # ffmpeg -i video.wav -c:a copy -vn audio_only.wav
    # Subprocess modülünü içe aktarın
    import subprocess
    # Ffmpeg komutlarını birer liste olarak oluşturun
    command1 = ["ffmpeg", "-i", "downloads/video.mp4", "-c:v", "copy", "-an", "video_only.mp4","-y"]

    # Subprocess.run fonksiyonunu kullanarak, komut listelerini çalıştırın
    subprocess.run(command1)


    # Ses dosyasını Vosk ile transkribe edin ve altyazı dosyası olarak kaydedin
    # Bu adımı daha önce yaptıysanız atlayabilirsiniz
    from vosk import Model, KaldiRecognizer
    import wave
    import json

    model = Model('assets/models/vosk-model-en-us-0.22')
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    subtitles = []
    start = 0
    end = 0
    text = ""
    with subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i",
                            "downloads/video.mp4",
                            "-ar", str(16000) , "-ac", "1", "-f", "s16le", "-"],
                            stdout=subprocess.PIPE).stdout as stream:
        with open("subtitles.srt", "w") as f:
            f.write(rec.SrtResult(stream))

    # Altyazı dosyasını LanguageTool ile düzeltin
    from language_tool_python import LanguageTool
    tool = LanguageTool('en-US')
    with open("subtitles.srt", "r") as f:
        lines = f.readlines()
    with open("subtitles_corrected.srt", "w") as f:
        for line in lines:
            if line.strip().isdigit() or "-->" in line.strip():
                f.write(line)
            elif line.strip():
                if(tool.check(line)):
                    corrected = tool.correct(line)
                    f.write(corrected + "\n")
            else:
                f.write("\n")

    # Altyazı dosyasına punctuationmodel ile noktalama işaretleri ekleyin
    from deepmultilingualpunctuation import PunctuationModel
    model = PunctuationModel()
    with open("subtitles_corrected.srt", "r") as f:
        lines = f.readlines()
    with open("subtitles_punctuated.srt", "w") as f:
        for line in lines:
            if line.strip().isdigit() or "-->" in line.strip():
                f.write(line)
            elif line.strip():
                punctuated = model.restore_punctuation(line)
                f.write(punctuated + "\n")
            else:
                f.write("\n")

    # Altyazı dosyasını çeviri modeli ile Türkçe'ye çevirin
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-tr")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-tr")
    with open("subtitles_corrected.srt", "r") as f:
        lines = f.readlines()
    with open("subtitles_translated.srt", "w") as f:
        for line in lines:
            if line.strip().isdigit() or "-->" in line.strip():
                f.write(line)
            elif line.strip():
                inputs = tokenizer.encode(line, return_tensors="pt")
                outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
                translated = tokenizer.decode(outputs[0],skip_special_tokens=True)
                f.write(translated + "\n")
            else:
                f.write("\n")

    # Altyazı dosyasını coqui ai tts ile seslendirin ve ses dosyaları olarak kaydedin
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    command3args = []
    offsets = []
    def get_wav_duration(file_path):
    # WAV dosyasını aç
        with wave.open(file_path, 'rb') as wav_file:
        # Dosyanın uzunluğunu saniye cinsinden al
            duration_in_secs = wav_file.getnframes() / wav_file.getframerate()
        return duration_in_secs*1000
    import shutil
    shutil.rmtree(os.getcwd()+"\outputs")
    os.mkdir(os.getcwd()+"/outputs")
    with open("subtitles_translated.srt", "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "-->" in line.strip():
            times = line.split(" --> ")
            import datetime
            time = datetime.datetime.strptime(times[0], "%H:%M:%S,%f")
            offsets.append(time.hour*3600+time.minute*60+time.second+time.microsecond/1000000)
        elif line.strip() and not line.strip().isdigit() and not "-->" in line.strip():
            tts.tts_to_file(speed=1.1, text = line,speaker_wav="assets/sample.mp3",language="tr",file_path=f"outputs/output{i}.wav")
            command3args.append(f"-i") # -i seçeneğini listeye ekle
            command3args.append(f"outputs/output{i}.wav") # ses dosyasının adını listeye ekle
    command3args.append("-filter_complex")
    output_folder = "outputs"
    args=''
    outputs = os.listdir(output_folder)
    for x in range(len(outputs)):
        if (x==0):
            args += f'[{x+1}:a]adelay={int(offsets[x]*1000)}|{int(offsets[x]*1000)}[a{x+1}];'
        else:
            off= get_wav_duration("outputs/"+outputs[x-1])
            args += f'[{x+1}:a]adelay={int(off)-1000+int(offsets[x]*1000)}|{int(off)-1000+int(offsets[x]*1000)}[a{x+1}];'
    for x in range(len(os.listdir(output_folder))):
        args += f'[a{x+1}]'
    command3args.append(args+f"amix=inputs={len(os.listdir(output_folder))}[a]")
    command3args.append("-map")
    command3args.append("0:v")
    command3args.append("-map")
    command3args.append("[a]")
    command3 = ["ffmpeg", "-i", "video_only.mp4"]+command3args+["-c:v","libx264","output.mp4","-y"]
    print(command3)
    subprocess.call(command3)
    # Ses dosyalarını videoya ekleyin ve dublajlı videoyu oluşturun
    # ffmpeg -i video_only.mp4 -i audio-*.wav -c:v
    return send_file("output.mp4", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)