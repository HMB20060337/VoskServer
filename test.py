import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Model yükleniyor...")
config = XttsConfig()
config.load_json("C:/Users/a/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="C:/Users/a/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2/")
model.cuda()

print("Konuşmacı gizli katmanları hesaplanıyor...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["assets/r.wav"])

print("Ses üretiliyor...")
out = model.inference(
"Merhaba, ben Copilot. Sizinle konuşmaktan çok mutluyum.",
"tr",
gpt_cond_latent,
speaker_embedding,
temperature=0.1, # Sesin doğallığını artırır
speed=1.25 # Sesin hızını artırır
)
torchaudio.save("xtts.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)