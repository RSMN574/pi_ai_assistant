import time
from pathlib import Path

import sherpa_onnx
import soundfile as sf

BASE = Path.home() / "ai_assistant/models/tts/kokoro-int8-multi-lang-v1_1"

model_path = BASE / "model.int8.onnx"
voices_path = BASE / "voices.bin"
tokens_path = BASE / "tokens.txt"
data_dir = BASE / "espeak-ng-data"
dict_dir = BASE / "dict"
lexicon = f"{BASE}/lexicon-us-en.txt,{BASE}/lexicon-zh.txt"

for path in (model_path, voices_path, tokens_path, data_dir, dict_dir):
    if not path.exists():
        raise FileNotFoundError(f"缺少模型檔案或目錄：{path}")

print("sherpa-onnx 版本：", getattr(sherpa_onnx, "__version__", "未知"))
print("正在載入 Kokoro int8 模型……", flush=True)

kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
    model=str(model_path),
    voices=str(voices_path),
    tokens=str(tokens_path),
    data_dir=str(data_dir),
    dict_dir=str(dict_dir),
    lexicon=lexicon,
    length_scale=1.0,
)

model_config = sherpa_onnx.OfflineTtsModelConfig(
    kokoro=kokoro,
    num_threads=2,
    provider="cpu",
)

config = sherpa_onnx.OfflineTtsConfig(
    model=model_config,
    max_num_sentences=1,
)

if not config.validate():
    raise RuntimeError("Kokoro 模型設定驗證失敗")

tts = sherpa_onnx.OfflineTts(config)

print("模型載入成功！", flush=True)
print("取樣率：", tts.sample_rate)
print("聲線數量：", tts.num_speakers)

text = "你好，我是你的樹莓派離線語音助手。"

print("正在合成中文語音……", flush=True)
start = time.perf_counter()

audio = tts.generate(text, sid=0, speed=1.0)

elapsed = time.perf_counter() - start
if len(audio.samples) == 0:
    raise RuntimeError("模型沒有產生音訊")

output = Path.home() / "ai_assistant/kokoro_test.wav"
sf.write(str(output), audio.samples, audio.sample_rate)

duration = len(audio.samples) / audio.sample_rate
print(f"音訊已儲存：{output}")
print(f"合成耗時：{elapsed:.2f} 秒")
print(f"音訊長度：{duration:.2f} 秒")
print(f"RTF：{elapsed / duration:.2f}")
