import os
import tempfile
import subprocess
import time
from pathlib import Path

import sherpa_onnx
import soundfile as sf


BASE = (
    Path(__file__).resolve().parent
    / "models/tts/kokoro-int8-multi-lang-v1_1"
)

MODEL_PATH = BASE / "model.int8.onnx"
VOICES_PATH = BASE / "voices.bin"
TOKENS_PATH = BASE / "tokens.txt"

LEXICON = ",".join([
    str(BASE / "lexicon-us-en.txt"),
    str(BASE / "lexicon-zh.txt"),
])

NUM_THREADS = int(os.getenv("TTS_NUM_THREADS", "4"))


class KokoroTTS:
    """使用 sherpa-onnx Kokoro int8 的離線語音合成。"""

    def __init__(self, sid=0):
        for path in (MODEL_PATH, VOICES_PATH, TOKENS_PATH):
            if not path.exists():
                raise FileNotFoundError(f"找不到 Kokoro 檔案：{path}")

        print(f"載入 Kokoro TTS（{NUM_THREADS} 執行緒）……", flush=True)

        kokoro_config = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(MODEL_PATH),
            voices=str(VOICES_PATH),
            tokens=str(TOKENS_PATH),
            lexicon=LEXICON,
            length_scale=1.0,
        )

        model_config = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=kokoro_config,
            num_threads=NUM_THREADS,
            provider="cpu",
        )

        config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            max_num_sentences=1,
        )

        if not config.validate():
            raise RuntimeError("Kokoro TTS 設定驗證失敗")

        self.tts = sherpa_onnx.OfflineTts(config)
        self.sid = sid

        print(
            f"Kokoro 已載入：取樣率 {self.tts.sample_rate} Hz，"
            f"聲線數量 {self.tts.num_speakers}",
            flush=True,
        )

    def synthesize(self, text, speed=1.0):
        """合成語音並回傳音訊物件及耗時。"""
        text = text.strip()
        text = text.replace("「", "，").replace("」", "，")
        text = text.replace("『", "，").replace("』", "，")
        if not text:
            raise ValueError("語音文字不可為空")

        start = time.perf_counter()
        audio = self.tts.generate(
            text,
            sid=self.sid,
            speed=speed,
        )
        elapsed = time.perf_counter() - start

        if len(audio.samples) == 0:
            raise RuntimeError("Kokoro 沒有產生音訊")

        return audio, elapsed

    def save(self, text, path, speed=1.0):
        """合成語音並儲存成 WAV。"""
        audio, elapsed = self.synthesize(text, speed)
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(path), audio.samples, audio.sample_rate)

        duration = len(audio.samples) / audio.sample_rate
        return {
            "path": str(path),
            "elapsed": elapsed,
            "duration": duration,
            "rtf": elapsed / duration if duration else 0,
        }

    def speak(self, text, speed=1.0):
        """合成後透過 aplay 播放，播放完成才返回。"""
        audio, elapsed = self.synthesize(text, speed)

        with tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="assistant_", delete=False
        ) as file:
            wav_path = Path(file.name)

        try:
            sf.write(
                str(wav_path),
                audio.samples,
                audio.sample_rate,
            )
            subprocess.run(
                ["aplay", "-q", str(wav_path)],
                check=True,
            )
        finally:
            wav_path.unlink(missing_ok=True)

        duration = len(audio.samples) / audio.sample_rate
        return {
            "elapsed": elapsed,
            "duration": duration,
            "rtf": elapsed / duration if duration else 0,
        }


if __name__ == "__main__":
    tts = KokoroTTS()
    text = "你好，我是你的樹莓派離線語音助手。"

    result = tts.save(
        text,
        Path.home() / "ai_assistant/kokoro_test.wav",
    )

    print(f"音訊：{result['path']}")
    print(f"合成耗時：{result['elapsed']:.2f} 秒")
    print(f"音訊長度：{result['duration']:.2f} 秒")
    print(f"RTF：{result['rtf']:.2f}")
