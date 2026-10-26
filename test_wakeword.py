
import time
import numpy as np
import sounddevice as sd
from pathlib import Path
from openwakeword.model import Model

MODEL_PATH = (
    Path.home()
    / "ai_assistant/models/wakeword/"
    / "Hey_piper_20260907_230125.onnx"
)

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
THRESHOLD = 0.5

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到模型：{MODEL_PATH}")

    model = Model(
        wakeword_models=[str(MODEL_PATH)],
        inference_framework="onnx",
    )

    print("Hey Piper 即時喚醒測試")
    print("請說出：Hey Piper")
    print("按 Ctrl+C 結束")
    print("-" * 40)

    last_trigger = 0.0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        ) as stream:
            while True:
                audio, overflowed = stream.read(CHUNK_SIZE)

                if overflowed:
                    print("提示：音訊緩衝區曾溢位")

                audio = np.squeeze(audio)
                prediction = model.predict(audio)

                for name, score in prediction.items():
                    score = float(score)

                    print(
                        f"\r{name}: {score:.4f}   ",
                        end="",
                        flush=True,
                    )

                    now = time.monotonic()

                    if score >= THRESHOLD and now - last_trigger >= 2:
                        last_trigger = now
                        print(f"\n偵測到 Hey Piper！分數：{score:.4f}")

    except KeyboardInterrupt:
        print("\n測試結束")

if __name__ == "__main__":
    main()
