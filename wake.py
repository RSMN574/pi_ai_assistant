"""OpenWakeWord 喚醒詞偵測模組。"""
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from openwakeword.model import Model


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    BASE_DIR / "models" / "wakeword"
    / "Hey_piper_20260907_230125.onnx"
)
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280


class WakeWordDetector:
    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        threshold=0.75,
        cooldown=3.0,
        device=None,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"找不到喚醒詞模型：{self.model_path}"
            )

        self.threshold = float(threshold)
        self.cooldown = float(cooldown)
        self.device = device
        self.audio_queue = queue.Queue(maxsize=20)
        self.audio_stream = None
        self.last_trigger = 0.0

        self.model = Model(
            wakeword_models=[str(self.model_path)],
            inference_framework="onnx",
        )
        self.model_name = self.model_path.stem

    def _audio_callback(self, indata, frames, time_info, status):
        samples = indata[:, 0].copy()
        try:
            self.audio_queue.put_nowait(samples)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(samples)
            except queue.Empty:
                pass
            except queue.Full:
                pass

    def start(self):
        if self.audio_stream is not None:
            return

        self._clear_queue()
        self.audio_stream = sd.InputStream(
            device=self.device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=self._audio_callback,
        )
        self.audio_stream.start()

    def _clear_queue(self):
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def wait_for_wake(self, timeout=None, show_scores=False):
        """等待喚醒詞；timeout=None 表示持續等待。"""
        if self.audio_stream is None:
            self.start()

        started = time.monotonic()

        while True:
            if (
                timeout is not None
                and time.monotonic() - started >= timeout
            ):
                return False

            try:
                audio = self.audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            prediction = self.model.predict(
                np.asarray(audio, dtype=np.int16)
            )
            score = float(prediction.get(self.model_name, 0.0))

            if show_scores:
                print(
                    f"\r喚醒分數：{score:.3f}   ",
                    end="",
                    flush=True,
                )

            now = time.monotonic()
            if (
                score >= self.threshold
                and now - self.last_trigger >= self.cooldown
            ):
                self.last_trigger = now
                if show_scores:
                    print()
                print(f"偵測到 Hey Piper，分數：{score:.3f}")
                return True

    def stop(self):
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        self._clear_queue()

    def close(self):
        self.stop()


if __name__ == "__main__":
    detector = WakeWordDetector()

    try:
        print("喚醒詞待機測試")
        print("請說：Hey Piper")
        print("按 Ctrl+C 結束")

        detector.start()
        while True:
            detector.wait_for_wake(show_scores=True)
            print("喚醒成功，3 秒後繼續等待。")
            detector.stop()
            time.sleep(3)
            detector.start()

    except KeyboardInterrupt:
        print("\n測試結束")
    finally:
        detector.close()
