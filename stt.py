import os
import queue
import time

import numpy as np
import sounddevice as sd
import sherpa_onnx


MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30",
)

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600
NUM_THREADS = 2

# 自動斷句設定
SILENCE_SECONDS = 0.45
MIN_SPEECH_SECONDS = 0.20
ENERGY_THRESHOLD = 0.012
MAX_UTTERANCE_SECONDS = 15


class SherpaSTT:
    def __init__(self, device=None):
        model_path = os.path.join(MODEL_DIR, "model.int8.onnx")
        tokens_path = os.path.join(MODEL_DIR, "tokens.txt")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"找不到模型：{model_path}")
        if not os.path.isfile(tokens_path):
            raise FileNotFoundError(f"找不到詞元表：{tokens_path}")

        print("正在載入 sherpa-onnx 中文模型……")

        self.recognizer = (
            sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                tokens=tokens_path,
                model=model_path,
                num_threads=NUM_THREADS,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                provider="cpu",
            )
        )

        self.device = device
        self.audio_queue = queue.Queue()
        self.audio_stream = None
        self.stream = None

        self.last_partial = ""
        self.last_emitted = ""
        self.speech_started = False
        self.speech_seconds = 0.0
        self.silence_seconds = 0.0
        self.utterance_seconds = 0.0

        print("中文模型載入成功。")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"\n音訊狀態：{status}")
        self.audio_queue.put(indata[:, 0].copy())

    def start(self):
        if self.audio_stream is not None:
            return

        self.stream = self.recognizer.create_stream()

        self.audio_stream = sd.InputStream(
            device=self.device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=self._audio_callback,
        )

        self.audio_stream.start()
        print("麥克風已開啟。")

    def _reset_utterance(self):
        self.stream = self.recognizer.create_stream()
        self.last_partial = ""
        self.speech_started = False
        self.speech_seconds = 0.0
        self.silence_seconds = 0.0
        self.utterance_seconds = 0.0

    def get_text(self, timeout=0.5):
        """等待完整句子；尚未完成時回傳空字串。"""
        if self.audio_stream is None:
            self.start()

        try:
            samples = self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return ""

        duration = len(samples) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))

        self.stream.accept_waveform(SAMPLE_RATE, samples)

        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        partial = self.recognizer.get_result(self.stream).strip()

        if rms >= ENERGY_THRESHOLD:
            self.speech_started = True
            self.speech_seconds += duration
            self.silence_seconds = 0.0
        elif self.speech_started:
            self.silence_seconds += duration

        if self.speech_started:
            self.utterance_seconds += duration

        if partial and partial != self.last_partial:
            self.last_partial = partial
            print(f"\r辨識中：{partial}   ", end="", flush=True)

        silence_finished = (
            self.speech_started
            and self.speech_seconds >= MIN_SPEECH_SECONDS
            and self.silence_seconds >= SILENCE_SECONDS
        )

        maximum_reached = (
            self.speech_started
            and self.utterance_seconds >= MAX_UTTERANCE_SECONDS
        )

        if silence_finished or maximum_reached:
            final_text = partial.strip()
            self._reset_utterance()

            if final_text:
                print(f"\n完整句子：{final_text}")
                return final_text

        return ""

    def reset(self):
        self._reset_utterance()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def stop(self):
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        self.stream = None

    def close(self):
        self.stop()


if __name__ == "__main__":
    stt = SherpaSTT()

    try:
        stt.start()
        print("請說一句中文，停止說話約 1 秒後輸出完整句子。")
        print("按 Ctrl+C 結束。")

        while True:
            text = stt.get_text(timeout=0.5)
            if text:
                print(f"[最終辨識] {text}")

    except KeyboardInterrupt:
        print("\n測試結束。")
    finally:
        stt.close()
