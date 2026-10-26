import sys
import queue

import sounddevice as sd
import sherpa_onnx

MODEL_DIR = (
    "sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30"
)

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600
NUM_THREADS = 2

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"\n音訊狀態：{status}", file=sys.stderr)

    audio_queue.put(indata[:, 0].copy())


def main():
    model_path = f"{MODEL_DIR}/model.int8.onnx"
    tokens_path = f"{MODEL_DIR}/tokens.txt"

    import os

    for path in (model_path, tokens_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到模型檔案：{path}")

    print("正在載入中文 CTC 模型……")

    recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=tokens_path,
        model=model_path,
        num_threads=NUM_THREADS,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
    )

    stream = recognizer.create_stream()

    print("模型載入成功。")
    print("開始即時中文語音辨識，按 Ctrl+C 結束。")

    last_text = ""

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
        ):
            while True:
                samples = audio_queue.get()

                stream.accept_waveform(SAMPLE_RATE, samples)

                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                text = recognizer.get_result(stream)

                if text and text != last_text:
                    print(f"\r辨識結果：{text}", end="", flush=True)
                    last_text = text

    except KeyboardInterrupt:
        print("\n測試結束。")
    except Exception as exc:
        print(f"\n執行失敗：{exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
