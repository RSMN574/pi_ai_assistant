import argparse
import time

from llm import OllamaLLM
from tts import KokoroTTS


EXIT_WORDS = {"退出", "結束助理", "停止助理", "關閉助理"}


def main():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 離線中文語音助手"
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="使用麥克風進行語音對話",
    )
    parser.add_argument(
        "--no-speak",
        action="store_true",
        help="只顯示 AI 回答，不播放語音",
    )
    args = parser.parse_args()

    print("=" * 40)
    print("Raspberry Pi 中文 AI 助手")
    print("AI：Ollama | 語音：Kokoro | 辨識：sherpa-onnx")
    print("=" * 40)

    llm = OllamaLLM()
    tts = None
    stt = None

    try:
        if not args.no_speak:
            tts = KokoroTTS()

        if args.voice:
            from stt import SherpaSTT

            stt = SherpaSTT()
            stt.start()
            print("\n語音模式已啟動，請直接說話。")
            print("說「退出」結束程式。")
        else:
            print("\n文字模式已啟動。")
            print("輸入「退出」結束程式。")

        while True:
            if args.voice:
                text = stt.get_text(timeout=0.5)
                if not text:
                    continue
            else:
                try:
                    text = input("\n你：").strip()
                except EOFError:
                    break

                if not text:
                    continue

            if text in EXIT_WORDS:
                print("助理：再見！")
                if tts:
                    tts.speak("再見，祝你有美好的一天。")
                break

            start = time.perf_counter()
            print("正在思考……", flush=True)
            answer = llm.ask(text)
            print(f"助理：{answer}")
            print(f"AI 回答耗時：{time.perf_counter() - start:.2f} 秒")

            if tts and answer:
                try:
                    result = tts.speak(answer)
                    print(
                        "語音合成耗時："
                        f"{result['elapsed']:.2f} 秒；"
                        f"音訊長度：{result['duration']:.2f} 秒；"
                        f"RTF：{result['rtf']:.2f}"
                    )
                except Exception as exc:
                    print(f"語音播放失敗：{exc}")

    except KeyboardInterrupt:
        print("\n收到結束指令。")
    finally:
        if stt:
            stt.close()

    print("語音助手已結束。")


if __name__ == "__main__":
    main()
