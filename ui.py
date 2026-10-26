import sys
import traceback

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QLineEdit, QTextEdit, QVBoxLayout, QHBoxLayout,
    QFrame, QCheckBox
)

from llm import OllamaLLM
from tts import KokoroTTS
from wake import WakeWordDetector


class InitWorker(QThread):
    ready = Signal(object, object, object)
    error = Signal(str)
    status = Signal(str)

    def run(self):
        stt = None
        try:
            self.status.emit("正在載入 AI……")
            llm = OllamaLLM()

            self.status.emit("正在載入語音合成……")
            tts = KokoroTTS()

            self.status.emit("正在預先載入中文語音辨識……")
            from stt import SherpaSTT
            stt = SherpaSTT()

            self.ready.emit(llm, tts, stt)
            stt = None  # 所有權交給主視窗
        except Exception:
            if stt is not None:
                stt.close()
            self.error.emit(traceback.format_exc())


class WakeWorker(QThread):
    status = Signal(str)

    def __init__(self):
        super().__init__()
        self.triggered = False

    def run(self):
        detector = None
        try:
            detector = WakeWordDetector(threshold=0.75, cooldown=3)
            detector.start()
            self.status.emit("待機中：請說 Hey Piper")

            while not self.isInterruptionRequested():
                if detector.wait_for_wake(timeout=0.5):
                    self.triggered = True
                    break
        except Exception:
            self.status.emit("喚醒模組發生錯誤")
            traceback.print_exc()
        finally:
            if detector is not None:
                detector.close()


class VoiceWorker(QThread):
    recognized = Signal(str)
    status = Signal(str)
    error = Signal(str)

    def __init__(self, stt):
        super().__init__()
        self.stt = stt

    def run(self):
        try:
            # 重用已載入的模型，只重新開啟麥克風。
            self.stt.start()
            self.stt.reset()
            self.status.emit("請說出你的問題……")

            while not self.isInterruptionRequested():
                text = self.stt.get_text(timeout=0.5)
                if text.strip():
                    self.recognized.emit(text.strip())
                    return
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.stt.stop()


class AssistantWorker(QThread):
    answer_ready = Signal(str)
    status = Signal(str)
    error = Signal(str)

    def __init__(self, llm, tts, prompt, enable_tts):
        super().__init__()
        self.llm = llm
        self.tts = tts
        self.prompt = prompt
        self.enable_tts = enable_tts

    def run(self):
        try:
            self.status.emit("AI 正在思考……")
            answer = self.llm.ask(self.prompt)
            if not answer:
                answer = "目前沒有取得回答，請稍後再試。"

            self.answer_ready.emit(answer)

            if self.enable_tts:
                self.status.emit("正在朗讀回答……")
                self.tts.speak(answer)
        except Exception as exc:
            self.error.emit(str(exc))


class AssistantWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.llm = None
        self.tts = None
        self.stt = None
        self.init_worker = None
        self.wake_worker = None
        self.voice_worker = None
        self.assistant_worker = None

        self.busy = False
        self.wake_enabled = True
        self.voice_result = ""

        self.setWindowTitle("小派 AI 助理")
        self.resize(480, 320)
        self.setMinimumSize(400, 280)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #10141c;
                color: #edf3ff;
                font-family: "Noto Sans CJK TC", sans-serif;
                font-size: 13px;
            }
            QFrame#header {
                background: #1a2333;
                border: 1px solid #2b394e;
                border-radius: 12px;
            }
            QLabel#title {
                color: white;
                font-size: 19px;
                font-weight: bold;
            }
            QLabel#status {
                color: #79c7ff;
                font-size: 11px;
            }
            QTextEdit {
                background: #171e29;
                border: 1px solid #2b394e;
                border-radius: 10px;
                padding: 8px;
            }
            QLineEdit {
                background: #202a39;
                border: 1px solid #35465e;
                border-radius: 10px;
                padding: 9px;
                color: white;
            }
            QPushButton {
                background: #2478d4;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 9px 12px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background: #384252;
                color: #a1a9b7;
            }
            QPushButton#secondary {
                background: #26364c;
                border: 1px solid #3b5575;
            }
            QCheckBox { color: #c6d3e5; }
        """)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(7)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)

        title_box = QVBoxLayout()
        title = QLabel("小派 AI 助理")
        title.setObjectName("title")
        subtitle = QLabel("Raspberry Pi · Local AI")
        subtitle.setStyleSheet("color:#9eafc7;font-size:10px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.status_label = QLabel("正在啟動……")
        self.status_label.setObjectName("status")
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignRight |
            Qt.AlignmentFlag.AlignVCenter
        )
        self.status_label.setWordWrap(True)

        header_layout.addLayout(title_box)
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)
        layout.addWidget(header)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setHtml(
            "<p style='color:#79c7ff'><b>小派</b></p>"
            "<p>你好！我已準備好陪你進行本地 AI 對話。</p>"
            "<p style='color:#94a6bf'>"
            "說出 Hey Piper，即可開始語音對話。</p>"
        )
        layout.addWidget(self.chat, 1)

        options = QHBoxLayout()

        self.tts_checkbox = QCheckBox("朗讀回答")
        self.tts_checkbox.setChecked(True)

        self.wake_button = QPushButton("喚醒：開啟")
        self.wake_button.setObjectName("secondary")
        self.wake_button.clicked.connect(self.toggle_wake)

        self.voice_button = QPushButton("語音輸入")
        self.voice_button.setObjectName("secondary")
        self.voice_button.clicked.connect(self.start_manual_voice)

        options.addWidget(self.tts_checkbox)
        options.addStretch()
        options.addWidget(self.wake_button)
        options.addWidget(self.voice_button)
        layout.addLayout(options)

        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("輸入問題……")
        self.input.returnPressed.connect(self.send_text)

        self.send_button = QPushButton("送出")
        self.send_button.clicked.connect(self.send_text)

        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.send_button)
        layout.addLayout(input_row)

        self.set_controls(False)

        self.init_worker = InitWorker()
        self.init_worker.status.connect(self.set_status)
        self.init_worker.ready.connect(self.on_ready)
        self.init_worker.error.connect(self.on_init_error)
        self.init_worker.start()

    def set_status(self, text):
        self.status_label.setText(text)

    def append_message(self, speaker, text, color):
        safe = (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
        )
        self.chat.append(
            f"<p style='color:{color};margin-top:8px'>"
            f"<b>{speaker}</b></p><p>{safe}</p>"
        )
        bar = self.chat.verticalScrollBar()
        bar.setValue(bar.maximum())

    def set_controls(self, enabled):
        self.input.setEnabled(enabled and not self.busy)
        self.send_button.setEnabled(enabled and not self.busy)
        self.voice_button.setEnabled(enabled and not self.busy)
        self.tts_checkbox.setEnabled(enabled and not self.busy)
        self.wake_button.setEnabled(enabled)

    def on_ready(self, llm, tts, stt):
        self.llm = llm
        self.tts = tts
        self.stt = stt
        self.set_controls(True)
        self.start_wake()
        self.input.setFocus()

    def on_init_error(self, message):
        self.set_status("模型載入失敗")
        self.append_message("系統錯誤", message, "#ff8888")

    def start_wake(self):
        if (
            not self.wake_enabled or self.busy
            or self.wake_worker is not None
        ):
            return

        self.wake_worker = WakeWorker()
        self.wake_worker.status.connect(self.set_status)
        self.wake_worker.finished.connect(self.on_wake_finished)
        self.wake_worker.start()
        self.wake_button.setText("喚醒：關閉")
        self.set_status("正在啟動喚醒監聽……")

    def on_wake_finished(self):
        worker = self.wake_worker
        triggered = worker.triggered if worker else False
        self.wake_worker = None

        if triggered and self.wake_enabled:
            self.set_status("已喚醒，請說出問題")
            self.start_voice(from_wake=True)
        elif self.wake_enabled and not self.busy:
            self.set_status("喚醒監聽已停止")

    def stop_wake(self):
        worker = self.wake_worker
        if worker is None:
            return

        worker.requestInterruption()
        worker.wait()
        self.wake_worker = None

    def toggle_wake(self):
        self.wake_enabled = not self.wake_enabled

        if self.wake_enabled:
            self.start_wake()
        else:
            self.stop_wake()
            self.wake_button.setText("喚醒：開啟")
            self.set_status("喚醒功能已關閉")

    def send_text(self):
        text = self.input.text().strip()
        if not text or self.busy or self.llm is None:
            return

        self.input.clear()
        self.append_message("你", text, "#79c7ff")
        self.start_assistant(text)

    def start_manual_voice(self):
        if self.busy:
            return

        # 先關閉喚醒麥克風，再交給 STT 使用。
        self.stop_wake()
        self.start_voice(from_wake=False)

    def start_voice(self, from_wake=False):
        if self.busy:
            return

        self.busy = True
        self.voice_result = ""
        self.set_controls(False)
        self.set_status("正在啟動語音辨識……")

        self.voice_worker = VoiceWorker(self.stt)
        self.voice_worker.status.connect(self.set_status)
        self.voice_worker.recognized.connect(self.on_voice_result)
        self.voice_worker.error.connect(self.show_error)
        self.voice_worker.finished.connect(self.on_voice_finished)
        self.voice_worker.start()

    def on_voice_result(self, text):
        self.voice_result = text.strip()
        if self.voice_result:
            self.append_message("你（語音）", self.voice_result, "#79c7ff")

    def on_voice_finished(self):
        self.voice_worker = None
        text = self.voice_result
        self.busy = False

        if text:
            self.start_assistant(text)
        else:
            self.set_controls(True)
            self.set_status("沒有辨識到語音")
            self.restart_wake_if_needed()

    def start_assistant(self, text):
        if self.busy:
            return

        self.busy = True
        self.set_controls(False)

        self.assistant_worker = AssistantWorker(
            self.llm,
            self.tts,
            text,
            self.tts_checkbox.isChecked(),
        )
        self.assistant_worker.status.connect(self.set_status)
        self.assistant_worker.answer_ready.connect(self.show_answer)
        self.assistant_worker.error.connect(self.show_error)
        self.assistant_worker.finished.connect(self.on_assistant_finished)
        self.assistant_worker.start()

    def show_answer(self, answer):
        self.append_message("小派", answer, "#a5e6ad")

    def show_error(self, message):
        self.append_message("錯誤", message, "#ff8888")
        self.set_status("發生錯誤")

    def on_assistant_finished(self):
        self.assistant_worker = None
        self.busy = False
        self.set_controls(True)
        self.restart_wake_if_needed()
        self.input.setFocus()

    def restart_wake_if_needed(self):
        if self.wake_enabled and not self.busy:
            self.start_wake()
        else:
            self.set_status("待命中")

    def closeEvent(self, event):
        self.wake_enabled = False

        workers = [
            self.init_worker,
            self.wake_worker,
            self.voice_worker,
            self.assistant_worker,
        ]

        for worker in workers:
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait()

        if self.stt is not None:
            self.stt.close()
            self.stt = None

        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("小派 AI 助理")

    window = AssistantWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
