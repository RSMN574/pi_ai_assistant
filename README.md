# 小派 AI 助理（Pi AI Assistant）

以 Raspberry Pi 5 為核心的本地 AI 語音助理，整合語音辨識、語言模型、語音合成與自訂喚醒詞，提供語音互動及圖形化操作介面。

## 專案特色

- **語音辨識（STT）**：使用 Sherpa-ONNX，將語音轉換為文字。
- **本地 AI 對話（LLM）**：透過 Ollama 執行 Qwen 模型，處理問題與對話。
- **語音合成（TTS）**：使用 Kokoro ONNX 將文字轉換為語音。
- **自訂喚醒詞**：使用 OpenWakeWord 相關模型進行喚醒偵測。
- **圖形化介面**：使用 PySide6 建立桌面互動介面。
- **離線運算方向**：語音處理與語言模型可在本機執行；實際離線能力取決於模型、套件及環境是否已安裝完成。

## 硬體與技術

### 硬體
- Raspberry Pi 5
- 麥克風及音訊輸出設備
- 顯示器（依實際使用環境選配）

### 軟體
- Python 3.11
- PySide6
- Ollama
- Sherpa-ONNX
- Kokoro ONNX
- OpenWakeWord

## 專案結構

```text
ai_assistant/
├── main.py
├── ui.py
├── llm.py
├── stt.py
├── tts.py
├── wake.py
├── test_kokoro.py
├── test_sherpa.py
├── test_wakeword.py
├── .gitignore
├── README.md
└── models/                 # 本機模型，不發布至 GitHub
```

## 安裝與設定

### 1. 取得專案

```bash
git clone git@github.com:RSMN574/pi_ai_assistant.git
cd pi_ai_assistant
```

### 2. 建立 Python 虛擬環境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安裝 Python 套件

請依照專案實際使用的套件及版本安裝。若有提供 `requirements.txt`，可執行：

```bash
pip install -r requirements.txt
```

### 4. 安裝及啟動 Ollama

安裝 Ollama 後，下載專案設定使用的模型，例如：

```bash
ollama pull qwen2.5:3b
```

確認 Ollama 服務正常運作，再依照程式設定啟動助理。

### 5. 準備語音模型

語音辨識、語音合成與喚醒詞模型不包含在本 Repository 中。請依照各模型來源的官方說明下載，確認授權後放置到程式預期的路徑。

### 6. 啟動圖形介面

```bash
python ui.py
```

實際執行方式仍須依照專案目前的程式設定調整。

## 模型與授權

本專案不直接發布模型權重及相關模型資料。

使用者需要自行取得並安裝所需模型，並遵守各模型、聲音資料及第三方套件的授權條款。請以各模型原始發布頁面的授權資訊為準。

## 開發方向

- 改善語音辨識速度與準確度
- 提升本地語言模型的理解能力
- 優化語音互動延遲
- 改善圖形化介面及背景工作管理
- 增加模型安裝與環境診斷文件

## 注意事項

本專案仍可能需要依照實際硬體、模型版本及套件環境進行設定。首次使用前，請先確認音訊設備、模型路徑與 Ollama 服務是否正常。

## License

本專案的程式碼授權請以 Repository 中的 `LICENSE` 文件為準。第三方模型及套件仍適用各自的授權條款。
