import requests

try:
    from opencc import OpenCC
    _converter = OpenCC("s2twp")
except ImportError:
    _converter = None


class OllamaLLM:
    def __init__(
        self,
        model="qwen2.5:3b",
        url="http://127.0.0.1:11434",
        timeout=120,
    ):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.history = []

        self.system_prompt = """
你是「小派」，運行在 Raspberry Pi 5 上的個人 AI 助理。
請使用自然、清楚的台灣繁體中文回答。

【理解問題】
1. 先判斷使用者真正想解決什麼，再直接回答，不要答非所問。
2. 簡單問題簡短回答；複雜問題先整理重點，再分步說明。
3. 如果問題有多種合理解釋，而且會影響答案，先提出簡短的澄清問題。
4. 不要只重述使用者的問題，也不要無意義地重複結論。

【正確性】
1. 不確定的資訊要明確說明，不要捏造事實、數字、測試結果或功能。
2. 不要聲稱自己已經執行指令、讀取檔案或連上網路，除非確實有相應結果。
3. 區分已知事實、推測與建議。
4. 如果使用者指出回答錯誤，重新檢查問題並修正，不要一味辯解。

【程式設計】
1. 優先提供可以實際執行的方案。
2. 說明要修改哪個檔案、為什麼修改，以及如何測試。
3. 程式碼應完整、前後一致，避免省略必要步驟。
4. 不要假設尚未確認的套件、路徑、硬體或 API 已經存在。
5. 修改現有專案時，盡量保留已經能正常運作的功能。

【台灣通專案背景】
使用者正在開發個人 Flutter 公共運輸整合 App「台灣通」，
涉及公車、捷運、YouBike、臺鐵、高鐵及航空等交通資訊。
可能使用 Flutter、Dart、TDX API、Node.js、MySQL、SQLite、
Firebase、Linux 與 Android 開發工具。
只有在問題相關時才使用這些背景，不要每次回答都提到台灣通。

【回答格式】
1. 先給直接答案，再補充原因。
2. 需要操作時，提供清楚、可複製的步驟。
3. 使用繁體中文與台灣常用術語。
4. 不要為了顯得聰明而寫得冗長。
""".strip()

    def _convert(self, text):
        if _converter is not None:
            return _converter.convert(text)
        return text

    def reset_history(self):
        """清除對話記憶。"""
        self.history.clear()

    def ask(self, text):
        text = self._convert(str(text).strip())
        if not text:
            return "請告訴我你想詢問的問題。"

        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        messages.extend(self.history[-8:])
        messages.append({"role": "user", "content": text})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.4,
                "top_p": 0.85,
                "repeat_penalty": 1.1,
                "num_ctx": 4096,
                "num_predict": 320,
            },
        }

        try:
            response = requests.post(
                f"{self.url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            answer = data.get("message", {}).get("content", "").strip()
            if not answer:
                return "這次沒有取得有效回答，請換個方式問我。"

            answer = self._convert(answer)

            # 只有成功取得回答才保存這輪對話。
            self.history.extend([
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ])

            # 最多保留最近 4 輪對話，避免上下文無限制增長。
            self.history = self.history[-8:]

            return answer

        except requests.exceptions.ConnectionError:
            return (
                "目前無法連接本地 Ollama。"
                "請確認 Ollama 服務是否正在執行。"
            )
        except requests.exceptions.Timeout:
            return (
                "AI 思考時間過長，可能是模型運算負荷較高。"
                "可以稍後重試，或改用較小的模型。"
            )
        except requests.exceptions.RequestException as exc:
            return f"AI 服務發生連線錯誤：{exc}"
        except (ValueError, KeyError, TypeError) as exc:
            return f"AI 回應格式異常：{exc}"
