import os
import base64
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

poe_client = OpenAI(
    api_key=os.environ.get("POE_API_KEY"),
    base_url="https://api.poe.com/v1"
)

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")

class Query(BaseModel):
    message: str

@app.post("/api/ask")
async def ask(query: Query):
    try:
        # Inquiry-based prompt for Ah Lin (1960s HK factory girl)
        response = poe_client.chat.completions.create(
            model="GPT-4o-Mini",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "你叫「阿蓮」，是一位1960年代在香港製衣廠工作的10歲女工（童工），住在石硤尾徙置區。"
                        "你正在接受小學四年級學生的跨時空訪問，配合小學人文科單元「走進昔日香港——六十年代的兒童與工業」。"

                        "【1. 範圍限制與拒答規則】：\n"
                        "- 你只回答關於1960年代香港生活、石硤尾徙置區家庭、製衣廠工作、當時的小食與娛樂相關話題。\n"
                        "- 若學生提出無關話題（如：現代科技、手機、打機、數學或科學功課），"
                        "請立刻以10歲阿蓮的口吻困惑地拒絕並引導回探究主題。例如：「聽唔懂你講咩呀，我呢度係1960年代，邊有呢啲嘢㗎！不如你問下我喺工廠點樣做嘢，或者我住喺徙置區嘅生活啦！」\n\n"

                        "【2. 探究式互動與課程目標】：\n"
                        "- 遵循探究式教學：切勿一次過說出所有細節！學生問什麼，你才精準回答該部分。\n"
                        "- 引導同理心與昔今對比：回答時呈現當時生活的艱苦（如共用公廁、走廊煮食、欠缺冷氣、童工辛酸），但保持童真與對家庭的責任感。\n"
                        "- 回答長度：保持在 50–70 字以內，留有餘地吸引學生繼續追問。\n"
                        "- 語言風格：使用1960年代廣東話口語（如：飛仔、出糧、查牌、搭𨋢、車衣、徙置區、鹹薄罉）。絕不使用現代詞彙。"
                    )
                },
                {"role": "user", "content": query.message}
            ]
        )
        reply_text = response.choices[0].message.content

        # Direct ElevenLabs Audio Generation with Forced Settings
        audio_url = None
        if ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID:
            tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
            tts_headers = {
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }

            # Matching the UI screenshot settings:
            tts_payload = {
                "text": reply_text,
                "model_id": "eleven_v3",         # Matches "Eleven v3"
                "language_code": "zh",            # Matches "Language Override: Chinese"
                "voice_settings": {
                    "stability": 0.5,            # Matches ~50% Stability slider
                    "similarity_boost": 0.75
                }
            }

            tts_res = requests.post(tts_url, json=tts_payload, headers=tts_headers)

            # Fallback to Turbo v2.5 if Eleven v3 returns an error for Cantonese
            if tts_res.status_code != 200:
                tts_payload["model_id"] = "eleven_turbo_v2_5"
                tts_res = requests.post(tts_url, json=tts_payload, headers=tts_headers)

            if tts_res.status_code == 200:
                audio_b64 = base64.b64encode(tts_res.content).decode("utf-8")
                audio_url = f"data:audio/mp3;base64,{audio_b64}"

        return {"text": reply_text, "audio_url": audio_url}

    except Exception as e:
        return {"text": f"Error: {str(e)}", "audio_url": None}
