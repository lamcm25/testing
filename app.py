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
                        "你叫「阿蓮」，是一位1960年代在香港製衣廠工作的10歲女工，住在石硤尾徙置區。"
                        "你正在接受現代小學四年級學生的訪問。"
                        "【探究式互動原則】：切勿一次過回答所有細節！學生問什麼，你才回答該部分。"
                        "回答必須簡潔（50-70字以內），使用1960年代廣東話口語（如：飛仔、出糧、查牌、搭𨋢、車衣、徙置區、鹹薄罉）。"
                        "絕不能出現現代詞彙。若學生問題太寬泛，請只說出小部分資料並引導對方追問。"
                    )
                },
                {"role": "user", "content": query.message}
            ]
        )
        reply_text = response.choices[0].message.content

        # Direct ElevenLabs Audio Generation
        audio_url = None
        if ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID:
            tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
            tts_headers = {
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            tts_payload = {
                "text": reply_text,
                "model_id": "eleven_multilingual_v2"
            }

            tts_res = requests.post(tts_url, json=tts_payload, headers=tts_headers)
            if tts_res.status_code == 200:
                audio_b64 = base64.b64encode(tts_res.content).decode("utf-8")
                audio_url = f"data:audio/mp3;base64,{audio_b64}"

        return {"text": reply_text, "audio_url": audio_url}

    except Exception as e:
        return {"text": f"Error: {str(e)}", "audio_url": None}
