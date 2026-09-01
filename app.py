import os
import requests
import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow connections from web browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
poe_client = openai.OpenAI(
    api_key=os.getenv("POE_API_KEY"),
    base_url="https://api.poe.com/v1"
)

CARTOON_IMAGE_URL = "https://imgur.com/a/dRryV40"  # Public link to your cartoon picture

class StudentQuery(BaseModel):
    message: str

@app.post("/api/ask")
async def ask_tutor(query: StudentQuery):
    try:
        # 1. Query your Custom Poe Bot (restricted to your PDF via GPT-5-mini)
        poe_response = poe_client.chat.completions.create(
            model="HK_Primary_Humanities",  # Must match your Poe Bot handle exactly
            messages=[{"role": "user", "content": query.message}]
        )
        cantonese_text = poe_response.choices[0].message.content

        # 2. Convert text to warm Cantonese audio via ElevenLabs
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{os.getenv('ELEVENLABS_VOICE_ID')}"
        headers = {
            "xi-api-key": os.getenv("ELEVENLABS_API_KEY"),
            "Content-Type": "application/json"
        }
        payload = {
            "text": cantonese_text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.8}
        }
        audio_res = requests.post(tts_url, json=payload, headers=headers)
        
        # 3. Request D-ID to animate your cartoon avatar face with Cantonese text
        did_url = "https://api.d-id.com/talks"
        did_headers = {
            "Authorization": f"Basic {os.getenv('DID_API_KEY')}",
            "Content-Type": "application/json"
        }
        did_payload = {
            "source_url": CARTOON_IMAGE_URL,
            "script": {
                "type": "text",
                "subtitles": "false",
                "provider": {"type": "microsoft", "voice_id": "zh-HK-HiuGaaiNeural"},
                "input": cantonese_text
            }
        }
        did_res = requests.post(did_url, json=did_payload, headers=did_headers)
        did_data = did_res.json()

        return {
            "text": cantonese_text,
            "talk_id": did_data.get("id")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))