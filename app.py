import os
import time
import json
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

DID_API_KEY = os.environ.get("DID_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")

class Query(BaseModel):
    message: str

@app.post("/api/ask")
async def ask(query: Query):
    try:
        # 1. Generate Cantonese response from Poe API
        response = poe_client.chat.completions.create(
            model="GPT-4o-Mini",
            messages=[
                {"role": "system", "content": "你是一位親切的小學人文科 AI 導師，請用語音簡潔、生動的廣東話回答小學生的問題（回答請保持在50字以內）。"},
                {"role": "user", "content": query.message}
            ]
        )
        reply_text = response.choices[0].message.content

        # 2. Trigger D-ID Video Generation
        if DID_API_KEY:
            did_headers = {
                "Authorization": f"Basic {DID_API_KEY}",
                "Content-Type": "application/json"
            }
            if ELEVENLABS_API_KEY:
                did_headers["x-api-key-external"] = json.dumps({"elevenlabs": ELEVENLABS_API_KEY})

            did_payload = {
                "source_url": "https://cdn.jsdelivr.net/gh/lamcm25/testing@main/avatar2.png",
                "script": {
                    "type": "text",
                    "input": reply_text,
                    "provider": {
                        "type": "elevenlabs",
                        "voice_id": ELEVENLABS_VOICE_ID or "21m00Tcm4TlvDq8ikWAM"
                    }
                }
            }

            talk_res = requests.post("https://api.d-id.com/talks", json=did_payload, headers=did_headers)
            talk_data = talk_res.json()

            if talk_res.status_code not in [200, 201]:
                err_msg = talk_data.get("message", str(talk_data))
                return {"text": f"{reply_text} (D-ID Error: {err_msg})", "video_url": None}

            talk_id = talk_data.get("id")

            # 3. Poll D-ID for completion status
            video_url = None
            if talk_id:
                for _ in range(25):
                    time.sleep(1)
                    status_res = requests.get(f"https://api.d-id.com/talks/{talk_id}", headers=did_headers)
                    status_data = status_res.json()
                    
                    if status_data.get("status") == "done":
                        video_url = status_data.get("result_url")
                        break
                    elif status_data.get("status") == "error":
                        err_detail = status_data.get("error", {})
                        return {"text": f"{reply_text} (Animation Error: {err_detail})", "video_url": None}

            return {"text": reply_text, "video_url": video_url}

        return {"text": reply_text, "video_url": None}

    except Exception as e:
        return {"text": f"Error: {str(e)}", "video_url": None}
