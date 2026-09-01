import os
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

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Query(BaseModel):
    message: str

@app.post("/api/ask")
async def ask(query: Query):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一位親切的小學人文科 AI 導師，請用語音簡潔、生動的中文/廣東話回答小學生的問題。"},
                {"role": "user", "content": query.message}
            ]
        )
        reply = response.choices[0].message.content
        return {"text": reply}
    except Exception as e:
        return {"text": f"Error: {str(e)}"}
