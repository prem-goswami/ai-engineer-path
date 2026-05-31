import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, List

load_dotenv()
app = FastAPI()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def get_ai_response(messages, temp=0.7):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=temp
        )
        return response.choices[0].message.content
    except Exception as e:
        return {"Error": f"Could not connect to AI {e}"}


class ChatRequest(BaseModel):
    sessionId: str
    message: str = Field(min_length=1)


chat_history: Dict[str, List[dict]] = {}


@app.get("/")
def homePage():
    return {"status": "API is Online", "history_keys": list(chat_history.keys())}


@app.get("/history/{sessionId}")
def get_history(sessionId: str):
    return chat_history.get(sessionId, [])


@app.post("/chat")
async def sendMessage(request: ChatRequest):
    sessionId = request.sessionId
    userMessage = request.message

    if sessionId not in chat_history:
        chat_history[sessionId] = [
            {
                "role": "system",
                "content": (
                    "You are a warm, optimistic assistant whose job is to make people feel encouraged and hopeful. "
                    "Start the first reply with a friendly greeting, then mention one positive recent human achievement or uplifting fact if you know one; do not invent news if you are unsure. "
                    "Briefly introduce yourself, keep every reply to less than 2 sentences, and keep the tone kind, inspiring, and natural."
                ),
            }
        ]

    chat_history[sessionId].append({"role": "user", "content": userMessage})

    ai_response = await get_ai_response(chat_history[sessionId])

    chat_history[sessionId].append({"role": "assistant", "content": ai_response})

    return {
        "session_id": sessionId,
        "response": ai_response,
        "history_length": len(chat_history[sessionId]),
    }
