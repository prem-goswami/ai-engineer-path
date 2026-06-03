import os
from dotenv import load_dotenv

# Async version of OpenAI client Multiple requests can be sent and transforms
# into asyncronus function, useful for streaming data.
from openai import AsyncOpenAI
from fastapi import FastAPI

# A specialized FastAPI class that sends data to the client "chunk by chunk"
# rather than waiting for the whole message to finish.
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# load env file
load_dotenv()
app = FastAPI()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# code to stream chunks of data in real time.
async def ai_streamer(meassages, sessionId):
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=meassages,
        stream=True,
        # stream=True is used to get a stream of data packets instead of a single JSon object
    )
    # full response is a Accumilator to append the entire response after receiving all of the response
    fullresponse = ""
    # async for iterates through the chunk arriving over the network
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            fullresponse += content
            yield content
    chat_history[sessionId].append({"role": "assistant", "content": fullresponse})


# calling AI
async def get_ai_response(messages, temp=0.7):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=temp
        )
        return response.choices[0].message.content
    except Exception as e:
        return {"Error": f"Could not connect to AI {e}"}


# pydantic
class ChatRequest(BaseModel):
    sessionId: str
    message: str = Field(min_length=1)


# chat history is a dictinoary whith strings as keys and list of dictionaries as values
chat_history: dict[str, list[dict]] = {}


@app.get("/")
def homePage():
    return {"status": "API is Online", "history_keys": list(chat_history.keys())}


@app.get("/history/{sessionId}")
def get_history(sessionId: str):
    # get is used to nbot throw and error if sessionId is missing instead respond with []
    return chat_history.get(sessionId, [])


@app.post("/chat")
async def sendMessage(request: ChatRequest):
    # access the user input json
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


# Endpoint to get stream of data
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
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

    return StreamingResponse(
        ai_streamer(chat_history[sessionId], sessionId), media_type="text/plain"
    )
