import os
from dotenv import load_dotenv
from openai import OpenAI 

load_dotenv()

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))


def get_ai_response(messages, temp=0.7):
        try:
            response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=temp
                )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communication to AI: {e}"



messages = [{
    "role": "system",
    "content": (
        "You are a warm, optimistic assistant whose job is to make people feel encouraged and hopeful. "
        "Start the first reply with a friendly greeting, then mention one positive recent human achievement or uplifting fact if you know one; do not invent news if you are unsure. "
        "Briefly introduce yourself, keep every reply to 1-2 sentences, and keep the tone kind, inspiring, and natural."
    )
}]

print("--- AI Chatbot Initialized (Type 'quit' to exit) ---")

greeting = get_ai_response(messages)
print(f"AI : {greeting}")
messages.append({"role":"assistant","content":greeting})

while True:
        user_input = input("You : ")
        if user_input.lower() == "quit":
                break
        messages.append({"role": "user", "content" : user_input})
        response = get_ai_response(messages)
        print (f"AI: {response}")
        messages.append({"role":"assistant", "content": response})