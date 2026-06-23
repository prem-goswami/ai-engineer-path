import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Instantiate an explicit in-memory database tracker
history_tracker = InMemoryChatMessageHistory()

model = ChatOpenAI(
    model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
)

# In modern LCEL, we use MessagesPlaceholder to inject raw history blocks dynamically
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful and precise corporate assistant."),
        MessagesPlaceholder(
            variable_name="history"
        ),  # This is where our memory will stream in!
        ("human", "{input}"),
    ]
)

# prompt for summarising conversations if messages length is more than 4
summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Progressively summarize the conversation provided, adding to the current summary.",
        ),
        (
            "human",
            "Current Summary: {current_summary}\n\nNew lines of conversation:\n{new_lines}\n\nNew Summary:",
        ),
    ]
)

# Secondary independent chain tasked exclusively with background compression
summary_compression_chain = summary_prompt | model | StrOutputParser()

buffer_chain = prompt | model | StrOutputParser()


# Global state trackers
current_summary_text = "The user initiated the chat connection."


def chat_with_summary_memory(user_input: str, max_tokens=100):
    global current_summary_text

    # If history grows too large, condense it
    if len(history_tracker.messages) > 4:
        print("\n🗜️ Token Threshold Breach! Triggering LLM Background Compression...")

        # Format the newest conversation lines into a string block
        new_lines_str = "\n".join(
            [f"{msg.type}: {msg.content}" for msg in history_tracker.messages]
        )

        # Compress history into a new summary paragraph
        current_summary_text = summary_compression_chain.invoke(
            {"current_summary": current_summary_text, "new_lines": new_lines_str}
        )

        # Wipe the raw historical message records out to clean up space
        history_tracker.clear()

    # Inject the summarized history context as a system message placeholder frame
    runtime_history = [
        ("system", f"Historical Summary of past turns: {current_summary_text}")
    ] + history_tracker.messages

    response = buffer_chain.invoke({"history": runtime_history, "input": user_input})

    history_tracker.add_user_message(user_input)
    history_tracker.add_ai_message(response)

    return response
