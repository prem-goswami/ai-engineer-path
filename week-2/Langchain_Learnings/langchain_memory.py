import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

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


history_tracker = InMemoryChatMessageHistory()

# The data flows from Prompt -> Model -> String Output Parser
buffer_chain = prompt | model | StrOutputParser()


def chat_with_buffer_memory(user_input: str):
    # Retrieve all accumulated history items from our database plane
    current_history = history_tracker.messages

    # Run our LCEL chain invocation
    response = buffer_chain.invoke({"history": current_history, "input": user_input})

    # Manually append the new conversational turn to preserve state
    history_tracker.add_user_message(user_input)
    history_tracker.add_ai_message(response)

    return response


# Test Execution
print("💬 Turn 1: Hello!")
print(chat_with_buffer_memory("Hi, my name is Prem"))

print("\n💬 Turn 2: Fact Verification")
print(chat_with_buffer_memory("What is my name?"))
