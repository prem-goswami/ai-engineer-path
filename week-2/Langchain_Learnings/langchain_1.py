import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
)


template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

# 3. Construct the prompt template object by replacing anchor variables with live variables
prompt_template = ChatPromptTemplate.from_template(template_string)

# in translation chain each module transfers data to the next module '|' pipe operator acts as a mediator
translation_chain = prompt_template | model | StrOutputParser()


customer_style = "American English in a calm and respectful tone"
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
translated_output = translation_chain.invoke(
    {"style": customer_style, "text": customer_email}
)

print(translated_output)
