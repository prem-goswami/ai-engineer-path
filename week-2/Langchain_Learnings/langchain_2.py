import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
)


class LeafBlowerReviewSchema(BaseModel):
    gift: bool
    delivery_days: int
    price_value: List[str]


# creates a Json parser that follows the pydantic model.
json_parser = JsonOutputParser(pydantic_object=LeafBlowerReviewSchema)


review_template = """\
For the following text, extract the required structured data information.

Text Content: {text}

{format_instructions}
"""
# create a detailed prompt with format instructions by assigning live variables
prompt = ChatPromptTemplate.from_template(
    template=review_template,
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)

structured_extraction_chain = (
    prompt | model | json_parser
)  # checks the model output and make sures it follows the schema


customer_review = """\
This leaf blower is pretty amazing. It has four settings:
candle blower, gentle breeze, windy city, and tornado. 
It arrived in two days, just in time for my wife's 
anniversary present. 
I think my wife liked it so much she was speechless. 
So far I've been the only one using it, and I've been 
using it every other morning to clear the leaves on our lawn. 
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

print("\n⏳ Executing Structured JSON Extraction Chain...")
extracted_data = structured_extraction_chain.invoke({"text": customer_review})
print(f"extracted_data : {extracted_data}")
