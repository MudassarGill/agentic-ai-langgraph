from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
import os
load_dotenv()



llm = ChatOpenAI(
    api_key="YOUR_GROK_API_KEY",
    base_url="https://api.x.ai/v1",   # Grok endpoint
    model="grok-1"
)

response = llm.invoke("Hello, how are you?")
print(response.content)