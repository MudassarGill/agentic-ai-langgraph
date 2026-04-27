from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage,
from langgraph.graph.message import add_messages

load_dotenv()

llm=HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta",task="text-generation",max_new_tokens=500,temperature=0.1)

chat_model=ChatHuggingFace(llm=llm) 



#Here we define the schema

class Chatbot(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

prompt=PromptTemplate(
    input_variables=["messages"],
    template="""
You are a helpful assistant.

{messages}

"""
)


workflow=StateGraph(Chatbot)
workflow.add_node("chatbot",)




    