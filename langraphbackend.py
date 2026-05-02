from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage
from langgraph.graph.message import add_messages
import sqlite3
import os
os.environ["LANGGRAPH_DEBUG"] = "true"
database="""create table thread_id(
    thread_id int primary key auto_increment,
    thread_title text
)"""

# we use sqlite3 to store the thread id and also the conversation

conn=sqlite3.connect("chatbot.db")
cursor=conn.cursor()

load_dotenv()

llm=HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",task="text-generation",max_new_tokens=500,temperature=0.1)

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

def chatbot_node(state:Chatbot) -> Chatbot:
    messages=state["messages"]
    response=chat_model.invoke(messages)
    return {"messages":[response]}

workflow=StateGraph(Chatbot)
workflow.add_node("chatbot",chatbot_node)
workflow.add_edge(START,"chatbot")
workflow.add_edge("chatbot",END)

app=workflow.compile()

initial_state={
    'messages':[HumanMessage(content='What is the capital of France?')]
}

app.invoke(initial_state)




    