#Here we use streamming and also with slide bar where our old and new converstion is stored just like chatgpt for 
#we have a buttion with new chat and also we have my coversion in the left bar where we can delete the converstion or rename it
# we add also thread id with im memory sever which save our conversation and we can access it later
import streamlit as st
from langgraphbackend import app
from typing import TypedDict,Annotated
from langchain_core.messages import HumanMessage,AIMessage,add_messages
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

class State(TypedDict):
    messages:Annotated[list,add_messages]

st.title("Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt:=st.chat_input("You are a helpful assistant"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder=st.empty()
        full_response=""
        for chunk in app.stream({"messages":st.session_state.messages}):
            if "messages" in chunk:
                for msg in chunk["messages"]:
                    if msg.type=="ai" and msg.content:
                        full_response+=msg.content
            message_placeholder.markdown(full_response+"▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append(AIMessage(content=full_response))

