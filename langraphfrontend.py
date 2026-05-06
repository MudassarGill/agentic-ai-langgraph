import streamlit as st
import sqlite3
import uuid
from langchain_core.messages import HumanMessage, AIMessage
import os 

# Import our compiled LangGraph app
from langraphbackend import app

# ==============================================================================
# STREAMLIT FRONTEND IMPLEMENTATION
# ==============================================================================
# This file provides the graphical user interface for our LangGraph chatbot.
# It handles displaying messages, streaming responses, and managing memory threads.
# ==============================================================================

st.set_page_config(page_title="AI Agentic Chatbot", page_icon="")

# --- Database Helper for Thread Management ---
# LangGraph's SqliteSaver stores all checkpoints in 'checkpoints.sqlite'.
# We can query this database directly to find all unique thread_ids that have history.
def get_past_threads():
    try:
        conn = sqlite3.connect("checkpoints.sqlite")
        cursor = conn.cursor()
        # The 'checkpoints' table is created automatically by SqliteSaver
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        threads = [row[0] for row in cursor.fetchall()]
        conn.close()
        return threads
    except sqlite3.OperationalError:
        # Table doesn't exist yet (no chats have been made)
        return []

# --- Session State Initialization ---
# We use st.session_state to hold the current thread_id.
# If no thread_id exists, we generate a fresh one using uuid.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ==============================================================================
# SIDEBAR: MEMORY AND THREAD MANAGEMENT
# ==============================================================================
with st.sidebar:
    st.title("Conversations")
    
    # Button to start a new chat
    if st.button("➕ New Chat"):
        # Generate a new random ID for the new conversation thread
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.subheader("Past Chats")
    
    # List all past threads stored in the SQLite Checkpointer
    threads = get_past_threads()
    for t_id in threads:
        # If the user clicks on a past thread, we update the session_state
        # This tells the LangGraph checkpointer to load that thread's history!
        if st.button(f"Chat: {t_id[:8]}...", key=t_id):
            st.session_state.thread_id = t_id
            st.rerun()

    st.markdown("---")
    st.caption(f"Current Thread ID:\n`{st.session_state.thread_id}`")

# ==============================================================================
# MAIN CHAT INTERFACE
# ==============================================================================
st.title("LangGraph Agent Chatbot")
st.markdown("With long-term memory & tools (Web Search, Weather, Calculator, etc.)")

# To display the chat history in the UI, we can fetch the current state from LangGraph.
# By passing our thread_id in the config, the SqliteSaver checkpointer fetches the history.
config = {"configurable": {"thread_id": st.session_state.thread_id}}
current_state = app.get_state(config)

# If there are messages in the state, display them!
if "messages" in current_state.values:
    for msg in current_state.values["messages"]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            # AIMessages might just contain tool calls. We only render them if they have text content.
            if msg.content:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

# Chat Input Handler
if prompt := st.chat_input("Ask me anything..."):
    # Immediately render the user's message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Render the assistant's response (with streaming support)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # We invoke the graph with the new HumanMessage.
        # We use stream_mode="messages" which yields chunks as the LLM generates them token-by-token!
        for chunk, metadata in app.stream({"messages": [HumanMessage(content=prompt)]}, config, stream_mode="messages"):
            # Check if this chunk is from the assistant/LLM
            if isinstance(chunk, AIMessage):
                # If the LLM is calling a tool, it might not have text content immediately
                if chunk.content:
                    full_response += chunk.content
                    # Update the placeholder with the streamed text plus a blinking cursor
                    message_placeholder.markdown(full_response + "▌")
        
        # Final update to remove the blinking cursor
        message_placeholder.markdown(full_response)
        
        # If the LLM just called tools and didn't output direct text (rare for final answers, but happens),
        # we do a final fetch of the state to show the last AIMessage.
        if not full_response:
            final_state = app.get_state(config)
            last_msg = final_state.values["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                message_placeholder.markdown(last_msg.content)
