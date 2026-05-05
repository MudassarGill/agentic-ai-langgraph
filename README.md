# Agentic AI LangGraph Chatbot (ChatGPT Clone)

This project is a powerful, full-stack LangGraph-based AI Agent that replicates the core functionalities of ChatGPT. It features long-term persistence (memory), streaming responses, and the ability to autonomously use external tools to answer complex user queries. 

The application is built with a backend powered by LangChain and LangGraph, and a sleek frontend interface powered by Streamlit.

## 🚀 Key Features

*   **LangGraph Agent Architecture:** Uses `create_react_agent` to enable dynamic reasoning and tool use (`langraphbackend.py`).
*   **Long-Term Memory (Persistence):** Implements LangGraph's `SqliteSaver` to persist entire conversation threads locally to a `checkpoints.sqlite` database. 
*   **Token-by-Token Streaming:** Replicates the ChatGPT "typewriter" effect using LangGraph's advanced `stream_mode="messages"`.
*   **Autonomous Tool Use:** Equipped with 5 tools out of the box:
    *   **Calculator:** For mathematical evaluations.
    *   **Weather:** Fetches live weather data via OpenMeteo (No API Key required).
    *   **DuckDuckGo Search:** For real-time web search.
    *   **Wikipedia:** For fetching detailed encyclopedic facts.
    *   **Current Time:** For timezone-aware temporal context.
*   **Streamlit Frontend:** A modern interface (`langraphfrontend.py`) featuring a sidebar for managing past conversation threads (just like ChatGPT).
*   **Flexible LLM Support:** Easily connect to Grok (xAI) or OpenAI models via `ChatOpenAI`. (HuggingFace compatibility is also supported for models with tool-calling capabilities).

## 🛠 Installation

1.  Clone the repository.
2.  Create a virtual environment (e.g., `myvenv`):
    ```bash
    python -m venv myvenv
    myvenv\Scripts\activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install streamlit langgraph langchain-huggingface langchain-community langchain-openai duckduckgo-search wikipedia numexpr langgraph-checkpoint-sqlite requests
    ```
4.  Set up your `.env` file with necessary API keys. For Grok (xAI), use:
    ```env
    XAI_API_KEY=your_grok_api_key
    ```
    *(Alternatively, you can use `OPENAI_API_KEY=your_openai_key`)*

## 💻 Usage

To run the Streamlit frontend application, execute the following command:

```bash
streamlit run langraphfrontend.py
```

The app will open in your default browser. You can click **"➕ New Chat"** to start a new thread, or select an older thread from the sidebar to resume a past conversation.

## 🧠 Learning Resources
The `langraphbackend.py` file contains extensive docstrings explaining how Memory (Checkpointers) and Tools are implemented in LangGraph. Be sure to read the code comments if you are using this repository for learning purposes!

## Author

*   **GitHub:** [MudssarGill](https://github.com/MudssarGill)
*   **LinkedIn:** [m-mudassar-885](https://www.linkedin.com/in/m-mudassar-885)

Hit the like if this is useful for you!