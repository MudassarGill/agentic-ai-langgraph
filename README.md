# Agentic AI LangGraph Chatbot

This project is a LangGraph-based chatbot application that integrates with HuggingFace models (`mistralai/Mistral-7B-Instruct-v0.2`) and provides a Streamlit frontend for an interactive chat experience.

## Features

*   **LangGraph Backend:** Utilizes LangGraph for state management and workflow execution (`langraphbackend.py`).
*   **HuggingFace Integration:** Uses `mistralai/Mistral-7B-Instruct-v0.2` via `HuggingFaceEndpoint` for generating responses.
*   **Streamlit Frontend:** A user-friendly web interface for chatting with the assistant (`langraphfrontend.py`).
*   **State Management:** Implements conversation history tracking and memory.

## Installation

1.  Clone the repository.
2.  Create a virtual environment (e.g., `myvenv`):
    ```bash
    python -m venv myvenv
    myvenv\Scripts\activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install langchain-huggingface langchain-core langgraph streamlit python-dotenv
    ```
4.  Set up your `.env` file with necessary API keys:
    ```env
    HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
    ```

## Usage

To run the Streamlit frontend application, execute the following command:

```bash
streamlit run langraphfrontend.py
```

## Author

*   [MudssarGill](https://github.com/MudssarGill)