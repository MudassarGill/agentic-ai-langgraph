import os
import sqlite3
from datetime import datetime
import requests
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

# Tools imports
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# LLM imports - We will configure it to use Grok (xAI) or OpenAI
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# TOOL DEFINITIONS
# ==============================================================================
# Tools are the actions our agent can take. By defining robust docstrings,
# we tell the LLM exactly when and how to use these tools.
# ==============================================================================

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    Use this tool whenever you need to perform calculations like addition, 
    subtraction, multiplication, division, or complex math equations.
    
    Args:
        expression (str): A mathematical expression (e.g., '2 + 2 * 5').
        
    Returns:
        str: The evaluated result as a string.
    """
    try:
        # Note: In production, eval can be dangerous. For a safer alternative,
        # we can use numexpr or ast.literal_eval. For learning, we will use eval.
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    Retrieves the current date and time.
    Use this tool whenever the user asks for the current time, today's date, 
    or any temporal context.
    
    Args:
        timezone (str): Optional timezone. Defaults to UTC.
        
    Returns:
        str: The current date and time formatted as a string.
    """
    return f"The current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def get_weather(city: str) -> str:
    """
    Retrieves current weather information for a given city.
    Use this tool whenever the user asks about the weather or temperature in a specific location.
    
    Args:
        city (str): The name of the city (e.g., 'London', 'New York').
        
    Returns:
        str: A text description of the weather conditions and temperature.
    """
    # Using the free Open-Meteo geocoding API to get coordinates
    geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
    try:
        geo_resp = requests.get(geocode_url).json()
        if "results" not in geo_resp or len(geo_resp["results"]) == 0:
            return f"Could not find coordinates for city: {city}"
            
        lat = geo_resp["results"][0]["latitude"]
        lon = geo_resp["results"][0]["longitude"]
        
        # Using Open-Meteo for weather data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_resp = requests.get(weather_url).json()
        
        current_weather = weather_resp.get("current_weather", {})
        temp = current_weather.get("temperature", "Unknown")
        return f"The current temperature in {city} is {temp}°C."
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

# Built-in Langchain Tools
# ------------------------
# 1. DuckDuckGoSearchRun: A search engine tool for querying the web.
# 2. WikipediaQueryRun: A tool for querying detailed articles from Wikipedia.
search_tool = DuckDuckGoSearchRun()
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Combine all tools into a list that the Agent will know about.
tools = [calculator, get_current_time, get_weather, search_tool, wikipedia_tool]


# ==============================================================================
# LLM CONFIGURATION
# ==============================================================================
# We initialize the LLM. You mentioned using Grok (xAI) or HuggingFace.
# Grok offers an OpenAI-compatible API, which works flawlessly with LangChain's
# tool-calling ecosystem using ChatOpenAI.
# 
# To use Grok, simply add to your .env:
# XAI_API_KEY="your_grok_key"
# ==============================================================================
api_key = os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy"

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1" if "XAI_API_KEY" in os.environ else None,
    model="grok-beta" if "XAI_API_KEY" in os.environ else "gpt-4o-mini",
    temperature=0.7
)

# ==============================================================================
# MEMORY / CHECKPOINTER IMPLEMENTATION
# ==============================================================================
# In LangGraph, "Memory" is implemented using "Checkpointers".
# A checkpointer saves the entire state of the graph (all messages, variables) 
# at every step. This allows the graph to pause, resume, and remember past turns.
# 
# We use `SqliteSaver`, which stores these states in an SQLite database file.
# When we run the agent, we pass a `config` dictionary:
# `{"configurable": {"thread_id": "some_unique_id"}}`
# 
# The checkpointer looks up the `thread_id` in the database, retrieves the past
# messages, injects them into the state, runs the new LLM turn, and saves the 
# updated state back to the database. This gives the agent Long-Term Persistence!
# ==============================================================================

memory = SqliteSaver.from_conn_string("checkpoints.sqlite")

# ==============================================================================
# LANGGRAPH AGENT CREATION
# ==============================================================================
# `create_react_agent` is a prebuilt LangGraph factory that wires together:
# 1. An LLM node (which can decide to call tools).
# 2. A Tool execution node.
# 3. Edges that loop back and forth between the LLM and the Tools until the LLM
#    decides it has the final answer.
# 4. The checkpointer (memory) to persist the conversation.
# ==============================================================================

app = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory
)