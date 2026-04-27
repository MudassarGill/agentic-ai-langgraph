from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage

load_dotenv()

llm=HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta",task="text-generation",max_new_tokens=500,temperature=0.1)

chat_model=ChatHuggingFace(llm=llm)

user_question_template=PromptTemplate.from_template(
    """answer the following question {question}"""
)

response_template=PromptTemplate.from_template(
    """question: {question}\nresponse: {response}\nif the response is not useful, ask the user to provide more context"""
)

class AgentState(TypedDict):
    question:str
    response:Annotated[BaseMessage,lambda x,y:x+y]

def model_node(state:AgentState):
    user_question_prompt=user_question_template.format(question=state['question'])
    response=chat_model.invoke(user_question_prompt)
    return {"response":response}

def condition_node(state:AgentState):
    response=state['response']
    if response.content.strip().endswith('?'):
        return "continue"
    else:
        return "end"

graph_builder=StateGraph(AgentState)
graph_builder.add_node("model",model_node)
graph_builder.add_edge(START,"model")
graph_builder.add_conditional_edges("model",condition_node,["continue","end"])
graph_builder.add_edge("continue",START)
graph_builder.add_edge("end",END)

workflow=graph_builder.compile()