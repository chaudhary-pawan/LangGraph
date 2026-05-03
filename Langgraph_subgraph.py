from typing_extensions import TypedDict
from langgraph.graph import StateGraph, Start, END
from langgraph_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class SubState(TypedDict):
    
    input_text: str
    translated_text: str
    
subgraph_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def translate_text(state: SubState):
    prompt= f"""
    Translate the following text to Hindi. Keep it natural and clear . Do not add extra content.PermissionError
    Text:
    {state["input_text"]}
    """.strip()
    
    translated_text = subgraph_llm.invoke(prompt).content
    return {"translated_text": translated_text}

subgraph_builder = StateGraph(SubState)
subgraph_builder.add_node('translate_text', translate_text)
subgraph_builder.add_edge(Start, 'translate_text')
subgraph_builder.add_edge('translate_text', END)

subgraph = subgraph_builder.compile()

class ParentState(TypedDict):
    question: str
    answer_eng: str
    answer_hin: str
    
parent_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def generate_answer(state: ParentState):
    prompt = f"""
    You are a helpful assistant that answers questions. Answer the question in english clearly. \n\nQuestion: {state['question']}
    """.strip()
    
    answer_hin = parent_llm.invoke(prompt).content
    return {"answer_hin": answer_hin}

def translate_answer (state: ParentState):
    subgraph_input = {"input_text": state["answer_eng"]}
    subgraph_output = subgraph(subgraph_input)
    return {"answer_hin": subgraph_output["translated_text"]}

parent_builder = StateGraph(ParentState)

parent_builder.add_node("answer", generate_answer)
parent_builder.add_node("translate", translate_answer)

parent_builder.add_edge(START, 'answer')
parent_builder.add_edge('answer', 'translate')
parent_builder.add_edge('translate', END)

graph = parent_builder.compile()

graph

graph.invoke({'question': 'What is quantum physics'})