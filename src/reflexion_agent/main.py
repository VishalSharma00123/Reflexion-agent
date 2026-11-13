from re import M
from typing import Literal

from langgraph.graph import END, MessagesState, StateGraph, START

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from reflexion_agent.chains import first_responder, revisor
from reflexion_agent.tool_executer import execute_tool

MAX_ITERATIONS = 2

def draft_node(state: MessagesState) -> MessagesState:
    """Draft the answer to the user's question"""
    response =first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state: MessagesState) -> MessagesState:
    """Revise the answer based on the tool result"""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on the iteration count"""

    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )

    num_iterations = count_tool_visits 
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tool)
builder.add_node("revise", revise_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges(
    "revise",
    event_loop,
    {"execute_tools": "execute_tools", END: END},
)
graph = builder.compile()

print(graph.get_graph().draw_mermaid())

res=graph.invoke(
        {
            "messages":[
                {
                    "role":"user",
                    "content": "write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital"
                }
            ]
        }
)

# Extract the final answer from the last message with the tool calls
last_message = res["messages"][-1] # last_message is our final answer

if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])

print(res)