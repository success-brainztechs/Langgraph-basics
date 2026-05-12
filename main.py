from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoning, tool_node

load_dotenv()

# Declaring some constants for code readability
AGENT_REASON="agent_reason"
ACT="act"
LAST=-1

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

# Lets define the flow
flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END: END,
    ACT: ACT
})

flow.add_edge(ACT, AGENT_REASON)
# To see the flow built above, you can see the copy (KZ#3) fig: langgraph_code 1.0

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What is the current temperature in Tokyo? List it and then triple it")]})
    print(res["messages"][LAST].content[0]['text'])
    
