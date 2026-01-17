from langgraph.graph import StateGraph
from app.state import ChatState
from app.nodes import rewrite_node, retrieve_node, generate_node, memory_node


def build_graph(retriever):
    graph = StateGraph(ChatState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    graph.add_node("generate", generate_node)
    graph.add_node("memory", memory_node)

    graph.set_entry_point("rewrite")

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "memory")

    return graph.compile()
