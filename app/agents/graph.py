"""LangGraph workflow: Corrective + Reflective RAG with MCP research tools.

triage -> knowledge base -> relevance grading
       -> (weak or outdated) research planning -> MCP tool calls -> grading
       -> (still weak) query rewrite -> retry
answer -> grounding check (revise once) -> usefulness check (rewrite and research once)
"""

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes import MedicalAgents
from app.agents.state import MedicalState
from app.agents.tools import KnowledgeBase, ResearchTools


def build_medical_graph(
    llm: BaseChatModel,
    knowledge_base: KnowledgeBase,
    research_tools: ResearchTools | None,
    checkpointer: BaseCheckpointSaver | None,
    *,
    fast_llm: BaseChatModel | None = None,
    memory_turns: int = 4,
    max_tool_calls: int = 3,
    max_research_rounds: int = 2,
    max_query_rewrites: int = 1,
    grounding_check: bool = True,
    max_grounding_retries: int = 1,
    usefulness_check: bool = True,
) -> CompiledStateGraph:
    agents = MedicalAgents(
        llm,
        knowledge_base,
        research_tools,
        fast_llm=fast_llm,
        memory_turns=memory_turns,
        max_tool_calls=max_tool_calls,
        max_research_rounds=max_research_rounds,
        max_query_rewrites=max_query_rewrites,
        max_grounding_retries=max_grounding_retries,
        # Usefulness is judged on a grounded answer, so it needs the grounding check.
        usefulness_check=usefulness_check and grounding_check,
    )
    graph = StateGraph(MedicalState)

    for name in (
        "triage",
        "emergency",
        "out_of_scope",
        "retrieve",
        "grade_documents",
        "plan_research",
        "call_tools",
        "rewrite_query",
        "generate",
        "insufficient",
        "finalize",
    ):
        graph.add_node(name, getattr(agents, name))

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        agents.route_after_triage,
        {"emergency": "emergency", "out_of_scope": "out_of_scope", "retrieve": "retrieve"},
    )

    # Corrective loop: private knowledge base, then MCP research tools, then query rewrites.
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        agents.route_after_grading,
        {
            "generate": "generate",
            "plan_research": "plan_research",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )
    graph.add_edge("plan_research", "call_tools")
    graph.add_edge("call_tools", "grade_documents")
    graph.add_conditional_edges(
        "rewrite_query",
        agents.route_after_rewrite,
        {"retrieve": "retrieve", "plan_research": "plan_research"},
    )

    # Reflection: grounding (revise once, then refuse) and usefulness (search again once).
    if grounding_check:
        graph.add_node("check_grounding", agents.check_grounding)
        graph.add_node("revise", agents.revise)
        graph.add_node("check_usefulness", agents.check_usefulness)
        graph.add_edge("generate", "check_grounding")
        graph.add_conditional_edges(
            "check_grounding",
            agents.route_after_grounding,
            {
                "check_usefulness": "check_usefulness",
                "finalize": "finalize",
                "revise": "revise",
                "insufficient": "insufficient",
            },
        )
        graph.add_edge("revise", "check_grounding")
        graph.add_conditional_edges(
            "check_usefulness",
            agents.route_after_usefulness,
            {"rewrite_query": "rewrite_query", "finalize": "finalize"},
        )
    else:
        graph.add_edge("generate", "finalize")

    for terminal in ("emergency", "out_of_scope", "insufficient"):
        graph.add_edge(terminal, "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
