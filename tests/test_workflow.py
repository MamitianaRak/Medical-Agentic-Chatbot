from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph import build_medical_graph
from app.agents.state import initial_state
from tests.fakes import FakeKnowledgeBase, FakeResearchTools, kb_passage

ANSWER_PATH = ["generate", "check_grounding", "check_usefulness", "finalize"]


def run(graph, question: str, thread: str = "t1", image: str = "") -> dict:
    return graph.invoke(
        initial_state(question, image),
        config={"configurable": {"thread_id": thread}, "recursion_limit": 40},
    )


def nodes(state: dict) -> list[str]:
    return [step["node"] for step in state["trace"]]


# --- Main path and safety paths ---------------------------------------------------


def test_routine_question_answers_from_knowledge_base(graph, kb, tools):
    state = run(graph, "What triggers asthma attacks?")

    assert nodes(state) == ["triage", "retrieve", "grade_documents", *ANSWER_PATH]
    assert state["source_used"] == "private_kb"
    assert state["grounding"] == "supported"
    assert state["usefulness"] == "useful"
    assert "[PDF p.394]" in state["answer"]
    assert [p["label"] for p in state["passages"]] == ["PDF p.394", "PDF p.398"]
    assert kb.queries == ["asthma triggers"]  # the standalone query from triage
    assert tools.calls == []  # the KB was enough: no MCP call


def test_emergency_skips_retrieval_and_the_llm_answer(graph, llm, kb):
    llm.triage = {**llm.triage, "urgency": "emergency", "language": "fr"}
    state = run(graph, "J'ai une douleur dans la poitrine qui irradie dans le bras")

    assert nodes(state) == ["triage", "emergency", "finalize"]
    assert state["source_used"] == "emergency"
    assert "Appelez immédiatement" in state["answer"]
    assert kb.queries == []
    assert llm.calls("Write the answer") == 0


def test_out_of_scope_question_is_refused(graph, llm, kb):
    llm.triage = {**llm.triage, "in_scope": False}
    state = run(graph, "Write me a poem about football")

    assert nodes(state) == ["triage", "out_of_scope", "finalize"]
    assert state["source_used"] == "out_of_scope"
    assert kb.queries == []


# --- Corrective loop: MCP research tools and query rewrites ----------------------


def test_weak_knowledge_base_triggers_mcp_research(graph, llm, tools):
    llm.relevant_rounds = [[], [1]]  # nothing in the KB, the PubMed result is relevant
    state = run(graph, "What is the newest asthma biologic?")

    assert nodes(state) == [
        "triage",
        "retrieve",
        "grade_documents",
        "plan_research",
        "call_tools",
        "grade_documents",
        *ANSWER_PATH,
    ]
    assert tools.calls == [("pubmed", "asthma biologics")]
    assert state["research_calls"] == [
        {"tool": "pubmed", "query": "asthma biologics", "results": 1}
    ]
    assert state["source_used"] == "research"
    assert state["passages"][0]["label"] == "PMID:34655640"


def test_current_info_question_combines_kb_and_research(graph, llm, tools):
    llm.triage = {**llm.triage, "needs_current_info": True}
    state = run(graph, "What are the current asthma guidelines?")

    assert "call_tools" in nodes(state)
    assert state["source_used"] == "mixed"
    assert {p["source"] for p in state["passages"]} == {"kb", "pubmed"}


def test_planner_only_calls_available_tools_and_respects_the_budget(llm, kb):
    tools = FakeResearchTools(tools=["pubmed", "medlineplus"])
    llm.relevant_rounds = [[], [1]]
    llm.plan = [
        {"tool": "drug_label", "query": "ibuprofen"},  # not available
        {"tool": "pubmed", "query": "a"},
        {"tool": "medlineplus", "query": "b"},
        {"tool": "pubmed", "query": "c"},
    ]
    graph = build_medical_graph(llm, kb, tools, MemorySaver(), max_tool_calls=2)

    run(graph, "Question")

    assert tools.calls == [("pubmed", "a"), ("medlineplus", "b")]


def test_unusable_plan_falls_back_to_medlineplus(graph, llm, tools):
    llm.relevant_rounds = [[], [1]]
    llm.malformed = {"plan"}
    state = run(graph, "What is the newest asthma biologic?")

    assert tools.calls == [("medlineplus", "asthma triggers")]
    assert state["trace"][3]["detail"].endswith("(fallback)")


def test_failing_tool_only_loses_its_own_results(graph, llm, tools):
    llm.relevant_rounds = [[], [1]]
    llm.plan = [{"tool": "drug_label", "query": "x"}, {"tool": "pubmed", "query": "y"}]
    tools.fail = {"drug_label"}

    state = run(graph, "Question")

    assert [c["results"] for c in state["research_calls"]] == [0, 1]
    assert state["source_used"] == "research"


def test_nothing_found_rewrites_the_query_then_researches_again(llm):
    kb = FakeKnowledgeBase(passages=[])
    tools = FakeResearchTools(passages=[])
    graph = build_medical_graph(llm, kb, tools, MemorySaver())

    state = run(graph, "What is the prognosis of an extremely rare syndrome?")

    assert nodes(state) == [
        "triage",
        "retrieve",
        "grade_documents",
        "plan_research",
        "call_tools",
        "grade_documents",
        "rewrite_query",
        "retrieve",
        "grade_documents",
        "plan_research",
        "call_tools",
        "grade_documents",
        "insufficient",
        "finalize",
    ]
    assert kb.queries == ["asthma triggers", "asthma exacerbation triggers"]
    assert state["source_used"] == "insufficient"
    assert llm.calls("Write the answer") == 0


def test_rewrite_can_recover_from_the_knowledge_base(llm):
    kb = FakeKnowledgeBase()
    tools = FakeResearchTools(passages=[])
    llm.relevant_rounds = [[], [1]]  # KB, then the KB again (empty tool results are not graded)
    graph = build_medical_graph(llm, kb, tools, MemorySaver())

    state = run(graph, "Question")

    assert nodes(state)[6:10] == ["rewrite_query", "retrieve", "grade_documents", "generate"]
    assert state["source_used"] == "private_kb"


def test_no_research_tools_and_no_evidence_gives_insufficient_answer(llm):
    graph = build_medical_graph(llm, FakeKnowledgeBase(passages=[]), None, MemorySaver())
    state = run(graph, "Question")

    assert "plan_research" not in nodes(state)
    assert nodes(state)[-2:] == ["insufficient", "finalize"]


# --- Reflection: grounding and usefulness -----------------------------------------


def test_unsupported_answer_is_revised_once(graph, llm):
    llm.grounding = ["unsupported", "supported"]
    state = run(graph, "What triggers asthma attacks?")

    assert nodes(state)[-5:] == [
        "check_grounding",
        "revise",
        "check_grounding",
        "check_usefulness",
        "finalize",
    ]
    assert state["answer"] == "Revised answer [PDF p.394]."
    revision_prompt = next(p for p in llm.prompts if "grounding check found claims" in p)
    assert "Asthma is caused by cold weather." in revision_prompt


def test_answer_still_unsupported_after_revision_is_never_shown(graph, llm):
    llm.grounding = ["unsupported", "unsupported"]
    state = run(graph, "What triggers asthma attacks?")

    assert nodes(state)[-2:] == ["insufficient", "finalize"]
    assert state["source_used"] == "insufficient"
    assert "[PDF" not in state["answer"]


def test_not_useful_answer_triggers_targeted_research(graph, llm, tools):
    llm.usefulness = [False, True]
    state = run(graph, "What triggers asthma attacks and how is it treated?")

    assert nodes(state)[3:] == [
        "generate",
        "check_grounding",
        "check_usefulness",
        "rewrite_query",
        "plan_research",
        "call_tools",
        "grade_documents",
        *ANSWER_PATH,
    ]
    plan_prompt = next(p for p in llm.prompts if "research planner" in p)
    assert "treatment options" in plan_prompt  # the missing information guides the plan
    assert state["usefulness"] == "useful"
    assert state["source_used"] == "mixed"


def test_still_not_useful_after_the_rewrite_keeps_the_answer(graph, llm):
    llm.usefulness = [False, False]
    state = run(graph, "What triggers asthma attacks?")

    assert nodes(state)[-2:] == ["check_usefulness", "finalize"]
    assert state["usefulness"] == "not_useful"
    assert state["answer"]


def test_reflection_steps_can_be_disabled(llm, kb):
    graph = build_medical_graph(llm, kb, None, MemorySaver(), grounding_check=False)
    state = run(graph, "What triggers asthma attacks?")

    assert nodes(state) == ["triage", "retrieve", "grade_documents", "generate", "finalize"]
    assert state["grounding"] == "skipped"
    assert state["usefulness"] == "skipped"


# --- Robustness and memory ---------------------------------------------------------


def test_triage_failure_still_catches_red_flags(graph, llm, kb):
    llm.malformed = {"triage"}
    state = run(graph, "My father has chest pain and is sweating")

    assert state["urgency"] == "emergency"
    assert state["trace"][0]["detail"].endswith("(fallback)")
    assert kb.queries == []


def test_triage_failure_catches_a_closing_throat_in_english_and_french(graph, llm):
    llm.malformed = {"triage"}
    for question in (
        "A bee stung me and my throat is closing",
        "Après une piqûre de guêpe j'ai la gorge qui se ferme",
    ):
        assert run(graph, question)["urgency"] == "emergency", question


def test_triage_failure_without_red_flags_answers_normally(graph, llm):
    llm.malformed = {"triage"}
    state = run(graph, "What triggers asthma attacks?")

    assert state["urgency"] == "routine"
    assert state["source_used"] == "private_kb"


def test_malformed_output_is_asked_again_once(graph, llm):
    llm.malformed_once = {"generate"}
    state = run(graph, "What triggers asthma attacks?")

    assert llm.calls("Write the answer") == 2
    assert "[PDF p.394]" in state["answer"]


def test_grading_failure_keeps_all_candidates(graph, llm):
    llm.malformed = {"grade"}
    state = run(graph, "What triggers asthma attacks?")

    assert len(state["passages"]) == 2
    assert "(fallback)" in state["trace"][2]["detail"]


def test_usefulness_failure_keeps_the_grounded_answer(graph, llm):
    llm.malformed = {"usefulness"}
    state = run(graph, "What triggers asthma attacks?")

    assert state["usefulness"] == "useful"
    assert nodes(state)[-1] == "finalize"


def test_memory_persists_across_turns_and_reaches_the_prompt(graph, llm):
    run(graph, "What triggers asthma attacks?", thread="same")
    state = run(graph, "And in children?", thread="same")

    assert [turn["question"] for turn in state["memory"]] == [
        "What triggers asthma attacks?",
        "And in children?",
    ]
    triage_prompts = [p for p in llm.prompts if "triage step" in p]
    assert "User: What triggers asthma attacks?" in triage_prompts[-1]


def test_image_observation_reaches_triage_and_answer(graph, llm):
    run(graph, "Is this rash worrying?", image="A red, raised rash on the forearm.")

    assert any("A red, raised rash" in p for p in llm.prompts if "triage step" in p)
    assert any("A red, raised rash" in p for p in llm.prompts if "Write the answer" in p)


def test_low_relevance_documents_are_dropped(llm):
    kb = FakeKnowledgeBase(passages=[kb_passage(10, "Unrelated text about anemia.")])
    llm.relevant = []
    graph = build_medical_graph(llm, kb, None, MemorySaver())

    state = run(graph, "What triggers asthma attacks?")

    assert state["passages"] == []
    assert state["source_used"] == "insufficient"


def test_urgent_question_asks_for_prompt_care(graph, llm):
    llm.triage = {**llm.triage, "urgency": "urgent"}
    run(graph, "I have had a fever of 39.5 for three days")

    answer_prompt = next(p for p in llm.prompts if "Write the answer" in p)
    assert "within 24 hours" in answer_prompt


def test_unknown_language_falls_back_to_english(graph, llm):
    llm.triage = {**llm.triage, "language": "xx"}
    state = run(graph, "What triggers asthma attacks?")

    assert state["language"] == "en"
