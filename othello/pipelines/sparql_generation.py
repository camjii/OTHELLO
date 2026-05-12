"""LangGraph pipeline that asks an LLM to author a SPARQL query, runs it
against Wikidata, and retries on errors. Supports parallel batch execution.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm

log = logging.getLogger(__name__)

load_dotenv()

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
DEFAULT_USER_AGENT = "OTHELLO/0.2 (https://github.com/; cekruger99@gmail.com)"

SYSTEM_PROMPT = """You are a SPARQL expert generating queries for Wikidata.

Conventions:
- Use `wdt:` for direct property relationships and `wd:` for entity values.
- Common properties: P31 (instance of), P17 (country), P30 (continent),
  P2044 (elevation), P421 (located in time zone), P569 (date of birth).
- Always include `SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }`
  when selecting entities so callers see readable labels.
- Wrap text/limit literals correctly and add LIMIT clauses.
- Return ONLY the SPARQL query — no commentary, no markdown fences.

Example:
SELECT ?item ?itemLabel WHERE {
  ?item wdt:P31 wd:Q5.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} LIMIT 10
"""


class SparqlGraphState(TypedDict, total=False):
    question: str
    sparql_query: str
    results: dict
    error: str
    attempt: int
    max_attempts: int


def _extract_query_text(response: Any) -> str:
    """Handle both classic ChatOpenAI and the newer ``responses/v1`` shape."""
    content = response.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, dict):
                value = part.get("text") or part.get("content")
                if isinstance(value, str):
                    chunks.append(value)
            elif isinstance(part, str):
                chunks.append(part)
        text = "\n".join(chunks)
    else:
        text = str(content)

    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    return text.strip()


def _make_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _make_sparql_client(user_agent: str = DEFAULT_USER_AGENT) -> SPARQLWrapper:
    client = SPARQLWrapper(WIKIDATA_SPARQL, agent=user_agent)
    client.setReturnFormat(JSON)
    return client


def build_sparql_graph(
    *,
    llm: Optional[ChatOpenAI] = None,
    sparql_client: Optional[SPARQLWrapper] = None,
):
    """Compile and return the SPARQL generate→execute→retry graph."""
    llm = llm or _make_llm()
    sparql_client = sparql_client or _make_sparql_client()

    def generate_sparql(state: SparqlGraphState) -> SparqlGraphState:
        question = state["question"]
        attempt = state.get("attempt", 0)
        prior_error = state.get("error", "")

        user_prompt = f"Generate a SPARQL query for Wikidata that answers: {question}"
        if attempt > 0 and prior_error:
            user_prompt += (
                f"\n\nPrevious attempt failed with error: {prior_error}\n"
                "Please correct the query."
            )

        response = llm.invoke([("system", SYSTEM_PROMPT), ("user", user_prompt)])
        state["sparql_query"] = _extract_query_text(response)
        state["attempt"] = attempt + 1
        return state

    def execute_sparql(state: SparqlGraphState) -> SparqlGraphState:
        query = state.get("sparql_query", "")
        try:
            sparql_client.setQuery(query)
            state["results"] = sparql_client.query().convert()
            state["error"] = ""
        except Exception as exc:  # SPARQLWrapper raises a variety of errors
            log.debug("SPARQL execution failed: %s", exc)
            state["error"] = str(exc)
            state["results"] = {}
        return state

    def check_results(state: SparqlGraphState) -> str:
        if state.get("error") and state.get("attempt", 0) < state.get("max_attempts", 1):
            return "refine"
        return "end"

    workflow = StateGraph(SparqlGraphState)
    workflow.add_node("generate", generate_sparql)
    workflow.add_node("execute", execute_sparql)
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "execute")
    workflow.add_conditional_edges(
        "execute",
        check_results,
        {"refine": "generate", "end": END},
    )
    return workflow.compile()


# Compile once at import time so callers paying the LangGraph cost can reuse it.
_default_app = None


def _get_default_app():
    global _default_app
    if _default_app is None:
        _default_app = build_sparql_graph()
    return _default_app


def run_sparql_pipeline_batch(
    questions: Iterable[str],
    max_attempts: int = 3,
    max_workers: int = 3,
    *,
    app=None,
    show_progress: bool = True,
) -> List[SparqlGraphState]:
    """Run the SPARQL pipeline for many questions in parallel, preserving order."""
    questions = list(questions)
    if not questions:
        return []

    app = app or _get_default_app()

    def run_one(question: str) -> SparqlGraphState:
        return app.invoke(
            {
                "question": question,
                "sparql_query": "",
                "results": {},
                "error": "",
                "attempt": 0,
                "max_attempts": max_attempts,
            }
        )

    results: List[Optional[SparqlGraphState]] = [None] * len(questions)
    iterator: Iterable

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(run_one, q): i for i, q in enumerate(questions)}
        iterator = as_completed(future_to_index)
        if show_progress:
            iterator = tqdm(iterator, total=len(questions), desc="SPARQL")
        for future in iterator:
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                log.exception("Pipeline error on question %r", questions[index])
                results[index] = {
                    "question": questions[index],
                    "sparql_query": "",
                    "results": {},
                    "error": f"Pipeline error: {exc}",
                    "attempt": 0,
                    "max_attempts": max_attempts,
                }

    return results  # type: ignore[return-value]
