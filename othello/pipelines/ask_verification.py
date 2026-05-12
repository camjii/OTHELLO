"""LangGraph pipeline that verifies an LLM answer against Wikidata via ASK queries.

The pipeline takes ``{"question", "answer"}``, decomposes the answer into atomic
claims, links entities/relations to Wikidata QIDs/PIDs, and dispatches one ASK
query per claim. A claim is ``SUPPORTED`` if any direction returns ``true``,
``REFUTED`` if the query is well-formed but returns ``false`` for every
direction, and ``NOT_FOUND`` when there is not enough material to verify it.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from othello.utils.text import (
    normalize_label,
    safe_json_list,
    safe_text,
)
from othello.utils.wikidata import WikidataClient, build_ask_query

log = logging.getLogger(__name__)

load_dotenv()


class GraphState(TypedDict, total=False):
    question: str
    answer: str
    rephrased_claim: str
    claims: List[str]
    parsed_entities: Dict[int, List[str]]
    parsed_relations: Dict[int, List[str]]
    entity_qids: Dict[int, List[str]]
    relation_pids: Dict[int, List[str]]
    queries: List[Optional[str]]
    results: List[Optional[bool]]
    claim_verdicts: List[str]
    verdict: str
    errors: List[str]


REPHRASE_PROMPT = """You are rephrasing a short answer into a complete, verifiable claim.

Rules:
- Convert the answer into a single complete sentence.
- Include the subject from the question so the claim stands alone.
- Be precise and factual; do not add information beyond the answer.
- Return ONLY the rephrased claim — no preface, no commentary.

Example:
Question: "What time zone is Salt Lake City in?"
Answer: "Mountain Time"
Rephrased: "Salt Lake City is in the Mountain Time zone."
"""

SPLIT_PROMPT = """Split the response into atomic, verifiable claims.

Rules:
- Each claim should state ONE fact.
- Each claim should be independently verifiable.
- Keep claims simple and direct; preserve all factual information.
- Output ONLY a valid JSON array of strings.

Example:
Input: "Albert Einstein had two sons, Hans Albert and Eduard, with his first wife Mileva Marić."
Output: ["Albert Einstein had a son named Hans Albert", "Albert Einstein had a son named Eduard", "Albert Einstein's first wife was Mileva Marić"]
"""

ENTITY_PROMPT = """Extract the named entities (people, places, organizations, concepts) from this claim.

Rules:
- Extract proper nouns and key concepts; use full names when available.
- List each distinct entity separately.
- Output ONLY a JSON array of entity names as strings.

Examples:
Claim: "Salt Lake City is in the Mountain Time zone"
Output: ["Salt Lake City", "Mountain Time"]

Claim: "Albert Einstein had a son named Hans Albert"
Output: ["Albert Einstein", "Hans Albert Einstein"]
"""

RELATION_PROMPT = """Extract the relationship/property being described in this claim.

Think about which Wikidata property would connect the entities.

Common properties:
- "located in time zone" / "time zone"
- "child" / "parent"
- "spouse" / "married to"
- "occupation" / "profession"
- "country" / "located in"

Rules:
- Extract the relationship as a Wikidata-style property name.
- Be specific but use common Wikidata terminology.
- Output ONLY a JSON array with one property name.

Examples:
Claim: "Salt Lake City is in the Mountain Time zone"
Output: ["time zone"]

Claim: "Albert Einstein had a son named Hans Albert"
Output: ["child"]
"""


def _default_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def build_verification_graph(
    *,
    llm: Optional[ChatOpenAI] = None,
    wikidata: Optional[WikidataClient] = None,
    debug: bool = False,
    debug_fn: Callable[[str], None] = log.debug,
):
    """Compile and return the verification graph.

    Arguments are all optional — sensible defaults are used so existing
    notebooks can call ``build_verification_graph()`` with no arguments.
    """
    llm = llm or _default_llm()
    wikidata = wikidata or WikidataClient()

    def log_dbg(msg: str) -> None:
        if debug:
            debug_fn(msg)

    def router(state: GraphState) -> GraphState:
        state.setdefault("errors", [])
        return state

    def route_rephrase(state: GraphState) -> str:
        answer = safe_text(state.get("answer", ""))
        words = answer.split()
        if len(words) <= 5 or "." not in answer:
            return "rephrase"
        return "skip_rephrase"

    def rephrase_answer(state: GraphState) -> GraphState:
        response = llm.invoke(
            [
                ("system", REPHRASE_PROMPT),
                (
                    "user",
                    f"Question: {state.get('question', '')}\n"
                    f"Answer: {state.get('answer', '')}",
                ),
            ]
        )
        rephrased = response.content.strip()
        state["rephrased_claim"] = rephrased
        log_dbg(f"Rephrased: {state.get('answer')!r} -> {rephrased!r}")
        return state

    def skip_rephrase(state: GraphState) -> GraphState:
        state["rephrased_claim"] = safe_text(state.get("answer", "")).strip()
        return state

    def split_response_into_claims(state: GraphState) -> GraphState:
        response = llm.invoke(
            [("system", SPLIT_PROMPT), ("user", f"Response: {state['rephrased_claim']}")]
        )
        claims = safe_json_list(response.content, [state["rephrased_claim"]])
        state["claims"] = claims
        log_dbg(f"Split into {len(claims)} claims: {claims}")
        return state

    def parse_entities(state: GraphState) -> GraphState:
        parsed: Dict[int, List[str]] = {}
        for i, claim in enumerate(state["claims"]):
            response = llm.invoke([("system", ENTITY_PROMPT), ("user", f"Claim: {claim}")])
            entities = [normalize_label(e) for e in safe_json_list(response.content)]
            parsed[i] = [e for e in entities if e]
            log_dbg(f"Claim {i}: entities={parsed[i]}")
        state["parsed_entities"] = parsed
        return state

    def parse_relations(state: GraphState) -> GraphState:
        parsed: Dict[int, List[str]] = {}
        for i, claim in enumerate(state["claims"]):
            response = llm.invoke([("system", RELATION_PROMPT), ("user", f"Claim: {claim}")])
            relations = [normalize_label(r) for r in safe_json_list(response.content)]
            parsed[i] = [r for r in relations if r]
            log_dbg(f"Claim {i}: relations={parsed[i]}")
        state["parsed_relations"] = parsed
        return state

    def find_entities(state: GraphState) -> GraphState:
        out: Dict[int, List[str]] = {}
        for i, entities in state.get("parsed_entities", {}).items():
            qids: List[str] = []
            for name in entities:
                qid = wikidata.search_entity(name)
                if qid:
                    qids.append(qid)
                else:
                    log_dbg(f"No QID for entity {name!r}")
            out[i] = qids
        state["entity_qids"] = out
        return state

    def find_relations(state: GraphState) -> GraphState:
        out: Dict[int, List[str]] = {}
        for i, relations in state.get("parsed_relations", {}).items():
            pids: List[str] = []
            for name in relations:
                pid = wikidata.search_property(name)
                if pid:
                    pids.append(pid)
                else:
                    log_dbg(f"No PID for relation {name!r}")
            out[i] = pids
        state["relation_pids"] = out
        return state

    def create_query(state: GraphState) -> GraphState:
        queries: List[Optional[str]] = []
        skipped: List[tuple] = []
        for i in range(len(state["claims"])):
            qids = state.get("entity_qids", {}).get(i, [])
            pids = state.get("relation_pids", {}).get(i, [])
            query = build_ask_query(qids, pids)
            if query is None:
                reason = (
                    f"insufficient entities ({len(qids)})"
                    if len(qids) < 2
                    else "no properties found"
                )
                skipped.append((i, reason))
            queries.append(query)
            log_dbg(f"Claim {i} query={query}")
        if skipped:
            log_dbg(f"Skipped {len(skipped)} claim(s): {skipped}")
        state["queries"] = queries
        return state

    def run_query(state: GraphState) -> GraphState:
        results: List[Optional[bool]] = []
        for idx, query in enumerate(state.get("queries", [])):
            if query is None:
                results.append(None)
                continue
            result = wikidata.ask(query)
            results.append(result)
            log_dbg(f"Query {idx} -> {result}")
            if result is None:
                state.setdefault("errors", []).append(f"SPARQL error for query {idx}")
        state["results"] = results
        return state

    def verdicts(state: GraphState) -> GraphState:
        per_claim: List[str] = []
        for i, result in enumerate(state.get("results", [])):
            if result is True:
                verdict = "SUPPORTED"
            elif result is False:
                verdict = "REFUTED"
            else:
                verdict = "NOT_FOUND"
            per_claim.append(verdict)
            claim = state.get("claims", [None] * (i + 1))[i]
            log_dbg(f"Claim {i} verdict={verdict} ({claim!r})")
        state["claim_verdicts"] = per_claim

        if "REFUTED" in per_claim:
            overall = "Refuted"
        elif (
            "SUPPORTED" in per_claim
            and per_claim.count("SUPPORTED") >= max(1, len(per_claim) / 3)
        ):
            overall = "Supported"
        elif per_claim and all(v == "NOT_FOUND" for v in per_claim):
            overall = "Not Found"
        else:
            overall = "Uncertain"
        state["verdict"] = overall
        log_dbg(f"Overall verdict: {overall}")
        return state

    graph = StateGraph(GraphState)
    graph.add_node("router", router)
    graph.add_node("rephrase", rephrase_answer)
    graph.add_node("skip_rephrase", skip_rephrase)
    graph.add_node("split", split_response_into_claims)
    graph.add_node("parse_entities", parse_entities)
    graph.add_node("parse_relations", parse_relations)
    graph.add_node("find_entities", find_entities)
    graph.add_node("find_relations", find_relations)
    graph.add_node("query", create_query)
    graph.add_node("run", run_query)
    graph.add_node("verdicts", verdicts)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_rephrase,
        {"rephrase": "rephrase", "skip_rephrase": "skip_rephrase"},
    )
    graph.add_edge("rephrase", "split")
    graph.add_edge("skip_rephrase", "split")
    graph.add_edge("split", "parse_entities")
    graph.add_edge("parse_entities", "parse_relations")
    graph.add_edge("parse_relations", "find_entities")
    graph.add_edge("find_entities", "find_relations")
    graph.add_edge("find_relations", "query")
    graph.add_edge("query", "run")
    graph.add_edge("run", "verdicts")
    graph.add_edge("verdicts", END)
    return graph.compile()
