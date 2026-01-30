from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import requests
import re
import os 
import math
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("OPENAI_API_KEY")


def build_verification_graph(debug: bool = False, debug_fn=print):
    class GraphState(TypedDict, total=False):
        question: str
        answer: str
        rephrased_claim: str
        claims: List[str]


        parsed_entities: dict
        parsed_relations: dict

        entity_qids: dict
        relation_pids: dict

        queries: List[Optional[str]]
        results: List[Optional[bool]]

        claim_verdicts: List[str]
        verdict: str
        errors: List[str]

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=key,
    )

    # ---------- helpers ----------
    def _strip_code_fences(s):
        s = s.strip()
        if s.startswith("```"):
            parts = s.split("```")
            if len(parts) >= 2:
                s = parts[1].strip()
                if s.lower().startswith("json"):
                    s = s[4:].strip()
        return s.strip()

    def _safe_json_list(s, fallback=None):
        fallback = fallback or []
        s = _strip_code_fences(s)
        try:
            v = json.loads(s)
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return [x.strip() for x in v if x.strip()]
        except Exception:
            pass
        return fallback

    def _normalize_label(s):
        s = s.strip().strip('"').strip("'")
        return re.sub(r"\s+", " ", s)

    def _safe_text(v):
        if v is None:
            return ""
        if isinstance(v, float) and math.isnan(v):
            return ""
        return str(v)

    session = requests.Session()
    entity_cache = {} #Builds a memory for searched entities
    prop_cache = {}

    def _wb_search_entity(name):
        name = _normalize_label(name)
        if not name:
            return None
        if name in entity_cache:
            return entity_cache[name]

        url = "https://www.wikidata.org/w/api.php"
        headers = {
                "User-Agent": "cekruger (cekruger99@gmail.com)"
        }
        params = {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "type": 'item',
            "format": "json",
            "limit": 5,
        }

        try:
            r = session.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            hits = r.json().get("search", [])
            qid = hits[0]["id"] if hits else None
        except Exception:
            qid = None

        entity_cache[name] = qid
        return qid

    def _wb_search_property(name):
        name = _normalize_label(name)
        if not name:
            return None
        if name in prop_cache:
            return prop_cache[name]

        url = "https://www.wikidata.org/w/api.php"
        headers = {
    "User-Agent": "cekruger (cekruger99@gmail.com)"
}
        params = {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "type": "property",
            "format": "json",
            "limit": 5,
        }

        try:
            r = session.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            hits = r.json().get("search", [])
            pid = hits[0]["id"] if hits else None
        except Exception:
            pid = None

        prop_cache[name] = pid
        return pid
    

    # ---------- nodes ----------
    def router(state):
        state.setdefault("errors", [])
        return state

    def route_rephrase(state):
        answer = _safe_text(state.get("answer", ""))
        if len(answer.split(".")) == 1:
            return "rephrase"
        return "skip_rephrase"

    def rephrase_answer(state):
        prompt = """
Use the question to rephrase a single-word or single-sentence response into a full claim.
Return ONLY the claim text.
"""
        response = llm.invoke(
            [
                ("system", prompt),
                ("user", f"Question: {state['question']}\nResponse: {state['answer']}"),
            ]
        )
        state["rephrased_claim"] = response.content.strip()
        return state

    def skip_rephrase(state):
        state["rephrased_claim"] = _safe_text(state.get("answer", "")).strip()
        return state

    def split_response_into_claims(state):
        prompt = """
Parse the response into atomic claims.
Output ONLY a JSON array of strings.
"""
        response = llm.invoke(
            [("system", prompt), ("user", f"Response: {state['rephrased_claim']}")]
        )
        claims = _safe_json_list(response.content, [state["rephrased_claim"]])
        state["claims"] = claims
        return state

    def parse_entities(state):
        parsed = {}
        prompt = """
Extract entity names needed to verify the claim.
Output ONLY a JSON array of strings.
"""
        for i, claim in enumerate(state["claims"]): 
            resp = llm.invoke([("system", prompt), ("user", f"Claim: {claim}")]) #Getting a response
            parsed[i] = [_normalize_label(e) for e in _safe_json_list(resp.content)] #Adding entities for the i-th claim to the dictionary as a list
        if debug:
            debug_fn("parsed_entities:", parsed)
        state["parsed_entities"] = parsed
        return state

    def parse_relations(state):
        parsed = {}
        prompt = """
Extract Wikidata-style property names.
Output ONLY a JSON array of strings.
"""
        for i, claim in enumerate(state["claims"]):
            resp = llm.invoke([("system", prompt), ("user", f"Claim: {claim}")])
            parsed[i] = [_normalize_label(r) for r in _safe_json_list(resp.content)]
        if debug:
            debug_fn("parsed_relations:", parsed)
        state["parsed_relations"] = parsed
        return state

    def find_entities(state):
        out = {}
        for i, ents in state.get("parsed_entities", {}).items():
            qids = [] #Creating a list for each entry in parsed_entities
            for e in ents:
                qid = _wb_search_entity(e)
                if qid:
                    qids.append(qid)
            out[i] = qids
        if debug:
            debug_fn("entity_qids:", out)
        state["entity_qids"] = out
        return state

    def find_relations(state):
        out = {}
        for i, rels in state.get("parsed_relations", {}).items():
            out[i] = [pid for r in rels if (pid := _wb_search_property(r))]
        if debug:
            debug_fn("relation_pids:", out)
        state["relation_pids"] = out
        return state

    def create_query(state):
        queries = []
        for i in range(len(state["claims"])): 
            qids = state.get("entity_qids", {}).get(i, [])
            pids = state.get("relation_pids", {}).get(i, []) #Getting entity_pids and relation_pids for each query

            if len(qids) < 2 or not pids:
                queries.append(None)
                continue

            a, b = qids[:2] #Using 2 qids for now, may need to increase
            p = pids[0]

            queries.append(
                f"""ASK WHERE {{
{{ wd:{a} wdt:{p} wd:{b} . }}
UNION
{{ wd:{b} wdt:{p} wd:{a} . }}
}}"""
            )
        
        queries = [q.replace('\n', '')  if q else None for q in queries]


        state["queries"] = queries
        if debug:
            debug_fn("queries:", queries)
        return state
    

    def run_query(state):
        url = "https://query.wikidata.org/sparql" 
        
        # sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        # # SPARQLWrapper versions vary; support both user-agent APIs.
        # if hasattr(sparql, "setUserAgent"):
        #     sparql.setUserAgent("cekruger (cekruger99@gmail.com)")
        # elif hasattr(sparql, "addCustomHttpHeader"):
        #     sparql.addCustomHttpHeader("User-Agent", "cekruger (cekruger99@gmail.com)")
        # sparql.setReturnFormat(JSON)

        results = []
        for query in state.get("queries", []):
            if not query:
                results.append(None)
                continue
            try:
                r = requests.get(url, params ={'query': query, 'format': 'json'})
                data = r.json()
                results.append(data['boolean'])
            except Exception:
                results.append(None)
                state.setdefault("errors", []).append(f"SPARQL error for query: {query}")

        state["results"] = results
        if debug:
            debug_fn("results:", results)
        return state

    def verdicts(state):
        verdicts = []
        for r in state.get("results", []):
            verdicts.append(
                "SUPPORTED" if r is True else "REFUTED" if r is False else "NOT_FOUND"
            )

        #Might have to mess around with this to get majority result
        state["claim_verdicts"] = verdicts
        state["verdict"] = (
            "Refuted"
            if "REFUTED" in verdicts
            else "Supported"
            if "SUPPORTED" in verdicts
            else "Not Found"
        )
        return state

    # ---------- graph wiring ----------
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
