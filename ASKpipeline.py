from typing import TypedDict, List, Optional, Dict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import json
import requests
import re
import os
import math
from dotenv import load_dotenv
from time import sleep

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

    # ---------- IMPROVED HELPERS ----------
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
    entity_cache = {}
    prop_cache = {}

    def _wb_search_entity(name):
        """Improved entity search with better error handling"""
        name = _normalize_label(name)
        if not name:
            return None
        if name in entity_cache:
            return entity_cache[name]

        url = "https://www.wikidata.org/w/api.php"
        headers = {"User-Agent": "cekruger (cekruger99@gmail.com)"}
        params = {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "type": "item",
            "format": "json",
            "limit": 5,
        }

        try:
            sleep(0.1)  # Rate limiting
            r = session.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            hits = r.json().get("search", [])
            qid = hits[0]["id"] if hits else None
            
            if debug and qid:
                debug_fn(f"Entity '{name}' -> {qid} ({hits[0].get('label', 'N/A')})")
        except Exception as e:
            if debug:
                debug_fn(f"Entity search failed for '{name}': {e}")
            qid = None

        entity_cache[name] = qid
        return qid

    def _wb_search_property(name):
        """Improved property search with better error handling"""
        name = _normalize_label(name)
        if not name:
            return None
        if name in prop_cache:
            return prop_cache[name]

        url = "https://www.wikidata.org/w/api.php"
        headers = {"User-Agent": "cekruger (cekruger99@gmail.com)"}
        params = {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "type": "property",
            "format": "json",
            "limit": 5,
        }

        try:
            sleep(0.1)  # Rate limiting
            r = session.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            hits = r.json().get("search", [])
            pid = hits[0]["id"] if hits else None
            
            if debug and pid:
                debug_fn(f"Property '{name}' -> {pid} ({hits[0].get('label', 'N/A')})")
        except Exception as e:
            if debug:
                debug_fn(f"Property search failed for '{name}': {e}")
            pid = None

        prop_cache[name] = pid
        return pid

    # ---------- IMPROVED NODES ----------
    def router(state):
        state.setdefault("errors", [])
        return state

    def route_rephrase(state):
        answer = _safe_text(state.get("answer", ""))
        # More robust check for whether rephrasing is needed
        words = answer.split()
        if len(words) <= 5 or "." not in answer:
            return "rephrase"
        return "skip_rephrase"

    def rephrase_answer(state):
        """IMPROVED: Better prompt for rephrasing"""
        prompt = """You are rephrasing a short answer into a complete, verifiable claim.

Rules:
- Convert the answer into a complete sentence
- Include the subject from the question
- Be precise and factual
- Do not add information not present in the answer
- Return ONLY the rephrased claim, nothing else

Example:
Question: "What time zone is Salt Lake City in?"
Answer: "Mountain Time"
Rephrased: "Salt Lake City is in the Mountain Time zone."
"""
        response = llm.invoke(
            [
                ("system", prompt),
                ("user", f"Question: {state['question']}\nAnswer: {state['answer']}"),
            ]
        )
        state["rephrased_claim"] = response.content.strip()
        
        if debug:
            debug_fn(f"Rephrased: '{state['answer']}' -> '{state['rephrased_claim']}'")
        
        return state

    def skip_rephrase(state):
        state["rephrased_claim"] = _safe_text(state.get("answer", "")).strip()
        return state

    def split_response_into_claims(state):
        """IMPROVED: Better claim splitting with examples"""
        prompt = """Split the response into atomic, verifiable claims.

Rules:
- Each claim should state ONE fact
- Each claim should be independently verifiable
- Keep claims simple and direct
- Preserve all factual information
- Output ONLY a valid JSON array of strings

Example:
Input: "Albert Einstein had two sons, Hans Albert and Eduard, with his first wife Mileva Marić."
Output: ["Albert Einstein had a son named Hans Albert", "Albert Einstein had a son named Eduard", "Albert Einstein's first wife was Mileva Marić"]
"""
        response = llm.invoke(
            [("system", prompt), ("user", f"Response: {state['rephrased_claim']}")]
        )
        
        claims = _safe_json_list(response.content, [state["rephrased_claim"]])
        
        if debug:
            debug_fn(f"Split into {len(claims)} claims: {claims}")
        
        state["claims"] = claims
        return state

    def parse_entities(state):
        """IMPROVED: Better entity extraction with context"""
        parsed = {}
        prompt = """Extract the named entities (people, places, organizations, concepts) from this claim.

Rules:
- Extract proper nouns and key concepts
- Use full names when available
- Separate different entities
- Output ONLY a JSON array of entity names as strings

Examples:
Claim: "Salt Lake City is in the Mountain Time zone"
Output: ["Salt Lake City", "Mountain Time"]

Claim: "Albert Einstein had a son named Hans Albert"
Output: ["Albert Einstein", "Hans Albert Einstein"]
"""
        for i, claim in enumerate(state["claims"]):
            resp = llm.invoke([("system", prompt), ("user", f"Claim: {claim}")])
            entities = [_normalize_label(e) for e in _safe_json_list(resp.content)]
            parsed[i] = entities
            
            if debug:
                debug_fn(f"Claim {i}: '{claim}' -> Entities: {entities}")
        
        state["parsed_entities"] = parsed
        return state

    def parse_relations(state):
        """IMPROVED: Better relation extraction"""
        parsed = {}
        prompt = """Extract the relationship/property being described in this claim.

Think about what Wikidata property would connect the entities.

Common properties:
- "located in time zone" / "time zone"
- "child" / "parent"
- "spouse" / "married to"
- "occupation" / "profession"
- "country" / "located in"

Rules:
- Extract the relationship as a Wikidata-style property name
- Be specific but use common Wikidata terminology
- Output ONLY a JSON array with one property name

Examples:
Claim: "Salt Lake City is in the Mountain Time zone"
Output: ["time zone"]

Claim: "Albert Einstein had a son named Hans Albert"
Output: ["child"]
"""
        for i, claim in enumerate(state["claims"]):
            resp = llm.invoke([("system", prompt), ("user", f"Claim: {claim}")])
            relations = [_normalize_label(r) for r in _safe_json_list(resp.content)]
            parsed[i] = relations
            
            if debug:
                debug_fn(f"Claim {i}: Relations: {relations}")
        
        state["parsed_relations"] = parsed
        return state

    def find_entities(state):
        """IMPROVED: Better entity linking with fallbacks"""
        out = {}
        for i, ents in state.get("parsed_entities", {}).items():
            qids = []
            for e in ents:
                qid = _wb_search_entity(e)
                if qid:
                    qids.append(qid)
                elif debug:
                    debug_fn(f"WARNING: Could not find entity '{e}'")
            out[i] = qids
        
        state["entity_qids"] = out
        return state

    def find_relations(state):
        """IMPROVED: Better property linking"""
        out = {}
        for i, rels in state.get("parsed_relations", {}).items():
            pids = []
            for r in rels:
                pid = _wb_search_property(r)
                if pid:
                    pids.append(pid)
                elif debug:
                    debug_fn(f"WARNING: Could not find property '{r}'")
            out[i] = pids
        
        state["relation_pids"] = out
        return state

    def create_query(state):
        """IMPROVED: Better query generation with validation"""
        queries = []
        skipped = []
        
        for i in range(len(state["claims"])):
            qids = state.get("entity_qids", {}).get(i, [])
            pids = state.get("relation_pids", {}).get(i, [])
            
            print(f'qids contains: {type(qids[0])}')

            print(f'pids contains: {type(pids[0])}')

            if len(qids) < 2:
                skipped.append((i, f"Insufficient entities ({len(qids)})"))
                queries.append(None)
                continue
            
            if not pids:
                skipped.append((i, "No properties found"))
                queries.append(None)
                continue

            # a, b = qids[:2] #first 2 entities
            # p = pids[0]

            for e1 in qids:
                for e2 in qids:
                    for r in pids:
                        queries.append('{{ wd:{' + e1 + '} wdt:{' + r + '} wd:{' + e2 + '} . }}')
            
            print(f'len(queries) = {len(queries)}')
            
            query = 'ASK WHERE {{ '
            for q in queries:
                query += q + ' UNION '
            
            query = query[:-7] + '}}'

            print(query)


        
#             query = f"""ASK WHERE {{
#   {{ wd:{a} wdt:{p} wd:{b} . }}
#   UNION
#   {{ wd:{b} wdt:{p} wd:{a} . }}
# }}"""
            
            # queries.append(query)
            
            # if debug:
            #     debug_fn(f"Query {i}: {a} --{p}-- {b}")

        if debug and skipped:
            debug_fn(f"Skipped {len(skipped)} queries: {skipped}")

        state["queries"] = queries
        return state

    def run_query(state):
        """IMPROVED: Better query execution with error handling"""
        url = "https://query.wikidata.org/sparql"
        results = []
        
        for idx, query in enumerate(state.get("queries", [])):
            if query is None:
                results.append(None)
                continue
            
            try:
                sleep(0.2)  # Rate limiting
                r = requests.get(
                    url,
                    params={"query": query, "format": "json"},
                    headers={"User-Agent": "cekruger (cekruger99@gmail.com)"},
                    timeout=30
                )
                r.raise_for_status()
                data = r.json()
                result = bool(data.get("boolean", False))
                results.append(result)
                
                if debug:
                    debug_fn(f"Query {idx} result: {result}")
                    
            except Exception as e:
                if debug:
                    debug_fn(f"Query {idx} failed: {e}")
                results.append(None)
                state.setdefault("errors", []).append(
                    f"SPARQL error for query {idx}: {str(e)}"
                )

        state["results"] = results
        return state

    def verdicts(state):
        """IMPROVED: More nuanced verdict logic"""
        verdicts = []
        for i, r in enumerate(state.get("results", [])):
            if r is True:
                verdict = "SUPPORTED" #pipeline worked correctly


            elif r is False:
                verdict = "REFUTED"
            else:
                verdict = "NOT_FOUND" #if query is malformed or part of the verification is missing 
            verdicts.append(verdict)
            
            if debug:
                claim = state.get("claims", [None] * (i + 1))[i]
                debug_fn(f"Claim {i} verdict: {verdict} - '{claim}'")

        state["claim_verdicts"] = verdicts

        
      



        
        # Overall verdict logic
        if "REFUTED" in verdicts:
            state["verdict"] = "Refuted"
        elif "SUPPORTED" in verdicts and verdicts.count("SUPPORTED") >= len(verdicts) / 3:
            state["verdict"] = "Supported"
        elif all(v == "NOT_FOUND" for v in verdicts):
            state["verdict"] = "Not Found"
        else:
            # Mixed results - be conservative
            state["verdict"] = "Uncertain"
        
        if debug:
            debug_fn(f"Overall verdict: {state['verdict']}")
        
        return state

    # ---------- GRAPH WIRING ----------
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

