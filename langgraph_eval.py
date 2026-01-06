
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from SPARQLWrapper import SPARQLWrapper, JSON
import os
from dotenv import load_dotenv
import json
import requests

load_dotenv()
key = os.getenv("OPENAI_API_KEY")


def build_verification_graph():
    
    class GraphState(TypedDict):
        question: str
        answer: str
        rephrased_claim: str
        claims: List[str]
        entities: List[Any]
        queries: List[str]
        results: List[Any]
        verdict: str

   
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=key,
    )


    def rephrase_answer(state: GraphState) -> GraphState:
        prompt = """
        Use the question to rephrase a single-word or single-sentence response into a full claim.
        Example:
        Question: Who founded McDonald's and when?
        Response: Ray Kroc
        Rephrased Claim: Ray Kroc joined McDonald's in 1954.
        """

        response = llm.invoke(
            [
                ("system", prompt),
                ("user", f"Question: {state['question']} Response: {state['answer']} Rephrased Claim:")
            ]
        )

        state["rephrased_claim"] = response.content.strip()
        return state

    def split_response_into_claims(state: GraphState) -> GraphState:
        prompt = """
        Parse the response into atomic claims.
        Output ONLY a JSON array of strings, with no additional text or markdown formatting.
        Example: ["claim 1", "claim 2", "claim 3"]
        """

        response = llm.invoke(
            [
                ("system", prompt),
                ("user", f"Response: {state['rephrased_claim']}")
            ]
        )

        # Safely parse JSON instead of using eval
        try:
            content = response.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            state["claims"] = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: treat the entire response as a single claim
            state["claims"] = [state["rephrased_claim"]]
        
        return state

    def find_entities(state: GraphState) -> GraphState:
        
        entities = [] #claim 0, claim 1
        for claim in state['claims']:

        
        
            data = {"text": claim}
            url = "https://labs.tib.eu/falcon/falcon2/api?mode=long"
            response = requests.post(url, json=data)
            entities.append(response)
        state['entities'] = entities
        return state


   
    def extract_entities_relations(state:GraphState) -> GraphState:
        ent_dict, rel_dict = {}, {} 
        for idx, entity in enumerate(state['entities']): #each entity json 
            ent = entity['entities_wikidata']
            rel = entity['relations wikidata']
            
            parsed_ents = []
            for i in range(len(ent)): #for each entity
                uri = ent[i]['URI'].split('/') #get uri 
                parsed_ents.append(uri[4]) #append 
            ent_dict[idx] = parsed_ents

            parsed_rels = []
            for i in range(len(rel)): #for each entity
                uri = rel[i]['URI'].split('/') #get uri 
                parsed_rels.append(uri[4]) #append 
            rel_dict[idx] = parsed_rels
        state['parsed_entities'] = ent_dict
        state['parsed_relations'] = rel_dict
        return state



    def create_query(state: GraphState) -> GraphState:
        queries = []
    

        for claim in state["claims"]:
            prompt = """
            Generate an ASK SPARQL query for Wikidata to verify this claim.

            Important:
            - Use wdt: for properties
            - Use wd: for entities
            - Include SERVICE wikibase:label if needed
            - Return ONLY the SPARQL query, no explanations or markdown
            - The query should return a boolean (true/false)
            """

            response = llm.invoke(
                [
                    ("system", prompt),
                    ("user", f"Claim: {claim}")
                ]
            )

            query = response.content.strip()
            # Remove markdown code blocks if present
            if query.startswith("```"):
                query = "\n".join(query.split("\n")[1:-1])
            
            queries.append(query)

        state["queries"] = queries
        return state

    def run_query(state: GraphState) -> GraphState:
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)
        results = []

        for query in state["queries"]:
            try:
                sparql.setQuery(query)
                result = sparql.query().convert()
                # ASK queries return a boolean in the 'boolean' field
                if 'boolean' in result:
                    results.append(result['boolean'])
                else:
                    results.append(None)
            except Exception as e:
                print(f"Query failed: {e}")
                results.append(None)

        state["results"] = results
        return state

    def return_majority(state: GraphState) -> GraphState:
        
        valid_results = [r for r in state["results"] if r is not None]
        if not valid_results:
            state["verdict"] = "Unable to verify"
        else:
            true_count = sum(1 for r in valid_results if r)
            false_count = len(valid_results) - true_count
            state["verdict"] = "True" if true_count >= false_count else "False"
        return state

    def router(state: GraphState) -> GraphState:
        return state

    def route_rephrase(state: GraphState) -> str:
        if len(state["answer"].split(".")) == 1:
            return "rephrase"
        else:
            return "skip_rephrase"

    def skip_rephrase(state: GraphState) -> GraphState:
        state["rephrased_claim"] = state["answer"]
        return state

    # Build the graph
    graph = StateGraph(GraphState)

    graph.add_node("router", router)
    graph.add_node("rephrase", rephrase_answer)
    graph.add_node("skip_rephrase", skip_rephrase)
    graph.add_node("split", split_response_into_claims)
    graph.add_node("find_entities", find_entities)
    graph.add_node("extract_entities_relations", extract_entities_relations)
    graph.add_node("query", create_query)
    graph.add_node("run", run_query)
    graph.add_node("vote", return_majority)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_rephrase,
        {
            "rephrase": "rephrase",
            "skip_rephrase": "skip_rephrase",
        },
    )

    graph.add_edge("rephrase", "split")
    graph.add_edge("skip_rephrase", "split")
    graph.add_edge("split", "find_entities")
    graph.add_edge('find_entities', 'extract_entities_relations')
    graph.add_edge("extract_entities_relations", "query")
    graph.add_edge("query", "run")
    graph.add_edge("run", "vote")
    graph.add_edge("vote", END)

    # Compile the graph
    return graph.compile()

# Build and compile the graph
app = build_verification_graph()

#print mermaid chart
# print(app.get_graph().draw_mermaid())

# question = "Who is the author of \"One Hundred Years of Solitude\"?"
# answer = "Gabriel García Márquez"



# # Run with initial state
# initial_state = {
#     "question": question,
#     "answer": answer,
#     "rephrased_claim": "",
#     "claims": [],
#     "entities": [],
#     "queries": [],
#     "results": [],
#     "verdict": ""
# }



# result = app.invoke(initial_state)
# print(result)



print(app.get_graph().draw_mermaid())