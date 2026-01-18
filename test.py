from ASKpipeline import build_verification_graph

app = build_verification_graph()


question = "Who was the author of 100 years of solitude?"
answer = "Gabriel Garcia Marquez"

initial_state = {
    "question": question,
    "answer": answer,
    "rephrased_claim": "",
    "claims": [],
    "entities": [],
    "parsed_entities": {},
    "parsed_relations": {},
    "queries": [],
    "results": [],
    "verdict": ""
}

result = app.invoke(initial_state)
print(result)


print(result['claim_verdicts'], result['verdict'])