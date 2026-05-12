"""OTHELLO: Observing The Hallucinations of Ebonics in LLM Outputs.

A research toolkit that compares LLM factual accuracy on Standard American
English (SAE) questions versus their African American Vernacular English
(AAVE) counterparts. Hallucinations are detected by grounding model
answers against Wikidata via a LangGraph-orchestrated SPARQL pipeline.
"""

from othello.pipelines.ask_verification import build_verification_graph
from othello.pipelines.sparql_generation import (
    build_sparql_graph,
    run_sparql_pipeline_batch,
)

__all__ = [
    "build_verification_graph",
    "build_sparql_graph",
    "run_sparql_pipeline_batch",
]

__version__ = "0.2.0"
