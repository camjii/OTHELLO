"""Backwards-compatible shim for the SPARQL generation pipeline.

The implementation lives in :mod:`othello.pipelines.sparql_generation`. The
module-level ``app`` is materialized lazily on first access so the import is
cheap and does not require ``OPENAI_API_KEY`` to be set.
"""

from othello.pipelines.sparql_generation import (
    SparqlGraphState as GraphState,
    build_sparql_graph,
    run_sparql_pipeline_batch,
)


class _LazyApp:
    """Compile the LangGraph workflow on first attribute access."""

    def __init__(self) -> None:
        self._app = None

    def _ensure(self):
        if self._app is None:
            self._app = build_sparql_graph()
        return self._app

    def __getattr__(self, name):
        return getattr(self._ensure(), name)

    def invoke(self, *args, **kwargs):
        return self._ensure().invoke(*args, **kwargs)


app = _LazyApp()

__all__ = ["GraphState", "app", "build_sparql_graph", "run_sparql_pipeline_batch"]
