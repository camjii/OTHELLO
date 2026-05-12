"""Backwards-compatible shim.

The implementation lives in :mod:`othello.pipelines.ask_verification`. This
module re-exports :func:`build_verification_graph` so existing notebooks like
``generate_verdicts.ipynb`` keep working with no changes.
"""

from othello.pipelines.ask_verification import build_verification_graph

__all__ = ["build_verification_graph"]
