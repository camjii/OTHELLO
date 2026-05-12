"""Evaluation helpers: run verification on dataframes and compute metrics."""

from othello.evaluation.verdicts import (
    PipelineResult,
    compare_runs,
    evaluate_dataframe,
    run_verification_on_datasets,
    summarize_results,
)

__all__ = [
    "PipelineResult",
    "compare_runs",
    "evaluate_dataframe",
    "run_verification_on_datasets",
    "summarize_results",
]
