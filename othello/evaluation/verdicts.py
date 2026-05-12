"""Compute factual-accuracy / hallucination-rate metrics from the ASK pipeline."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Per-dataset summary suitable for direct JSON serialization."""

    method: str
    raw_data: Dict[str, int]
    performance_metrics: Dict[str, Optional[float]]
    sample_outputs: Dict[str, List[Dict[str, str]]]
    detailed_results: List[Optional[bool]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_dataframe(
    df: pd.DataFrame,
    *,
    graph,
    method: str = "pipeline",
    question_col: str = "SAE Question",
    answer_col: str = "Answer",
    max_examples: int = 3,
    show_progress: bool = True,
) -> PipelineResult:
    """Verify each row of ``df`` and roll the verdicts into a ``PipelineResult``."""
    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(list(iterator), desc=f"Verify ({method})")

    detailed: List[Optional[bool]] = []
    factual_examples: List[Dict[str, str]] = []
    hallucination_examples: List[Dict[str, str]] = []
    factual = 0
    hallucinated = 0
    verified = 0

    for _, row in iterator:
        state = {
            "question": row.get(question_col, ""),
            "answer": row.get(answer_col, ""),
        }
        try:
            out = graph.invoke(state)
        except Exception as exc:
            log.warning("Verification crashed on %r: %s", state.get("question"), exc)
            detailed.append(None)
            continue

        results = out.get("results") or []
        valid = [r for r in results if r is not None]
        if not valid:
            detailed.append(None)
            continue

        is_factual = any(valid)
        detailed.append(is_factual)
        verified += 1
        sample = {"question": str(row.get(question_col)), "answer": str(row.get(answer_col))}
        if is_factual:
            factual += 1
            if len(factual_examples) < max_examples:
                factual_examples.append(sample)
        else:
            hallucinated += 1
            if len(hallucination_examples) < max_examples:
                hallucination_examples.append(sample)

    if verified:
        factual_pct = factual / verified * 100
        hallucination_pct = hallucinated / verified * 100
    else:
        factual_pct = hallucination_pct = None

    return PipelineResult(
        method=method,
        raw_data={
            "factual_count": factual,
            "hallucination_count": hallucinated,
            "total_verified": verified,
        },
        performance_metrics={
            "factual_accuracy_pct": factual_pct,
            "hallucination_rate_pct": hallucination_pct,
        },
        sample_outputs={
            "factual_examples": factual_examples,
            "hallucination_examples": hallucination_examples,
        },
        detailed_results=detailed,
    )


def run_verification_on_datasets(
    datasets: Dict[str, pd.DataFrame],
    *,
    build_graph_fn: Optional[Callable] = None,
    question_col: str = "SAE Question",
    answer_col: str = "Answer",
    method: str = "pipeline",
    show_progress: bool = True,
) -> Dict[str, PipelineResult]:
    """Convenience: evaluate a dict of dataframes with one shared graph."""
    if build_graph_fn is None:
        from othello.pipelines.ask_verification import build_verification_graph as build_graph_fn
    graph = build_graph_fn()

    return {
        name: evaluate_dataframe(
            df,
            graph=graph,
            method=method,
            question_col=question_col,
            answer_col=answer_col,
            show_progress=show_progress,
        )
        for name, df in datasets.items()
    }


def summarize_results(results: Dict[str, PipelineResult]) -> Dict[str, Dict[str, Any]]:
    return {name: result.to_dict() for name, result in results.items()}


def compare_runs(
    vanilla: Dict[str, PipelineResult],
    pipeline: Dict[str, PipelineResult],
) -> Dict[str, Dict[str, Any]]:
    """Compute deltas between a vanilla baseline and the grounded pipeline."""
    comparison: Dict[str, Dict[str, Any]] = {}
    for name, vanilla_res in vanilla.items():
        pipeline_res = pipeline.get(name)
        if pipeline_res is None:
            continue
        v_factual = vanilla_res.performance_metrics["factual_accuracy_pct"]
        p_factual = pipeline_res.performance_metrics["factual_accuracy_pct"]
        v_halluc = vanilla_res.performance_metrics["hallucination_rate_pct"]
        p_halluc = pipeline_res.performance_metrics["hallucination_rate_pct"]
        if None in (v_factual, p_factual, v_halluc, p_halluc):
            continue

        factual_delta = p_factual - v_factual
        halluc_delta = v_halluc - p_halluc
        comparison[name] = {
            "factual_accuracy": {
                "vanilla_pct": v_factual,
                "pipeline_pct": p_factual,
                "absolute_improvement_pct": factual_delta,
                "relative_improvement_pct": (factual_delta / v_factual * 100) if v_factual else 0,
            },
            "hallucination_rate": {
                "vanilla_pct": v_halluc,
                "pipeline_pct": p_halluc,
                "absolute_reduction_pct": halluc_delta,
                "relative_reduction_pct": (halluc_delta / v_halluc * 100) if v_halluc else 0,
            },
            "raw_counts": {
                "vanilla": vanilla_res.raw_data,
                "pipeline": pipeline_res.raw_data,
            },
        }
    return comparison
