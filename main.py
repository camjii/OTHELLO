"""Command-line entry point for OTHELLO.

Available subcommands:
    download     - Fetch one or all supported QA benchmarks.
    list         - List supported datasets and their local cache paths.
    sparql       - Generate SPARQL queries for a dataset's questions.
    verify       - Verify (question, answer) pairs against Wikidata.

Examples:
    python main.py list
    python main.py download --name triviaqa
    python main.py download --all
    python main.py sparql --dataset qald --limit 20 --out results.json
    python main.py verify --input answers.csv --out verdicts.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from othello.datasets import (
    DATASET_LOADERS,
    DATASET_URLS,
    download_all,
    download_dataset,
    load_dataset,
)
from othello.evaluation import evaluate_dataframe, summarize_results
from othello.pipelines import build_verification_graph, run_sparql_pipeline_batch


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    )


def cmd_list(_: argparse.Namespace) -> int:
    print("Supported datasets:")
    for name in sorted(DATASET_LOADERS):
        print(f"  - {name:18s}  {DATASET_URLS[name]}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    target = Path(args.data_dir)
    if args.all:
        for name, path in download_all(target).items():
            print(f"{name:18s} -> {path}")
        return 0
    if not args.name:
        print("--name or --all is required", file=sys.stderr)
        return 2
    path = download_dataset(args.name, target)
    print(f"{args.name} -> {path}")
    return 0


def cmd_sparql(args: argparse.Namespace) -> int:
    df = load_dataset(args.dataset, data_dir=args.data_dir, limit=args.limit)
    questions = df["question"].dropna().astype(str).tolist()
    results = run_sparql_pipeline_batch(
        questions,
        max_attempts=args.max_attempts,
        max_workers=args.max_workers,
    )
    payload = [
        {
            "question": q,
            "sparql_query": r.get("sparql_query", ""),
            "attempts": r.get("attempt", 0),
            "error": r.get("error", ""),
            "results": r.get("results", {}),
        }
        for q, r in zip(questions, results)
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(payload)} rows -> {args.out}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    if args.input.suffix == ".csv":
        df = pd.read_csv(args.input)
    elif args.input.suffix in {".json", ".jsonl"}:
        df = pd.read_json(args.input, lines=args.input.suffix == ".jsonl")
    else:
        print(f"Unsupported input format: {args.input.suffix}", file=sys.stderr)
        return 2

    graph = build_verification_graph(debug=args.debug)
    result = evaluate_dataframe(
        df,
        graph=graph,
        method=args.method,
        question_col=args.question_col,
        answer_col=args.answer_col,
    )

    payload = summarize_results({args.input.stem: result})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    metrics = result.performance_metrics
    print(
        f"Verified {result.raw_data['total_verified']} rows | "
        f"factual={metrics['factual_accuracy_pct']:.2f}% | "
        f"hallucination={metrics['hallucination_rate_pct']:.2f}%"
    )
    print(f"Wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="othello", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List supported datasets").set_defaults(func=cmd_list)

    p_dl = sub.add_parser("download", help="Download one or all datasets")
    p_dl.add_argument("--name", choices=sorted(DATASET_URLS))
    p_dl.add_argument("--all", action="store_true", help="Download every supported dataset")
    p_dl.add_argument("--data-dir", default="datasets", help="Where to cache files")
    p_dl.set_defaults(func=cmd_download)

    p_sq = sub.add_parser("sparql", help="Generate SPARQL queries for a dataset")
    p_sq.add_argument("--dataset", required=True, choices=sorted(DATASET_LOADERS))
    p_sq.add_argument("--data-dir", default="datasets")
    p_sq.add_argument("--limit", type=int, default=None, help="Only process N rows")
    p_sq.add_argument("--max-attempts", type=int, default=3)
    p_sq.add_argument("--max-workers", type=int, default=4)
    p_sq.add_argument("--out", default="sparql_results.json")
    p_sq.set_defaults(func=cmd_sparql)

    p_vf = sub.add_parser("verify", help="Verify (question, answer) pairs against Wikidata")
    p_vf.add_argument("--input", type=Path, required=True, help="CSV/JSON file with questions+answers")
    p_vf.add_argument("--question-col", default="SAE Question")
    p_vf.add_argument("--answer-col", default="Answer")
    p_vf.add_argument("--method", default="pipeline", help="Label written into the JSON output")
    p_vf.add_argument("--out", default="verdicts.json")
    p_vf.add_argument("--debug", action="store_true", help="Verbose verification logs")
    p_vf.set_defaults(func=cmd_verify)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
