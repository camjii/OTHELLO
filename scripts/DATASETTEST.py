"""Smoke test: end-to-end SPARQL generation for a small CSV of questions.

Reads ``Reverted_LC.csv``, runs the SPARQL pipeline over its ``SAE Question``
column, and writes ``Reverted_LC_processed.json`` next to it.

Run with::

    python DATASETTEST.py --input Reverted_LC.csv --output Reverted_LC_processed.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the package importable when this script is invoked from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from othello.pipelines import run_sparql_pipeline_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Reverted_LC.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("Reverted_LC_processed.json")
    )
    parser.add_argument("--question-col", default="SAE Question")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input CSV not found: {args.input}", file=sys.stderr)
        return 1

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = pd.read_csv(args.input)
    questions = df[args.question_col].astype(str).tolist()

    states = run_sparql_pipeline_batch(
        questions,
        max_attempts=args.max_attempts,
        max_workers=args.max_workers,
    )

    df["sparql_query"] = [s.get("sparql_query", "") for s in states]
    df["results"] = [s.get("results", {}) for s in states]
    df["error"] = [s.get("error", "") for s in states]
    df["attempt"] = [s.get("attempt", 0) for s in states]
    df.to_json(args.output, orient="records")
    print(f"Wrote {len(df)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
