#!/usr/bin/env bash
# Download every supported QA benchmark into <repo>/datasets/.
# Supported: QALD, Mintaka, HotpotQA, TriviaQA, WebQuestions,
#            SimpleQuestions, LC-QuAD 2.0, Natural Questions, FreebaseQA.
set -euo pipefail

cd "$(dirname "$0")/.."   # always run from the repo root
mkdir -p datasets translations models

python - <<'PY'
from othello.datasets import download_all
for name, path in download_all().items():
    print(f"{name:18s} -> {path}")
PY
