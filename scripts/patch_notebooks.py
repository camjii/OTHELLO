"""One-shot helper that rewrites notebook paths and imports after the move
from the project root into ``notebooks/``. Idempotent — running it twice is
safe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PATH_REPLACEMENTS = {
    "./aave_translations/": "../data/aave/",
    "./sae_results/": "../data/sae/",
    "./sparql_results/": "../data/sparql/",
    "./processed_names/": "../data/processed_names/",
    "./LLM_answers/": "../data/llm_answers/",
    "./final_results/": "../results/",
    "./models/": "../models/",
    "./hf_home": "../hf_home",
    "./datasets/": "../datasets/",
    "datasets/qald.json": "../datasets/qald.json",
    "datasets/hotpot.json": "../datasets/hotpot.json",
    "datasets/mintaka.json": "../datasets/mintaka.json",
    # legacy shims -> new package paths
    "from ASKpipeline import build_verification_graph":
        "from othello.pipelines.ask_verification import build_verification_graph",
    "from ASKpipeline import *":
        "from othello.pipelines.ask_verification import build_verification_graph",
    "from sparql_langgraph_pipeline import run_sparql_pipeline_batch":
        "from othello.pipelines.sparql_generation import run_sparql_pipeline_batch",
    "from sparql_langgraph_pipeline import *":
        "from othello.pipelines.sparql_generation import build_sparql_graph, run_sparql_pipeline_batch",
}

BOOTSTRAP_TAG = "# OTHELLO bootstrap: make the package importable from notebooks/"
BOOTSTRAP = (
    f"{BOOTSTRAP_TAG}\n"
    "import sys\n"
    "from pathlib import Path\n"
    "_repo_root = Path.cwd().parent.resolve()\n"
    "if str(_repo_root) not in sys.path:\n"
    "    sys.path.insert(0, str(_repo_root))\n"
)


def _apply_replacements(text: str) -> str:
    for old, new in PATH_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def patch_cell(cell: dict) -> bool:
    if cell.get("cell_type") not in {"code", "markdown"}:
        return False
    source = cell.get("source", "")
    if isinstance(source, list):
        joined = "".join(source)
    else:
        joined = source
    new = _apply_replacements(joined)
    if new == joined:
        return False
    cell["source"] = new.splitlines(keepends=True)
    return True


def has_bootstrap(notebook: dict) -> bool:
    for cell in notebook.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if BOOTSTRAP_TAG in src:
            return True
    return False


def make_bootstrap_cell() -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": BOOTSTRAP.splitlines(keepends=True),
    }


def patch_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in notebook.get("cells", []):
        if patch_cell(cell):
            changed = True

    if not has_bootstrap(notebook):
        notebook.setdefault("cells", []).insert(0, make_bootstrap_cell())
        changed = True

    if changed:
        path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return changed


def main(targets: list[str] | None = None) -> int:
    notebooks_dir = Path(__file__).resolve().parent.parent / "notebooks"
    paths = (
        [notebooks_dir / t for t in targets]
        if targets
        else sorted(notebooks_dir.glob("*.ipynb"))
    )
    for path in paths:
        if not path.exists():
            print(f"skip (missing): {path}")
            continue
        changed = patch_notebook(path)
        print(f"{'patched' if changed else 'unchanged'}: {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
