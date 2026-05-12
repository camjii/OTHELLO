# OTHELLO

A research toolkit for measuring how the choice of dialect — African American
Vernacular English (AAVE) versus Standard American English (SAE) — affects the
factuality of LLM answers. The pipeline:

1. **Translates** every question to AAVE (OpenAI) and back to SAE (local Qwen3).
2. **Generates** a SPARQL query with an LLM and runs it against Wikidata to
   collect a grounding context.
3. **Answers** the question with FLAN-T5 (`small` / `base` / `large`) in two
   modes — *grounded* (context-aware) and *vanilla* (parametric only).
4. **Verifies** every claim by re-checking it as a Wikidata `ASK` query and
   reports factual accuracy versus hallucination rate.

## Repository layout

```
OTHELLO/
├── main.py                 CLI entry point (`python main.py …`)
├── pyproject.toml          Package metadata + `othello` console script
├── requirements.txt        Pinned dependencies
├── .env.example            OPENAI_API_KEY + HF_TOKEN template
│
├── othello/                The importable Python package
│   ├── pipelines/
│   │   ├── ask_verification.py    question + answer  → Wikidata ASK verdict
│   │   └── sparql_generation.py   question           → SPARQL → result
│   ├── datasets/loaders.py        9 QA benchmark loaders
│   ├── translation/               AAVE / SAE bidirectional translators
│   ├── qa/flan_t5.py              FLAN-T5 grounded + vanilla inference
│   ├── evaluation/verdicts.py     Factual accuracy & vanilla-vs-pipeline deltas
│   └── utils/                     Wikidata client, JSON/text helpers
│
├── notebooks/              Experiment notebooks (each begins with a bootstrap cell)
├── scripts/                gen_datasets.sh, DATASETTEST.py, patch_notebooks.py
│
├── datasets/               Raw benchmark downloads (populated by gen_datasets.sh)
├── data/                   Processed artifacts produced by the experiments
│   ├── aave/                 AAVE versions of QALD / Mintaka / HotpotQA
│   ├── sae/                  Back-translated SAE questions
│   ├── sparql/               SPARQL queries + raw Wikidata results
│   ├── processed_names/      Names parsed from Wikidata results (FLAN context)
│   └── llm_answers/          FLAN-T5 answers per variant × mode
└── results/                Final per-model verdicts and comparative analysis
```

The legacy modules `ASKpipeline.py` and `sparql_langgraph_pipeline.py` at the
repo root are kept as thin re-export shims for any external script that still
imports them; new code should import from `othello.*` directly.

## Supported datasets

| Name                | Source                                                                                            |
|---------------------|---------------------------------------------------------------------------------------------------|
| `qald`              | [QALD-9 Plus](https://github.com/KGQA/QALD_9_plus) (Wikidata test split)                          |
| `mintaka`           | [Mintaka](https://github.com/amazon-science/mintaka) (test split)                                 |
| `hotpot`            | [HotpotQA](https://hotpotqa.github.io/) (sample)                                                  |
| `triviaqa`          | [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) (RC sample)                                   |
| `webquestions`      | [WebQuestions](https://nlp.stanford.edu/software/sempre/)                                         |
| `simplequestions`   | [SimpleQuestions-Wikidata](https://github.com/askplatypus/wikidata-simplequestions)               |
| `lcquad`            | [LC-QuAD 2.0](https://github.com/AskNowQA/LC-QuAD2.0) (Wikidata)                                  |
| `naturalquestions`  | [Natural Questions Open](https://github.com/google-research-datasets/natural-questions)           |
| `freebaseqa`        | [FreebaseQA](https://github.com/kelvin-jiang/FreebaseQA)                                          |

Run `python main.py list` to see all of them.

## Installation

```bash
git clone <repo>
cd OTHELLO
uv sync                    # or: pip install -e .
cp .env.example .env       # then fill in OPENAI_API_KEY and HF_TOKEN
```

`OPENAI_API_KEY` powers the AAVE translator, claim splitter, and SPARQL
generator. `HF_TOKEN` is only needed if you want to (re)download the Qwen3 /
FLAN-T5 weights from the Hugging Face Hub.

## Quickstart

```bash
# Inspect supported benchmarks
python main.py list

# Download every benchmark into ./datasets/
./scripts/gen_datasets.sh           # or: python main.py download --all

# Download just one
python main.py download --name triviaqa

# Generate SPARQL queries for the first 25 QALD questions
python main.py sparql --dataset qald --limit 25 --out results/qald_sparql.json

# Verify (question, answer) pairs against Wikidata
python main.py verify --input data/llm_answers/flan-t5-base/LLM_Answers_qald.csv \
                      --out results/qald_verdicts.json
```

## Programmatic API

```python
from othello import build_verification_graph, run_sparql_pipeline_batch
from othello.datasets import load_dataset
from othello.evaluation import evaluate_dataframe

df = load_dataset("qald", limit=50)
states = run_sparql_pipeline_batch(df["question"].tolist(), max_workers=6)

graph = build_verification_graph()
result = evaluate_dataframe(answered_df, graph=graph, method="pipeline")
print(result.performance_metrics)
```

## Running the experiment notebooks

Each notebook under `notebooks/` begins with a bootstrap cell that adds the
project root to `sys.path` so the `othello` package is importable without
installation. Launch Jupyter from the repository root (`jupyter lab` /
`jupyter notebook`) and open any notebook — paths inside use relative
references like `../data/...` and `../results/...`.

If you move notebooks again or change directories, re-run
`python scripts/patch_notebooks.py` to refresh paths.

## License & citation

Research code released for review. If you use OTHELLO in academic work, please
cite the repository.
