"""FLAN-T5 inference helpers used by the OTHELLO experiments.

Two modes are supported:
    * Grounded: the answer must come from the supplied ``context``.
    * Vanilla:  the model relies on its parametric knowledge alone.

The same class handles all variants (``small``, ``base``, ``large``); the
``mode`` flag selects the prompt template.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Literal, Optional

import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

Mode = Literal["grounded", "vanilla"]

GROUNDED_PROMPT = """Context: {context}

Question: {question}

Answer the question ONLY using the context provided, don't rely on your own \
knowledge. Only make claims that can be supported by context."""

VANILLA_PROMPT = """Question: {question}

Answer the given question to the best of your knowledge. Provide a succinct \
and clear short answer."""


class FlanT5Answerer:
    """Wraps a FLAN-T5 checkpoint and produces short-form answers."""

    def __init__(
        self,
        model_path: str | Path = "./models/google/flan-t5-base",
        *,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_input_length: int = 512,
        max_new_tokens: int = 100,
        num_beams: int = 4,
        load: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self._tokenizer = None
        self._model = None
        if load:
            self._load()

    def _load(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            local_files_only=True,
        )

    def _ensure_loaded(self) -> None:
        if self._model is None or self._tokenizer is None:
            self._load()

    def answer(
        self,
        question: str,
        context: Optional[str] = None,
        *,
        mode: Mode = "grounded",
    ) -> str:
        self._ensure_loaded()
        if mode == "grounded":
            prompt = GROUNDED_PROMPT.format(context=context or "", question=question)
        else:
            prompt = VANILLA_PROMPT.format(question=question)

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            early_stopping=True,
            do_sample=False,
        )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def answer_many(
        self,
        questions: Iterable[str],
        contexts: Optional[Iterable[Optional[str]]] = None,
        *,
        mode: Mode = "grounded",
        show_progress: bool = True,
    ) -> List[str]:
        questions = list(questions)
        contexts = list(contexts) if contexts is not None else [None] * len(questions)
        iterator = zip(questions, contexts)
        if show_progress:
            iterator = tqdm(list(iterator), desc=f"FLAN-T5 ({mode})")
        return [self.answer(q, c, mode=mode) for q, c in iterator]


def answer_dataframe(
    df: pd.DataFrame,
    answerer: FlanT5Answerer,
    *,
    question_col: str = "SAE Question",
    context_col: Optional[str] = "names",
    output_col: str = "Answer",
    mode: Mode = "grounded",
) -> pd.DataFrame:
    """Answer every row of ``df``, attaching the result as ``output_col``."""
    questions = df[question_col].astype(str).tolist()
    contexts = df[context_col].tolist() if (context_col and context_col in df) else None
    df = df.copy()
    df[output_col] = answerer.answer_many(questions, contexts, mode=mode)
    return df
