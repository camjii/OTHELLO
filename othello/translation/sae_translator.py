"""Translate AAVE questions back into Standard American English with Qwen3 (or another local chat model)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

from tqdm import tqdm

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a strict translation engine. "
    "You MUST only output the translated SAE (Standard American English) questions. "
    "Do not explain, do not reason, do not add tags or extra text. "
    "Only return the final grammatically correct questions, nothing else.\n\n"
    "Examples:\n"
    "AAVE: Who be Einstein's ol' lady?\n"
    "SAE: Who was Einstein's wife?\n"
    "AAVE: When Obama come in the world?\n"
    "SAE: When was Obama born?\n"
    "AAVE: Where King Jr. do that 'I got a dream' speech at?\n"
    "SAE: Where did Martin Luther King Jr. give his 'I Have a Dream' speech?\n"
    "AAVE: What year Mandela start runnin' South Africa, and how long he stay holdin' that seat?\n"
    "SAE: In what year did Mandela become the President of South Africa, and how long did he remain in office?\n\n"
    "Remember: ONLY output the translated SAE questions, line by line, with no commentary."
)


class SAETranslator:
    """Wraps a HuggingFace chat-style model for AAVE → SAE translation."""

    def __init__(
        self,
        model_path: str | Path = "./models/Qwen3-8B",
        *,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        max_new_tokens: int = 250,
        load: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        if load:
            self._load()

    def _load(self) -> None:
        import torch  # local import keeps the package importable without torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.torch_dtype, torch.float16)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, fix_mistral_regex=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.device_map,
            local_files_only=True,
        )

    def translate(self, question: str) -> str:
        if self._model is None or self._tokenizer is None:
            self._load()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Transform this AAVE question into complete, grammatically perfect SAE. "
                    f"{question}"
                ),
            },
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0,
        )
        decoded = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        # Keep only the first non-empty translated line for downstream consistency.
        for line in decoded.split("\n"):
            if line.strip():
                return line.strip()
        return decoded.strip()

    def translate_many(
        self,
        questions: Iterable[str],
        *,
        show_progress: bool = True,
    ) -> List[str]:
        items = list(questions)
        iterator = tqdm(items, desc="SAE translation") if show_progress else items
        return [self.translate(q) for q in iterator]
