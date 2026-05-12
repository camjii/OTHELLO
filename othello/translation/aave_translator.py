"""Translate Standard American English questions into AAVE using OpenAI."""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

log = logging.getLogger(__name__)

load_dotenv()

DEFAULT_FEW_SHOT: Sequence[str] = (
    "I was bewildered, but I knew dat it was no gud asking his ass to explain.",
    "Cochran pontificated windily for da camera.",
    "I don’t want them to follow in my footsteps, as I ain’t go to no college, "
    "but I want them to go.",
)

SYSTEM_PROMPT = (
    "You are a helpful assistant that translates Standard American English (SAE) "
    "to African American Vernacular English (AAVE)."
)


def _build_messages(text: str, few_shot: Sequence[str]) -> list[dict]:
    examples = "\n".join(f"{i + 1}. {ex}" for i, ex in enumerate(few_shot))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Translate the following sentence from Standard English to "
                "African American Vernacular English (AAVE). Ensure the "
                "translation maintains the structure of the original sentence "
                "without adding extra information.\n\n"
                f"Examples for reference:\n{examples}\n\n"
                f"Text: {text}\n\nAAVE Translation:"
            ),
        },
    ]


class AAVETranslator:
    """OpenAI-backed SAE -> AAVE translator."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 150,
        few_shot: Sequence[str] = DEFAULT_FEW_SHOT,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.few_shot = tuple(few_shot)
        self._client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def translate(self, text: str) -> str:
        if not text:
            return ""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=_build_messages(text, self.few_shot),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def translate_many(
        self,
        texts: Iterable[str],
        *,
        show_progress: bool = True,
    ) -> List[str]:
        items = list(texts)
        iterator = tqdm(items, desc="AAVE translation") if show_progress else items
        return [self.translate(t) for t in iterator]


def translate_to_aave(text: str, few_shot: Sequence[str] = DEFAULT_FEW_SHOT) -> str:
    """One-shot helper that mirrors the original module-level function."""
    return AAVETranslator(few_shot=few_shot).translate(text)
