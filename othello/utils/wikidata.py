"""Wikidata search, ASK, and result-parsing helpers."""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from time import sleep
from typing import Any, Dict, Iterable, Optional

import requests

from othello.utils.text import normalize_label

log = logging.getLogger(__name__)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

DEFAULT_USER_AGENT = "OTHELLO/0.2 (https://github.com/; cekruger99@gmail.com)"


@dataclass
class WikidataClient:
    """Lightweight Wikidata client with per-process caches and polite rate limiting."""

    user_agent: str = DEFAULT_USER_AGENT
    search_pause: float = 0.1
    sparql_pause: float = 0.2
    timeout: int = 30

    def __post_init__(self) -> None:
        self._session = requests.Session()
        self._entity_cache: Dict[str, Optional[str]] = {}
        self._property_cache: Dict[str, Optional[str]] = {}

    @property
    def _headers(self) -> Dict[str, str]:
        return {"User-Agent": self.user_agent}

    def _search(self, name: str, kind: str) -> Optional[str]:
        params = {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "type": kind,
            "format": "json",
            "limit": 5,
        }
        sleep(self.search_pause)
        try:
            resp = self._session.get(
                WIKIDATA_API, params=params, headers=self._headers, timeout=self.timeout
            )
            resp.raise_for_status()
            hits = resp.json().get("search", [])
            return hits[0]["id"] if hits else None
        except (requests.RequestException, ValueError, KeyError) as exc:
            log.debug("Wikidata %s search failed for %r: %s", kind, name, exc)
            return None

    def search_entity(self, name: str) -> Optional[str]:
        """Return the top QID matching ``name`` or ``None``."""
        name = normalize_label(name)
        if not name:
            return None
        if name not in self._entity_cache:
            self._entity_cache[name] = self._search(name, "item")
        return self._entity_cache[name]

    def search_property(self, name: str) -> Optional[str]:
        """Return the top PID matching ``name`` or ``None``."""
        name = normalize_label(name)
        if not name:
            return None
        if name not in self._property_cache:
            self._property_cache[name] = self._search(name, "property")
        return self._property_cache[name]

    def ask(self, query: str) -> Optional[bool]:
        """Run a SPARQL ASK query; ``None`` is returned on failure."""
        sleep(self.sparql_pause)
        try:
            resp = self._session.get(
                WIKIDATA_SPARQL,
                params={"query": query, "format": "json"},
                headers=self._headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return bool(resp.json().get("boolean", False))
        except (requests.RequestException, ValueError) as exc:
            log.debug("SPARQL ASK failed: %s", exc)
            return None


def _coerce_json(payload: Any) -> Optional[dict]:
    """Accept dicts, JSON strings, or python-literal strings."""
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        return None
    text = payload.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, SyntaxError, TypeError):
        return None


def parse_wikidata_result(payload: Any) -> Optional[str]:
    """Extract a single human-readable value from a Wikidata SPARQL response."""
    data = _coerce_json(payload)
    if data is None:
        return None

    if "boolean" in data:
        return str(data["boolean"])

    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        return None

    for binding in bindings:
        if not isinstance(binding, dict):
            continue

        label_keys = [k for k in binding if k.endswith("Label")]
        for key in label_keys:
            value = binding.get(key, {}).get("value")
            if value:
                return value

        for key in binding:
            if any(token in key.lower() for token in ("date", "time", "inception")):
                value = binding.get(key, {}).get("value")
                if value:
                    return str(value).split("T", 1)[0]

        for key in binding:
            if key in label_keys:
                continue
            value = binding.get(key, {}).get("value")
            if value:
                return value

    return None


def build_ask_query(entity_ids: Iterable[str], property_ids: Iterable[str]) -> Optional[str]:
    """Build a directional SPARQL ASK query over every entity pair x property.

    Returns ``None`` when there is not enough material to verify a claim
    (fewer than two unique entities, or zero properties).
    """
    entities = [e for e in dict.fromkeys(entity_ids) if e]
    properties = [p for p in dict.fromkeys(property_ids) if p]
    if len(entities) < 2 or not properties:
        return None

    patterns = []
    for subject in entities:
        for obj in entities:
            if subject == obj:
                continue
            for prop in properties:
                patterns.append(f"{{ wd:{subject} wdt:{prop} wd:{obj} . }}")

    if not patterns:
        return None
    return "ASK WHERE { " + " UNION ".join(patterns) + " }"
