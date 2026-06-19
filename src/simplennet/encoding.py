"""Input encoders for turning application data into numeric model rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


@dataclass
class UniversalEncoder:
    """Deterministically maps mixed Python values into fixed-width numeric rows."""

    width: int = 64

    def transform_many(self, values: Iterable[Any]) -> list[list[float]]:
        return [self.transform_one(value) for value in values]

    def transform_one(self, value: Any) -> list[float]:
        row = [0.0 for _ in range(self.width)]
        self._emit(value, row, "$")
        return row

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"kind": "universal", "width": self.width}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "UniversalEncoder":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(width=int(data["width"]))

    def _emit(self, value: Any, row: list[float], path: str) -> None:
        if value is None:
            self._add_token(row, f"{path}:none")
            return
        if isinstance(value, bool):
            self._add_number(row, f"{path}:bool", 1.0 if value else 0.0)
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self._add_number(row, f"{path}:number", float(value))
            return
        if isinstance(value, (datetime, date)):
            self._add_number(row, f"{path}:timestamp", float(value.toordinal()))
            self._add_token(row, f"{path}:date:{value.isoformat()}")
            return
        if isinstance(value, dict):
            if not value:
                self._add_token(row, f"{path}:empty-dict")
                return
            for key in sorted(value, key=lambda item: str(item)):
                self._emit(value[key], row, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            items = list(value)
            if not items:
                self._add_token(row, f"{path}:empty-list")
                return
            for index, item in enumerate(items):
                self._emit(item, row, f"{path}[{index}]")
            self._add_number(row, f"{path}:length", float(len(items)))
            return

        self._add_token(row, f"{path}:text:{str(value).casefold()}")

    def _add_number(self, row: list[float], token: str, value: float) -> None:
        index, sign = self._slot(token)
        row[index] += sign * value

    def _add_token(self, row: list[float], token: str) -> None:
        index, sign = self._slot(token)
        row[index] += sign

    def _slot(self, token: str) -> tuple[int, float]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "little") % self.width
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        return index, sign
