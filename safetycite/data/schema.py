"""QA example schema + jsonl IO."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class QAExample:
    id: str
    domain: str
    question: str
    gold_citations: list[str]  # canonical section ids, e.g. ["1926.501"]
    reference_answer: str
    source_section: str = ""  # primary section the example was derived from
    kind: str = "scenario"  # "scenario" | "heading"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> QAExample:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in d.items() if k in known})


def write_jsonl(path: str | Path, rows: list[QAExample]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return len(rows)


def read_jsonl(path: str | Path) -> list[QAExample]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[QAExample] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(QAExample.from_dict(json.loads(line)))
    return out
