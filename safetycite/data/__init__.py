"""Dataset construction: corpus → per-domain Q&A (train/val/test) + chat rendering."""

from safetycite.data.render import SYSTEM_PROMPT, render_messages, render_target
from safetycite.data.schema import QAExample, read_jsonl, write_jsonl

__all__ = [
    "QAExample",
    "SYSTEM_PROMPT",
    "read_jsonl",
    "render_messages",
    "render_target",
    "write_jsonl",
]
