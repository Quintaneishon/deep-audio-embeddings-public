"""Small helpers for durable, code-only training checkpoints."""

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(value: Any, destination: str | Path) -> None:
    """Write a checkpoint atomically so interruption cannot replace a good file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
