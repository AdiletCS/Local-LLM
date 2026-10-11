"""Automatic CPU and memory defaults for the normal runtime."""

import os
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None


APP_DATA_DIR = Path(
    os.environ.get("LOCALAPPDATA")
    or (Path.home() / "AppData" / "Local" if os.name == "nt" else Path.home() / ".local" / "share")
) / "LocalLLM"
CONTEXT_OPTIONS = (2048, 4096, 8192, 16384)
GIB = 1024 ** 3


def _cpu_counts():
    if psutil is None:
        return None, None
    try:
        return psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)
    except Exception:
        return None, None


def effective_threads(physical=None, logical=None):
    """Prefer physical cores; fall back safely when hardware details are missing."""
    if physical is None and logical is None:
        physical, logical = _cpu_counts()
    maximum = logical or physical or 256
    return max(1, min(physical or logical or 1, maximum))


def _total_memory_bytes(total_ram=None):
    if total_ram is not None:
        return total_ram
    if psutil is None:
        return None
    try:
        return psutil.virtual_memory().total
    except Exception:
        return None


def automatic_batch_sizes(total_ram=None):
    """Keep llama.cpp CPU batches modest on low-memory Windows computers."""
    total_ram = _total_memory_bytes(total_ram)
    if total_ram is None or total_ram < 8 * GIB:
        return 128, 64
    if total_ram < 16 * GIB:
        return 256, 128
    return 512, 256


def automatic_response_tokens(total_ram=None, context_size=None):
    """Choose a useful response cap while leaving room for the prompt in context."""
    if context_size is not None:
        try:
            context_size = max(512, int(context_size))
            if context_size < 2048:
                return max(128, context_size // 4)
            return min(1024, max(512, context_size // 4))
        except (TypeError, ValueError):
            pass
    total_ram = _total_memory_bytes(total_ram)
    if total_ram is None or total_ram < 8 * GIB:
        return 512
    if total_ram < 16 * GIB:
        return 512
    return 768
