"""Opt-in acceleration for Hugging Face Hub downloads.

`hf_transfer` gives multi-connection downloads that saturate a fast NIC when
pulling base models and adapters from the Hub. `huggingface_hub` only uses it
when the ``HF_HUB_ENABLE_HF_TRANSFER`` environment variable is truthy *and* the
package is importable -- and it reads that variable once, when it is first
imported.

retrain imports `transformers` / `huggingface_hub` lazily (only when a run loads
a model), so calling `enable_if_available()` at ``import retrain`` sets the flag
in time to take effect, with no import-order gymnastics for the caller.

The check is deliberately guarded: setting the flag *without* the package makes
`huggingface_hub` raise on every download, so we enable it only when
`hf_transfer` is actually installed, and we never override a value the user set
themselves (so ``HF_HUB_ENABLE_HF_TRANSFER=0`` remains a working off switch on a
node that has the package).
"""

from __future__ import annotations

import importlib.util
import os

_ENV = "HF_HUB_ENABLE_HF_TRANSFER"


def enable_if_available() -> bool:
    """Enable hf_transfer for Hub downloads when it is safe to do so.

    Returns ``True`` if this call turned it on, ``False`` otherwise (the user
    already set the variable either way, or the package is not installed).
    """
    if os.environ.get(_ENV) is not None:
        return False  # respect an explicit choice, on or off
    if importlib.util.find_spec("hf_transfer") is None:
        return False  # the flag without the package would crash every download
    os.environ[_ENV] = "1"
    return True
