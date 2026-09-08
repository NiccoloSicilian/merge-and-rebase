"""Register this repo's own open_clip model configs.

The mini CLIP pair (ViT-Mini / ViT-Mini-Plus) is not an open_clip built-in.
``pretrain_mini_clip.py`` registers it by writing JSON into the installed
open_clip's ``model_configs`` directory, which does not survive a rebuilt venv
or an upgraded open_clip -- and leaves every other entry point raising
``RuntimeError: Model config for 'ViT-Mini-Plus' not found in built-ins``.

Shipping the configs here and pointing open_clip at them keeps the registration
with the code that needs it: no writes into site-packages, and nothing to redo
after reinstalling.
"""

from __future__ import annotations

from pathlib import Path

_CONFIG_DIR = Path(__file__).parent / "openclip_model_configs"
_registered = False


def register_local_openclip_configs() -> None:
    """Point open_clip at ``openclip_model_configs/``. Safe to call repeatedly."""
    global _registered
    if _registered or not _CONFIG_DIR.is_dir():
        return

    import open_clip

    known = set(open_clip.list_models())
    if not {p.stem for p in _CONFIG_DIR.glob("*.json")} - known:
        _registered = True
        return

    open_clip.add_model_config(_CONFIG_DIR)
    _registered = True
