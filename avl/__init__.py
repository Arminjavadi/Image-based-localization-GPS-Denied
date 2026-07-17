"""Absolute Visual Localization (AVL) for GPS-denied UAV navigation."""

__all__ = ["AVLLocalizer"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "AVLLocalizer":
        from avl.localizer import AVLLocalizer

        return AVLLocalizer
    raise AttributeError(f"module 'avl' has no attribute {name!r}")
