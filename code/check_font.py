#!/usr/bin/env python3
"""Report the font that the figure generator will use."""
from __future__ import annotations
import argparse
from matplotlib import font_manager


def resolve():
    for name in ("Arial", "Arimo", "Liberation Sans", "Nimbus Sans"):
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            actual = font_manager.FontProperties(fname=path).get_name()
        except (ValueError, OSError):
            continue
        if actual.casefold() == name.casefold():
            return actual, path, actual.casefold() == "arial"
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    return font_manager.FontProperties(fname=path).get_name(), path, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-arial", action="store_true", help="Exit with status 2 if genuine Arial is not installed.")
    args = parser.parse_args()
    family, path, is_arial = resolve()
    print(f"Selected figure font: {family}")
    print(f"Font file: {path}")
    print(f"Genuine Arial available: {'yes' if is_arial else 'no'}")
    if not is_arial:
        print("Install a properly licensed Arial font locally and refresh Matplotlib's font cache to use Arial.")
    if args.require_arial and not is_arial:
        raise SystemExit(2)
