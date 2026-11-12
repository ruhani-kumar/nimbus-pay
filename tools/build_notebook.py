import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "nb"
OUT = ROOT / "notebook.ipynb"


def lines(text):
    text = text.strip("\n") + "\n"
    parts = text.splitlines(keepends=True)
    parts[-1] = parts[-1].rstrip("\n")
    return parts


def cell(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(text)}


def build():
    files = sorted((p for p in SRC.iterdir() if p.suffix in {".md", ".py"}), key=lambda p: (p.stem, p.suffix != ".md"))
    nb = {
        "cells": [cell(p) for p in files],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name} with {len(files)} cells")


def run_local(stop_after):
    files = sorted((p for p in SRC.iterdir() if p.suffix == ".py"), key=lambda p: p.stem)
    scope = {"__name__": "__main__"}
    for p in files:
        if p.stem > stop_after:
            break
        print(f"\n=== {p.name}")
        exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), scope)


if __name__ == "__main__":
    build()
    if len(sys.argv) > 2 and sys.argv[1] == "--run-until":
        run_local(sys.argv[2])
