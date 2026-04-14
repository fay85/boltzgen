"""
Convert boltzgen from CUDA to MUSA.
Applies the same mapping rules as torch_musa's musa_converter.py
but without requiring libcst. Safe for boltzgen since all 'cuda'/'CUDA'
occurrences are actual CUDA API references.
"""

import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")

REPLACEMENTS = [
    ("CUDA", "MUSA"),
    ("cuda", "musa"),
    ("NCCL", "MCCL"),
    ("nccl", "mccl"),
    ("nvcc", "mcc"),
]

LAUNCH_SCRIPT = os.path.join(ROOT, "src", "boltzgen", "resources", "main.py")


def convert_py_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content
    for old, new in REPLACEMENTS:
        content = content.replace(old, new)

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


def add_import_torch_musa(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if "import torch_musa" in content:
        return False

    lines = content.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i
            break

    lines.insert(insert_idx, "import torch_musa")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return True


def main():
    converted = []
    for dirpath, _, filenames in os.walk(SRC_DIR):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(dirpath, fn)
            if convert_py_file(fp):
                converted.append(os.path.relpath(fp, ROOT))

    print(f"Converted {len(converted)} Python files:")
    for f in sorted(converted):
        print(f"  {f}")

    if add_import_torch_musa(LAUNCH_SCRIPT):
        print(f"\nAdded 'import torch_musa' to {os.path.relpath(LAUNCH_SCRIPT, ROOT)}")
    else:
        print(f"\n'import torch_musa' already present in {os.path.relpath(LAUNCH_SCRIPT, ROOT)}")


if __name__ == "__main__":
    main()
