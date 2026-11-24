"""
Lets the tests load code from the numbered project folders.

Folder names like "3_mini_gpt" cannot be imported the normal way (they start
with a digit), so load("3_mini_gpt", "gpt") loads the file by its path.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load(folder, name):
    key = f"{folder}.{name}"
    if key in sys.modules:
        return sys.modules[key]
    sys.path.insert(0, str(ROOT / folder))  # so the file can import its neighbours
    try:
        spec = importlib.util.spec_from_file_location(key, ROOT / folder / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module
