"""Category taxonomy loader.

Loads the categories from a YAML file and exposes as a simple list.

Functions:
    load_categories(path: str = "config/categories.yaml") -> list[str]
"""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import List


def load_categories(path: str = "config/categories.yaml") -> List[str]:
    """Load category list from YAML file.

    Args:
        path: Path to YAML taxonomy file.

    Returns:
        List of category strings.
    """
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    cats = data.get("categories", [])
    if not isinstance(cats, list) or not cats:
        raise ValueError("No categories found in taxonomy file.")
    return [str(c) for c in cats]
