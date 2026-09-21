"""Common utility functions used by the full pipeline."""
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any, Dict, Iterable, List
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

def project_path(*parts: str) -> Path:
    """Return an absolute project-root-relative path."""
    return ROOT.joinpath(*parts)

def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def set_seed(seed: int) -> None:
    """Set Python and NumPy seeds. Torch is seeded in train_drl.py."""
    random.seed(seed); np.random.seed(seed)

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)

def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path); ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f: json.dump(obj, f, indent=2)

def load_json(path: str | Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def write_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path); ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row) + '\n')

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f: return [json.loads(line) for line in f if line.strip()]
