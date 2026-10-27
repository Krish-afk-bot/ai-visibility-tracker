import os
from pathlib import Path
from typing import Dict, Any

THRESHOLDS_PATH = Path(__file__).parent / "thresholds.yaml"

def load_thresholds() -> Dict[str, float]:
    """
    Load threshold configuration from thresholds.yaml.
    Falls back to safe built-in defaults or custom parser if PyYAML is not installed.
    """
    defaults = {
        "mention_f1": 0.95,
        "position_within1": 0.80,
        "competitor_fallback_f1": 0.70,
        "sentiment_accuracy": 0.70,
    }
    
    if not THRESHOLDS_PATH.exists():
        return defaults

    try:
        import yaml
        with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    except ImportError:
        # Simple line-by-line fallback parser for basic key: value YAML
        res = {}
        with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip().split("#")[0].strip()
                    try:
                        res[k] = float(v)
                    except ValueError:
                        pass
        if res:
            return res

    return defaults
