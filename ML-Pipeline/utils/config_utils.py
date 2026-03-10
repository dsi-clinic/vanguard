from pathlib import Path
from typing import Any
import yaml

def load_pipeline_config(config_path: str = "ML-Pipeline/config_pcr.yaml") -> dict[str, Any]:
    """Load configuration YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)