import yaml
import shutil
from pathlib import Path
from datetime import datetime

def load_pipeline_config(config_path: str = "ML-Pipeline/config_pcr.yaml"):
    """
    Standardized loader for Vanguard project configs to ensure everyone uses the same paths and toggles.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = config['experiment_setup']['name']
    outdir = Path(config['experiment_setup']['base_outdir']) / f"{exp_name}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(config_path, outdir / "config_used.yaml")
    
    return config, outdir