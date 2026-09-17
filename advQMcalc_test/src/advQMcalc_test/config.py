from dataclasses import dataclass
from pathlib import Path

@dataclass
class WorkflowConfig:
    """
    Holds user-provided configuration for a workflow run.
    """

    cif_path: Path
    calc_type: str = "scf"
