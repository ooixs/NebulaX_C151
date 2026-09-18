from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pandas as pd

PREDICTORS = {
    "Door": "Door.code.predict",
    "ACV": "ACV.code.predict",
    "Rail Corrugation": "rail_corrugation.predict",
    "SHM": "SHM.code.predict",
}


def predict(subsystem: str, input_path: str | Path) -> pd.DataFrame:
    if subsystem not in PREDICTORS:
        raise ValueError(f"unknown subsystem {subsystem!r}; choose one of {tuple(PREDICTORS)}")
    return import_module(PREDICTORS[subsystem]).predict(input_path)


def predict_with_evidence(subsystem: str, input_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Return official predictions plus non-submission evidence for technician review."""
    if subsystem not in PREDICTORS:
        raise ValueError(f"unknown subsystem {subsystem!r}; choose one of {tuple(PREDICTORS)}")
    module = import_module(PREDICTORS[subsystem])
    if hasattr(module, "predict_with_evidence"):
        return module.predict_with_evidence(input_path)
    return module.predict(input_path), {}
