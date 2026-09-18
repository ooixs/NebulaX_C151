"""Helpers for descriptive, non-diagnostic comparisons in the technician UI."""
from __future__ import annotations

import math
import statistics


def comparison(label, value, population, unit="", digits=3):
    """Compare one value with peers without treating the result as a fault threshold."""
    values = [float(item) for item in population if item is not None and math.isfinite(float(item))]
    value = float(value)
    average = statistics.fmean(values) if values else value
    deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
    z_score = (value - average) / deviation if deviation > 1e-12 else 0.0
    if len(values) < 2:
        status = "only"
    elif z_score >= 2:
        status = "outlier_high"
    elif z_score <= -2:
        status = "outlier_low"
    elif z_score >= 1:
        status = "above"
    elif z_score <= -1:
        status = "below"
    else:
        status = "typical"
    return {
        "label": label,
        "value": round(value, digits),
        "average": round(average, digits),
        "unit": unit,
        "status": status,
        "z_score": round(z_score, 2),
        "peer_count": len(values),
    }
