"""
Temporal Segmentation Engine for Door Subsystem
Identifies door opening and closing operational cycles from continuous sensor streams.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


def segment_door_stream(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Segments a continuous door sensor stream into discrete operational cycles.
    
    In the onboard controller log stream, active cycles are recorded at 50 Hz (20 ms interval),
    while inactive/idle dwell intervals exhibit sampling breaks (delta t > 100 ms).
    
    Returns:
        List of dicts containing segment boundaries, operation type, and slice indices.
    """
    if len(df) == 0:
        return []

    # Vectorized millisecond computation
    split_cols = df["Datetime"].str.split("-", expand=True).astype(np.int64)
    # Millisecond timeline: S*1000 + M*60000 + H*3600000 + D*86400000
    ms = (
        split_cols[6]
        + split_cols[5] * 1000
        + split_cols[4] * 60000
        + split_cols[3] * 3600000
        + split_cols[2] * 86400000
    )
    dt_ms_diff = ms.diff()

    # Gap threshold: any gap > 100 ms or negative wrap indicates a new operational cycle
    gap_mask = (dt_ms_diff > 100) | (dt_ms_diff < 0)
    split_indices = [0] + list(df.index[gap_mask]) + [len(df)]

    segments = []
    for i in range(len(split_indices) - 1):
        s_idx = split_indices[i]
        e_idx = split_indices[i + 1]
        seg_df = df.iloc[s_idx:e_idx]

        # Ignore tiny spurious noise bursts (< 20 samples = 0.4s)
        if len(seg_df) < 20:
            continue

        # Determine operation: Open vs Close
        is_opening_mode = seg_df["Door is opening"].sum() > seg_df["Door is closing"].sum()
        pos_start = seg_df.iloc[0]["Door leaf position"]
        pos_end = seg_df.iloc[-1]["Door leaf position"]
        op = "Open" if (is_opening_mode or (pos_end > pos_start)) else "Close"

        segments.append({
            "segment_idx": len(segments),
            "start_time": seg_df.iloc[0]["Datetime"],
            "end_time": seg_df.iloc[-1]["Datetime"],
            "start_idx": s_idx,
            "end_idx": e_idx - 1,
            "n_rows": len(seg_df),
            "operation": op,
            "df": seg_df
        })

    return segments
