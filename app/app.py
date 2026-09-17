"""Offline, non-technical NebulaX condition-monitoring application."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent if (APP_DIR.parent / "Rail Corrugation").exists() else APP_DIR


@st.cache_resource
def rail_module():
    script = ROOT / "Rail Corrugation/code/predict.py"
    spec = importlib.util.spec_from_file_location("nebulax_rail_predict", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


st.set_page_config(page_title="NebulaX Condition Monitor", page_icon="🚆", layout="wide")
st.title("NebulaX Condition Monitor")
st.subheader("Rail corrugation")
st.write("Upload one or more one-second sensor CSV files. Processing stays on this machine.")
st.caption("Side I uses bearing positions 1, 3, 5 and 7; Side II uses positions 2, 4, 6 and 8.")

uploads = st.file_uploader("Rail recordings", type=["csv"], accept_multiple_files=True)
if uploads:
    rows, errors = [], []
    with st.spinner("Checking recordings and calculating rail evidence…"):
        with tempfile.TemporaryDirectory(prefix="nebulax-rail-") as directory:
            work = Path(directory)
            predictor = rail_module()
            for upload in uploads:
                path = work / Path(upload.name).name
                path.write_bytes(upload.getvalue())
                try:
                    rows.append(predictor.predict_one(path, with_proba=True))
                except Exception as exc:  # one corrupt item must not abort the batch
                    errors.append({"file_id": upload.name, "error": str(exc)})
    if rows:
        details = pd.DataFrame(rows)
        for _, row in details.iterrows():
            st.metric(row.file_id, row.prediction, f"{row.speed_mps:.2f} m/s")
            st.caption(f"Side I evidence {row['p_Side I']:.3f} · Side II evidence {row['p_Side II']:.3f}")
        submission = details[["file_id", "prediction"]].to_csv(index=False).encode()
        st.download_button("Download rail_predictions.csv", submission, "rail_predictions.csv", "text/csv")
        st.caption("Evidence scores rank model support; they are not calibrated probabilities.")
    if errors:
        st.warning("Some files could not be processed. Other valid files were still completed.")
        st.dataframe(pd.DataFrame(errors), hide_index=True)
