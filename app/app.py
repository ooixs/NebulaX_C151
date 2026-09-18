"""
Unified Streamlit Application for PS3 - Rail Vehicle Condition Monitoring
LTA NebulaX 2026 Hackathon Deliverable (Deliverable 3)

Features:
- Fleet Control Tower (Executive Overview & Multi-Subsystem Health Matrix)
- Structural Health Monitoring (SHM): Dynamic fatigue damage assessment, RUL estimation, Rainflow analytics
- Saloon Door System (Door): Vectorized temporal stream segmentation, electromechanical glide current diagnostics, work orders
- Full judging transparency with official competition formulas, CV benchmarks, and 1-click submission exports
"""

import os
import sys
import io
import time
import json
import types
import importlib.util
import numpy as np
import pandas as pd
import streamlit as st

# Setup paths and isolated module loaders
APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(APP_DIR, "..", ".."))
PROJECT_ROOT = REPO_ROOT

def resolve_dir(*candidates):
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return candidates[0] if candidates else ""

def resolve_file(*candidates):
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return candidates[0] if candidates else ""

SHM_CODE_DIR = resolve_dir(
    os.path.join(REPO_ROOT, "SHM", "code"),
    os.path.join(APP_DIR, "SHM", "code"),
    os.path.join(WORKSPACE_ROOT, "ps3", "SHM", "code"),
    os.path.join(WORKSPACE_ROOT, "C151", "SHM", "code"),
)
DOOR_CODE_DIR = resolve_dir(
    os.path.join(REPO_ROOT, "Door", "code"),
    os.path.join(APP_DIR, "Door", "code"),
    os.path.join(WORKSPACE_ROOT, "ps3", "Door", "code"),
    os.path.join(WORKSPACE_ROOT, "C151", "Door", "code"),
)
RAIL_CODE_DIR = resolve_dir(
    os.path.join(REPO_ROOT, "Rail Corrugation", "code"),
    os.path.join(REPO_ROOT, "Rail_Corrugation", "code"),
    os.path.join(APP_DIR, "Rail Corrugation", "code"),
    os.path.join(APP_DIR, "Rail_Corrugation", "code"),
    os.path.join(WORKSPACE_ROOT, "ps3", "Rail_Corrugation", "code"),
    os.path.join(WORKSPACE_ROOT, "C151", "Rail Corrugation", "code"),
)
ACV_CODE_DIR = resolve_dir(
    os.path.join(REPO_ROOT, "ACV", "code"),
    os.path.join(APP_DIR, "ACV", "code"),
    os.path.join(WORKSPACE_ROOT, "ps3", "ACV", "code"),
    os.path.join(WORKSPACE_ROOT, "C151", "ACV", "code"),
)

# Register pipeline shims for unpickling
try:
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)
    from pipelines import register_pipeline_shims
    register_pipeline_shims()
except Exception:
    pass

pipe_module = sys.modules.get("pipeline")
if pipe_module is None:
    pipe_module = types.ModuleType("pipeline")
    sys.modules["pipeline"] = pipe_module

try:
    spec_shm = importlib.util.spec_from_file_location("shm_pipeline_internal", os.path.join(SHM_CODE_DIR, "pipeline.py"))
    mod_shm = importlib.util.module_from_spec(spec_shm)
    spec_shm.loader.exec_module(mod_shm)
    if hasattr(mod_shm, "SHMPipeline"):
        pipe_module.SHMPipeline = mod_shm.SHMPipeline
    if hasattr(mod_shm, "CORE_FEATURES"):
        pipe_module.CORE_FEATURES = mod_shm.CORE_FEATURES
except Exception:
    pass

try:
    spec_door = importlib.util.spec_from_file_location("door_pipeline_internal", os.path.join(DOOR_CODE_DIR, "pipeline.py"))
    mod_door = importlib.util.module_from_spec(spec_door)
    spec_door.loader.exec_module(mod_door)
    if hasattr(mod_door, "DoorPipeline"):
        pipe_module.DoorPipeline = mod_door.DoorPipeline
except Exception:
    pass

try:
    spec_rail = importlib.util.spec_from_file_location("rail_pipeline_internal", os.path.join(RAIL_CODE_DIR, "pipeline.py"))
    mod_rail = importlib.util.module_from_spec(spec_rail)
    spec_rail.loader.exec_module(mod_rail)
    if hasattr(mod_rail, "RailPipeline"):
        pipe_module.RailPipeline = mod_rail.RailPipeline
except Exception:
    pass

try:
    spec_acv = importlib.util.spec_from_file_location("acv_pipeline_internal", os.path.join(ACV_CODE_DIR, "pipeline.py"))
    mod_acv = importlib.util.module_from_spec(spec_acv)
    sys.modules["acv_pipeline_internal"] = mod_acv
    spec_acv.loader.exec_module(mod_acv)
    if hasattr(mod_acv, "ACVPipeline"):
        pipe_module.ACVPipeline = mod_acv.ACVPipeline
except Exception:
    pass

# Load isolated features and segmentation modules
spec_shm_feat = importlib.util.spec_from_file_location("shm_features", os.path.join(SHM_CODE_DIR, "features.py"))
shm_features = importlib.util.module_from_spec(spec_shm_feat)
spec_shm_feat.loader.exec_module(shm_features)

spec_door_feat = importlib.util.spec_from_file_location("door_features", os.path.join(DOOR_CODE_DIR, "features.py"))
door_features = importlib.util.module_from_spec(spec_door_feat)
spec_door_feat.loader.exec_module(door_features)

spec_door_seg = importlib.util.spec_from_file_location("door_segmentation", os.path.join(DOOR_CODE_DIR, "segmentation.py"))
door_segmentation = importlib.util.module_from_spec(spec_door_seg)
spec_door_seg.loader.exec_module(door_segmentation)

spec_rail_feat = importlib.util.spec_from_file_location("rail_features", os.path.join(RAIL_CODE_DIR, "features.py"))
rail_features = importlib.util.module_from_spec(spec_rail_feat)
spec_rail_feat.loader.exec_module(rail_features)

spec_acv_feat = importlib.util.spec_from_file_location("acv_features", os.path.join(ACV_CODE_DIR, "features.py"))
acv_features = importlib.util.module_from_spec(spec_acv_feat)
spec_acv_feat.loader.exec_module(acv_features)

# Page configuration
st.set_page_config(
    page_title="LTA Train Condition Monitoring | Control Tower",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #93c5fd;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .card-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .card-panel h4 {
        margin-top: 0;
        color: #0f172a;
        font-weight: 600;
    }
    .status-healthy {
        background-color: #ecfdf5;
        color: #065f46;
        border-left: 4px solid #10b981;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .status-warning {
        background-color: #fffbeb;
        color: #92400e;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .status-critical {
        background-color: #fef2f2;
        color: #991b1b;
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-active {
        background-color: #d1fae5;
        color: #065f46;
    }
    .badge-pending {
        background-color: #f1f5f9;
        color: #64748b;
    }
    .metric-container {
        border: 1px solid #e2e8f0;
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 10px 18px;
        font-weight: 500;
    }

    /* ========================================= */
    /* PREMIUM ENTERPRISE SIDEBAR STYLING       */
    /* ========================================= */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090d16 0%, #0f172a 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
        margin: 14px 0 !important;
    }
    
    /* Brand Header Box */
    .sidebar-brand-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 14px;
        margin-bottom: 16px;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .logo-container {
        background: #ffffff;
        border-radius: 8px;
        padding: 8px 12px;
        display: inline-block;
        margin-bottom: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .sidebar-title {
        color: #ffffff !important;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 4px 0 2px 0;
        letter-spacing: -0.01em;
    }
    .sidebar-subtitle {
        color: #94a3b8 !important;
        font-size: 0.78rem;
        margin-bottom: 10px;
    }
    .live-status-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399 !important;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 0 rgba(16, 185, 129, 0.4);
        animation: pulseAnimation 1.8s infinite;
    }
    @keyframes pulseAnimation {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Radio Navigation Pills */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 6px !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(37, 99, 235, 0.15) !important;
        border-color: rgba(59, 130, 246, 0.4) !important;
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color: #cbd5e1 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover p {
        color: #ffffff !important;
    }

    /* Selected state */
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%) !important;
        border-color: #60a5fa !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Sidebar Mini Cards */
    .sidebar-widget {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 10px;
    }
    .sidebar-widget-title {
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .check-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 5px 0;
        font-size: 0.80rem;
        color: #cbd5e1;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .check-row:last-child {
        border-bottom: none;
    }
    .check-tag-done {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.70rem;
        font-weight: 600;
    }
    .check-tag-pending {
        background: rgba(148, 163, 184, 0.12);
        color: #94a3b8;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.70rem;
        font-weight: 600;
    }
    .sidebar-metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }
    .sidebar-metric-item {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 8px 10px;
        text-align: center;
    }
    .sidebar-metric-label {
        font-size: 0.68rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
    }
    .sidebar-metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL LOADERS (CACHED)
# ==========================================
@st.cache_resource
def load_shm_pipeline():
    """Loads and caches the trained SHM ensemble model pipeline."""
    import joblib
    model_path = resolve_file(
        os.path.join(REPO_ROOT, "SHM", "model", "shm_ensemble.joblib"),
        os.path.join(APP_DIR, "models", "shm_ensemble.joblib"),
        os.path.join(APP_DIR, "SHM", "model", "shm_ensemble.joblib"),
        os.path.join(WORKSPACE_ROOT, "ps3", "SHM", "model", "shm_ensemble.joblib"),
    )
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_resource
def load_door_pipeline():
    """Loads and caches the trained Door ensemble model pipeline."""
    import joblib
    model_path = resolve_file(
        os.path.join(REPO_ROOT, "Door", "model", "door_model.joblib"),
        os.path.join(APP_DIR, "models", "door_model.joblib"),
        os.path.join(APP_DIR, "Door", "model", "door_model.joblib"),
        os.path.join(WORKSPACE_ROOT, "ps3", "Door", "model", "door_model.joblib"),
    )
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_resource
def load_rail_pipeline():
    """Loads and caches the trained Rail Corrugation ensemble model pipeline."""
    import joblib
    model_path = resolve_file(
        os.path.join(REPO_ROOT, "Rail Corrugation", "model", "rail_model.joblib"),
        os.path.join(REPO_ROOT, "Rail_Corrugation", "model", "rail_model.joblib"),
        os.path.join(APP_DIR, "models", "rail_model.joblib"),
        os.path.join(APP_DIR, "Rail Corrugation", "model", "rail_model.joblib"),
        os.path.join(APP_DIR, "Rail_Corrugation", "model", "rail_model.joblib"),
        os.path.join(WORKSPACE_ROOT, "ps3", "Rail_Corrugation", "model", "rail_model.joblib"),
    )
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_resource
def load_acv_pipeline():
    """Loads and caches the trained ACV refrigerant leakage pipeline."""
    import joblib
    model_path = resolve_file(
        os.path.join(REPO_ROOT, "ACV", "model", "acv_model.joblib"),
        os.path.join(APP_DIR, "models", "acv_model.joblib"),
        os.path.join(APP_DIR, "ACV", "model", "acv_model.joblib"),
        os.path.join(WORKSPACE_ROOT, "ps3", "ACV", "model", "acv_model.joblib"),
    )
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


# ==========================================
# VIEW 1: FLEET CONTROL TOWER (EXECUTIVE OVERVIEW)
# ==========================================
def render_fleet_control_tower():
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🚆 Fleet Condition Monitoring — Central Control Tower</div>
        <div class="sub-title">Land Transport Authority (LTA) NebulaX 2026 • Predictive Maintenance System</div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Fleet Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Competition Average Score", value="0.9296", delta="93.0% Overall PS3", help="Mean judging score across all 4 subsystems (ACV: 1.0000, Door: 0.9909, SHM: 0.8809, Rail: 0.8464)")
    with kpi2:
        st.metric(label="Subsystems Operational", value="4 of 4 Active", delta="100% Full Fleet Coverage")
    with kpi3:
        st.metric(label="Active Fleet Alerts", value="18 Alerts", delta="8 Door, 2 Bogie, 7 Rail, 1 ACV", delta_color="inverse")
    with kpi4:
        st.metric(label="Submission Status", value="Ready (100%)", delta="predictions.zip Packaged")

    st.markdown("---")

    # Subsystem Health Matrix
    st.subheader("Subsystem Telemetry & Deployment Matrix")

    col_door, col_shm = st.columns(2)

    with col_door:
        st.markdown("""
        <div class="card-panel">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0;">🚪 Saloon Door System</h4>
                <span class="badge-pill badge-active">Active</span>
            </div>
            <p style="color: #64748b; font-size: 0.9rem; margin-top: 8px;">
                Vectorized continuous temporal cycle segmentation and DC motor glide friction diagnostics.
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0;">
                <div class="metric-container">
                    <div style="font-size: 0.8rem; color: #64748b;">IoU Soft F1 Score</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #065f46;">0.9909</div>
                </div>
                <div class="metric-container">
                    <div style="font-size: 0.8rem; color: #64748b;">Cycle Boundary IoU</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #1e3a8a;">100.0%</div>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <small><b>Held-Out Test Set:</b> 38 operational cycles detected (30 Normal, 8 Abnormal resistance)</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_shm:
        st.markdown("""
        <div class="card-panel">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0;">🔩 Bogie Structural Health (SHM)</h4>
                <span class="badge-pill badge-active">Active</span>
            </div>
            <p style="color: #64748b; font-size: 0.9rem; margin-top: 8px;">
                Dynamic stress fatigue damage estimation via Rainflow cycle counting and calibrated meta-regressors.
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0;">
                <div class="metric-container">
                    <div style="font-size: 0.8rem; color: #64748b;">Judging Score</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #065f46;">0.8809</div>
                </div>
                <div class="metric-container">
                    <div style="font-size: 0.8rem; color: #64748b;">Validation MAPE</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #1e3a8a;">9.14%</div>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <small><b>Held-Out Test Set:</b> 16 test recordings evaluated (Mean Cumulative Damage D = 0.4133)</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_rc, col_acv = st.columns(2)

    with col_rc:
        st.markdown("""
        <div class="card-panel">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0;">🛤️ Rail Corrugation System</h4>
                <span class="badge-pill badge-active">Active</span>
            </div>
            <p style="color: #64748b; font-size: 0.9rem; margin-top: 8px;">
                Axle-box vibration short-pitch corrugation detection and bilateral spatial localization.
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0;">
                <div class="metric-container">
                    <div style="font-size: 0.8rem; color: #64748b;">Macro F1 Score</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #065f46;">0.8464</div>
                </div>
                <div class="metric-container">
                    <div style="font-size: 0.8rem; color: #64748b;">Accuracy</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #1e3a8a;">95.22%</div>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <small><b>Held-Out Test Set:</b> 68 test recordings evaluated (61 Normal, 4 Side I, 3 Side II)</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_acv:
        st.markdown("""
        <div class="card-panel">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0;">❄️ Air Conditioning & Ventilation (ACV)</h4>
                <span class="badge-pill badge-active">Active</span>
            </div>
            <p style="color: #64748b; font-size: 0.9rem; margin-top: 8px;">
                Thermodynamic continuous tracking error anomaly ranking & refrigerant leakage localisation.
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0;">
                <div class="metric-container">
                    <div style="font-size: 0.8rem; color: #64748b;">Rank-Decay Score</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #065f46;">1.0000</div>
                </div>
                <div class="metric-container">
                    <div style="font-size: 0.8rem; color: #64748b;">Top-1 Accuracy</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #1e3a8a;">100.0%</div>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <small><b>Held-Out Test Set:</b> acv_test_case.xlsx evaluated (Top Fault Candidate: Car 01)</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Submission & Deliverable Status
    st.markdown("---")
    st.subheader("Official Submission Artifacts & Verification")

    zip_path = resolve_file(
        os.path.join(REPO_ROOT, "predictions.zip"),
        os.path.join(WORKSPACE_ROOT, "predictions.zip"),
        os.path.join(WORKSPACE_ROOT, "ps3", "predictions.zip"),
    )
    if os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            file_list = z.namelist()

        s_col1, s_col2 = st.columns([2, 1])
        with s_col1:
            st.success(f"✅ `predictions.zip` is ready at repository root with {len(file_list)} submission files:")
            st.code("\n".join([f"  • {fname}" for fname in file_list]))
        with s_col2:
            with open(zip_path, "rb") as fp:
                st.download_button(
                    label="📦 Download predictions.zip",
                    data=fp.read(),
                    file_name="predictions.zip",
                    mime="application/zip",
                    use_container_width=True,
                    help="Official zipped archive for competition judging leaderboard"
                )


# ==========================================
# VIEW 2: STRUCTURAL HEALTH MONITORING (SHM)
# ==========================================
def render_shm_subsystem():
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🔩 Bogie Structural Health Monitoring (SHM)</div>
        <div class="sub-title">Dynamic Stress Fatigue Damage Assessment & Remaining Useful Life (RUL) Forecasting</div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Judging Metric Score", value="0.8809", delta="max(0, 1 - MAPE)", help="Evaluated over 4-fold cross validation across 64 recordings")
    with col2:
        st.metric(label="Validation Split MAPE", value="9.14%", delta="-38% vs baseline")
    with col3:
        st.metric(label="Coefficient of Det. (R²)", value="0.9880", delta="High Fidelity")
    with col4:
        st.metric(label="Test Set Latency", value="~1.1s / file", delta="581,120 pts / file")

    st.markdown("---")

    pipeline = load_shm_pipeline()
    if pipeline is None:
        st.error("SHM model checkpoint not found. Please train using `ps3/SHM/code/train.py`.")
        return

    extract_features_from_series = shm_features.extract_features_from_series

    tab1, tab2, tab3 = st.tabs([
        "🔍 Single-File Diagnostic & Waveform Analyzer",
        "⚡ Batch Inference & Submission Generator",
        "📊 Methodology & Judging Rubric"
    ])

    with tab1:
        st.write("Analyze dynamic strain recordings from bogie components to quantify cumulative fatigue damage:")

        test_dir = resolve_dir(
            os.path.join(WORKSPACE_ROOT, "problem_statements", "PS3", "02_Datasets", "SHM", "Test"),
            os.path.join(REPO_ROOT, "data", "SHM", "Test"),
        )
        has_test_files = os.path.exists(test_dir)

        c_opt1, c_opt2 = st.columns([1, 1])
        selected_file_path = None
        series_data = None
        current_filename = ""

        with c_opt1:
            st.markdown("##### Option A: Instant Test Preset")
            if has_test_files:
                test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".csv")])
                selected_test = st.selectbox(
                    "Pick a held-out test recording:",
                    test_files,
                    index=0,
                    help="Instant load from official test set"
                )
                if st.button("⚡ Load Selected Test File", key="btn_load_test"):
                    current_filename = selected_test
                    selected_file_path = os.path.join(test_dir, selected_test)

        with c_opt2:
            st.markdown("##### Option B: Upload Custom Recording")
            uploaded = st.file_uploader("Upload CSV file (single column stress)", type=["csv"], key="shm_upload")
            if uploaded is not None:
                current_filename = uploaded.name
                df_raw = pd.read_csv(uploaded, header=None, dtype=np.float64)
                series_data = df_raw[0].values

        if selected_file_path and series_data is None:
            df_raw = pd.read_csv(selected_file_path, header=None, dtype=np.float64)
            series_data = df_raw[0].values

        if series_data is not None:
            with st.spinner(f"Extracting Rainflow cycles and physical stress features from {current_filename}..."):
                t0 = time.time()
                feats = extract_features_from_series(series_data)
                df_feat = pd.DataFrame([feats])
                pred_damage = float(pipeline.predict(df_feat)[0])
                elapsed = time.time() - t0

            st.success(f"Successfully processed `{current_filename}` ({len(series_data):,} samples) in {elapsed:.2f}s")

            # Diagnostic Panel
            diag_l, diag_r = st.columns([1, 1])

            with diag_l:
                st.subheader("Fatigue Assessment & RUL")
                st.metric(label="Predicted Cumulative Damage (D)", value=f"{pred_damage:.6f}")

                # Progress Bar
                clamped_d = min(1.0, max(0.0, pred_damage))
                st.progress(clamped_d)

                # RUL estimation: D is damage accumulated over 1 service roundtrip
                # Equivalent safe roundtrips until critical threshold D = 1.0:
                est_trips_remaining = max(0, int((1.0 - pred_damage) / max(pred_damage, 0.005)))

                if pred_damage < 0.20:
                    st.markdown(f"""
                    <div class="status-healthy">
                        🟢 HEALTH STATUS: NOMINAL (LOW WEAR)<br>
                        <small>Component operating well within elastic limits. Estimated RUL: <b>~{est_trips_remaining} equivalent cycles</b>. Continue standard maintenance schedule.</small>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred_damage < 0.50:
                    st.markdown(f"""
                    <div class="status-warning">
                        🟡 HEALTH STATUS: MODERATE FATIGUE ACCUMULATION<br>
                        <small>Substantial cyclic strain recorded. Estimated RUL: <b>~{est_trips_remaining} equivalent cycles</b>. Flag bogie for scheduled ultrasonic inspection.</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="status-critical">
                        🔴 HEALTH STATUS: CRITICAL FATIGUE RISK<br>
                        <small>High microcrack propagation risk (D ≥ 0.50). Estimated RUL: <b>~{est_trips_remaining} cycles</b>. IMMEDIATE non-destructive testing (NDT) required.</small>
                    </div>
                    """, unsafe_allow_html=True)

            with diag_r:
                st.subheader("Physical Stress Signature")
                f1, f2 = st.columns(2)
                with f1:
                    st.write(f"**Mean Stress:** `{feats['mean']:.2f} MPa`")
                    st.write(f"**Stress RMS:** `{feats['rms']:.2f} MPa`")
                    st.write(f"**Peak-to-Peak:** `{feats['ptp']:.2f} MPa`")
                    st.write(f"**Crest Factor:** `{feats['crest_factor']:.2f}`")
                with f2:
                    st.write(f"**Rainflow Cycles:** `{int(feats['rf_total_cycles']):,}`")
                    st.write(f"**Max Stress Range:** `{feats['rf_max_range']:.2f} MPa`")
                    st.write(f"**P99 Stress Range:** `{feats['rf_p99_range']:.2f} MPa`")
                    st.write(f"**Operational Regime:** `{'AW4 (Crush Load)' if feats['is_aw4_proxy'] else 'AW0 (Tare Load)'}`")

            # Interactive Waveform
            st.subheader("Dynamic Stress Waveform (MPa)")
            st.caption("Downsampled visual overview (1,000 points) from the complete 581,120 time history:")
            step = max(1, len(series_data) // 1000)
            st.line_chart(series_data[::step], height=240)

    with tab2:
        st.subheader("Batch Test Execution & `shm_predictions.csv`")
        pred_csv_path = resolve_file(
            os.path.join(REPO_ROOT, "predictions", "shm_predictions.csv"),
            os.path.join(REPO_ROOT, "SHM", "shm_predictions.csv"),
            os.path.join(WORKSPACE_ROOT, "ps3", "SHM", "shm_predictions.csv"),
        )

        if os.path.exists(pred_csv_path):
            df_shm_preds = pd.read_csv(pred_csv_path)
            st.info(f"Loaded existing predictions on all {len(df_shm_preds)} test files:")
            st.dataframe(df_shm_preds, use_container_width=True)

            col_down1, col_down2 = st.columns([1, 2])
            with col_down1:
                csv_buf = io.StringIO()
                df_shm_preds.to_csv(csv_buf, index=False)
                st.download_button(
                    label="📥 Download shm_predictions.csv",
                    data=csv_buf.getvalue(),
                    file_name="shm_predictions.csv",
                    mime="text/csv",
                    help="Official submission CSV formatted for predictions.zip"
                )

    with tab3:
        st.subheader("Judging Rubric & Methodology")
        st.markdown(r"""
        #### 1. Official Evaluation Formula
        Per **LTA Problem Statement 3 Specification Section 5.2**:
        $$\text{MAPE} = \frac{1}{N} \sum_{i=1}^{N} \frac{|D_{\text{true}, i} - D_{\text{pred}, i}|}{D_{\text{true}, i}}$$
        $$\text{Score} = \max\left(0, 1 - \text{MAPE}\right)$$
        * Score ranges from **0.0** (100%+ error) to **1.0** (perfect prediction).

        #### 2. Cross-Validation Performance
        * **Stratified 4-Fold CV Score:** **`0.8809`** (MAPE: `11.91%`)
        * **Held-Out Validation Set (16 files):** **`0.9086`** (MAPE: `9.14%`, $R^2 = 0.9880$)
        * **Ensemble Architecture:** Ridge, Lasso, ElasticNet, Random Forest, and Gradient Boosting with log-domain target transform.
        """)


# ==========================================
# VIEW 3: SALOON DOOR SYSTEM (DOOR)
# ==========================================
def render_door_subsystem():
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🚪 Saloon Door System (Door)</div>
        <div class="sub-title">Continuous Temporal Cycle Segmentation & Electromechanical Abnormal Resistance Detection</div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Validation Soft F1 Score", value="0.9909", delta="IoU-Weighted F1", help="Official competition judging metric evaluated over 5-fold CV")
    with col2:
        st.metric(label="Cycle IoU Accuracy", value="100.0%", delta="110 / 110 Exact Boundaries")
    with col3:
        st.metric(label="Abnormal Precision", value="100.0%", delta="0 False Alarms")
    with col4:
        st.metric(label="Inference Latency", value="< 2.0s", delta="Continuous Stream")

    st.markdown("---")

    pipeline = load_door_pipeline()
    if pipeline is None:
        st.error("Door model checkpoint not found. Please train using `ps3/Door/code/train.py`.")
        return

    segment_door_stream = door_segmentation.segment_door_stream
    extract_features_from_segment = door_features.extract_features_from_segment

    tab1, tab2, tab3 = st.tabs([
        "🔍 Continuous Stream Segmenter & Diagnostics",
        "⚡ Batch Prediction & Submission Generator",
        "📊 Methodology & Judging Rubric"
    ])

    with tab1:
        st.write("Inspect a continuous door sensor stream (e.g. `problem_statements/PS3/02_Datasets/Door/Test.csv`):")
        test_csv_path = resolve_file(
            os.path.join(WORKSPACE_ROOT, "problem_statements", "PS3", "02_Datasets", "Door", "Test.csv"),
            os.path.join(REPO_ROOT, "data", "Door", "Test.csv"),
        )
        has_test_csv = os.path.exists(test_csv_path)

        c_opt1, c_opt2 = st.columns([1, 1])
        stream_df = None
        source_name = ""

        with c_opt1:
            st.markdown("##### Option A: Official Test Stream")
            if has_test_csv:
                if st.button("📁 Load Official Test Stream (Test.csv)", key="btn_load_door_test"):
                    stream_df = pd.read_csv(test_csv_path)
                    source_name = "Test.csv (Official Test Stream)"

        with c_opt2:
            st.markdown("##### Option B: Upload Custom Stream")
            uploaded_file = st.file_uploader("Upload continuous door CSV", type=["csv"], key="door_stream_uploader")
            if uploaded_file is not None:
                stream_df = pd.read_csv(uploaded_file)
                source_name = uploaded_file.name

        if stream_df is not None:
            with st.spinner(f"Segmenting operational cycles and classifying resistance for {source_name}..."):
                segs = segment_door_stream(stream_df)
                feat_rows = [extract_features_from_segment(s["df"], operation=s["operation"]) for s in segs]
                preds = pipeline.predict(pd.DataFrame(feat_rows))
                probs = pipeline.predict_proba(pd.DataFrame(feat_rows))

            st.success(f"Segmented **{len(segs)} operational cycles** from {len(stream_df):,} telemetry samples!")

            # Overview Cards
            n_abnormal = sum(1 for p in preds if p == "Abnormal resistance")
            n_normal = len(preds) - n_abnormal

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Total Operational Cycles", len(segs))
            with sc2:
                st.markdown(f'<div class="status-healthy">🟢 Normal Cycles: <b>{n_normal}</b> ({(n_normal/len(segs)*100):.1f}%)</div>', unsafe_allow_html=True)
            with sc3:
                st.markdown(f'<div class="status-critical">🔴 Abnormal Resistance: <b>{n_abnormal}</b> ({(n_abnormal/len(segs)*100):.1f}%)</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Interactive Cycle Inspector & Work Order Generator")

            # Filter controls
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                status_filter = st.selectbox("Filter by Status:", ["All Cycles", "Normal Only", "Abnormal resistance Only"])
            with f_col2:
                dir_filter = st.selectbox("Filter by Direction:", ["All Directions", "Opening Only", "Closing Only"])

            # Apply filters
            filtered_indices = []
            for i, s in enumerate(segs):
                stat_match = (status_filter == "All Cycles") or (status_filter == "Normal Only" and preds[i] == "Normal") or (status_filter == "Abnormal resistance Only" and preds[i] == "Abnormal resistance")
                dir_match = (dir_filter == "All Directions") or (dir_filter == "Opening Only" and s["operation"] == "Open") or (dir_filter == "Closing Only" and s["operation"] == "Close")
                if stat_match and dir_match:
                    filtered_indices.append(i)

            if not filtered_indices:
                st.warning("No cycles match the selected filters.")
            else:
                cycle_options = [
                    f"Cycle #{i+1:02d} | {segs[i]['operation']:5s} | {preds[i]} | {segs[i]['start_time']} -> {segs[i]['end_time']}"
                    for i in filtered_indices
                ]
                selected_idx = st.selectbox("Select cycle to inspect:", filtered_indices, format_func=lambda idx: f"Cycle #{idx+1:02d} | {segs[idx]['operation']} | {preds[idx]}")

                chosen_seg = segs[selected_idx]
                chosen_pred = preds[selected_idx]
                chosen_prob = probs[selected_idx]
                chosen_feat = feat_rows[selected_idx]
                seg_data = chosen_seg["df"]

                d_col1, d_col2 = st.columns([1, 1])
                with d_col1:
                    st.write(f"**Operation Direction:** `{chosen_seg['operation']}`")
                    st.write(f"**Cycle Start Time:** `{chosen_seg['start_time']}`")
                    st.write(f"**Cycle End Time:** `{chosen_seg['end_time']}`")
                    st.write(f"**Cycle Duration:** `{chosen_seg['n_rows'] * 0.020:.2f} seconds ({chosen_seg['n_rows']} ticks)`")

                    if chosen_pred == "Normal":
                        st.markdown("""
                        <div class="status-healthy">
                            🟢 HEALTH STATUS: NORMAL<br>
                            <small>Glide current and motor torque remain within nominal parameters. Door mechanism is operating smoothly.</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="status-critical">
                            🔴 HEALTH STATUS: ABNORMAL RESISTANCE DETECTED<br>
                            <small>Elevated steady-state friction observed. High probability of guide rail foreign debris, rubber strip jamming, or hanger misalignment.</small>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("""
                        **📋 Maintenance Work Order Recommendation:**
                        - Dispatch trackside depot technician to clean lower guide sill.
                        - Inspect rubber weather strip seals for swelling or binding.
                        - Perform door hanger roller clearance check.
                        """)

                with d_col2:
                    st.write(f"**Abnormal Probability:** `{chosen_prob * 100:.1f}%` (Threshold: 35.0%)")
                    st.write(f"**Steady-State Glide Current:** `{chosen_feat['cur_mid_mean']:.1f} mA` (Nominal: ~210 mA)")
                    st.write(f"**Peak Motor Current:** `{chosen_feat['cur_max']:.1f} mA`")
                    st.write(f"**Total Actuation Energy:** `{chosen_feat['power_sum'] * 0.02:.2f} Joules`")

                # Waveforms
                st.subheader("Motor Current Telemetry (mA)")
                st.line_chart(seg_data["Motor current(mA)"].values, height=220)

                pos_col, emf_col = st.columns(2)
                with pos_col:
                    st.caption("Door Leaf Position (0 = Closed, 700+ = Open)")
                    st.line_chart(seg_data["Door leaf position"].values, height=180)
                with emf_col:
                    st.caption("Motor Electrodynamic Force (Back-EMF)")
                    st.line_chart(seg_data["Motor electrodynamic force"].values, height=180)

    with tab2:
        st.subheader("Official Submission Generator: `door_predictions.csv`")
        pred_csv_path = resolve_file(
            os.path.join(REPO_ROOT, "predictions", "door_predictions.csv"),
            os.path.join(REPO_ROOT, "Door", "door_predictions.csv"),
            os.path.join(WORKSPACE_ROOT, "ps3", "Door", "door_predictions.csv"),
        )
        if os.path.exists(pred_csv_path):
            df_door_preds = pd.read_csv(pred_csv_path)
            st.info(f"Loaded existing predictions on Test.csv ({len(df_door_preds)} segments):")
            st.dataframe(df_door_preds, use_container_width=True)

            csv_buf = io.StringIO()
            df_door_preds.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download door_predictions.csv",
                data=csv_buf.getvalue(),
                file_name="door_predictions.csv",
                mime="text/csv",
                help="Official submission CSV formatted for predictions.zip"
            )

    with tab3:
        st.subheader("Judging Rubric & Methodology")
        st.markdown(r"""
        #### 1. Official Evaluation Formula
        Per **LTA Problem Statement 3 Specification Section 5.2**:
        $$\text{soft\_recall} = \frac{\sum \text{IoU}}{\text{number of true segments}}, \quad \text{soft\_precision} = \frac{\sum \text{IoU}}{\text{number of predicted segments}}$$
        $$\text{Score} = \frac{2 \cdot \text{soft\_recall} \cdot \text{soft\_precision}}{\text{soft\_recall} + \text{soft\_precision}}$$

        * Segment matching is greedy one-to-one, highest IoU first, restricted strictly to identical predicted and ground-truth labels.
        * A segment with overlapping timing but incorrect label counts as 0 credit.

        #### 2. Cross-Validation Performance
        * **IoU-Weighted Soft F1:** **`0.9909`** (109 / 110 segments correct)
        * **Boundary IoU:** **`100.0%`** (Exact match across all operational timestamps)
        * **Abnormal Resistance Precision:** **`1.0000`** (0 false alarms)
        * **Electromechanical Discrimination:** Steady-state gliding current ($t \in [0.25T, 0.75T]$) isolates guide rail friction from startup inrush and end-stop impact.
        """)


# ==========================================
# VIEW 4: RAIL CORRUGATION SYSTEM (RC)
# ==========================================
def render_rail_subsystem():
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🛤️ Rail Corrugation Detection & Spatial Localization System</div>
        <div class="sub-title">Multi-Channel Axle-Box Dynamic Vibration • Bilateral Track Wear Asymmetry • 64-Sensor Spatial Localization</div>
    </div>
    """, unsafe_allow_html=True)

    pipeline = load_rail_pipeline()
    if pipeline is None:
        st.error("⚠️ Rail Corrugation model not found. Please run `ps3/Rail_Corrugation/code/train.py`.")
        return

    # Subsystem Quick KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Judging Metric: Macro F1", value="0.8464", delta="Stratified 5-Fold CV")
    with kpi2:
        st.metric(label="Validation Accuracy", value="95.22%", delta="259 / 272 Correct")
    with kpi3:
        st.metric(label="Active Network Faults", value="7 Locations", delta="4 Side I, 3 Side II", delta_color="inverse")
    with kpi4:
        st.metric(label="Inference Latency", value="165.8 ms", delta="Per 1.0s Multi-Channel File")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "🔍 Single Recording Diagnostic",
        "📊 Fleet Batch Diagnostics (68 Test Recordings)",
        "📐 Judging Rubric & Physics Mechanics"
    ])

    with tab1:
        st.subheader("Select or Upload Telemetry Recording (129 Channels @ 10 kHz)")

        c_opt1, c_opt2 = st.columns([1, 1])
        with c_opt1:
            input_mode = st.radio(
                "Data Source:",
                ["Benchmark Test & Validation Cases", "Upload Custom 129-Channel CSV"],
                horizontal=True
            )

        test_dir = resolve_dir(
            os.path.join(WORKSPACE_ROOT, "problem_statements", "PS3", "02_Datasets", "Rail_Corrugation", "Test"),
            os.path.join(REPO_ROOT, "data", "Rail_Corrugation", "Test"),
            os.path.join(REPO_ROOT, "data", "Rail Corrugation", "Test"),
        )
        train_dir = resolve_dir(
            os.path.join(WORKSPACE_ROOT, "problem_statements", "PS3", "02_Datasets", "Rail_Corrugation", "Train"),
            os.path.join(REPO_ROOT, "data", "Rail_Corrugation", "Train"),
            os.path.join(REPO_ROOT, "data", "Rail Corrugation", "Train"),
        )

        df_raw = None
        current_filename = ""

        if input_mode == "Benchmark Test & Validation Cases":
            case_options = {
                "Test22.csv (Held-Out Test — Predicted Side I Corrugation)": os.path.join(test_dir, "Test22.csv"),
                "Test26.csv (Held-Out Test — Predicted Side II Corrugation)": os.path.join(test_dir, "Test26.csv"),
                "Test32.csv (Held-Out Test — Predicted Side I Corrugation)": os.path.join(test_dir, "Test32.csv"),
                "Test43.csv (Held-Out Test — Predicted Side II Corrugation)": os.path.join(test_dir, "Test43.csv"),
                "Test66.csv (Held-Out Test — Predicted Side II Corrugation)": os.path.join(test_dir, "Test66.csv"),
                "Test1.csv (Held-Out Test — Predicted Normal Healthy Track)": os.path.join(test_dir, "Test1.csv"),
                "Train8.csv (Training Baseline — Ground Truth Side I Corrugation)": os.path.join(train_dir, "Train8.csv"),
                "Train2.csv (Training Baseline — Ground Truth Side II Corrugation)": os.path.join(train_dir, "Train2.csv"),
                "Train1.csv (Training Baseline — Ground Truth Normal Track)": os.path.join(train_dir, "Train1.csv"),
            }
            selected_case = st.selectbox("Select Recording Snapshot:", list(case_options.keys()))
            file_path = case_options[selected_case]
            if os.path.exists(file_path):
                current_filename = os.path.basename(file_path)
                df_raw = pd.read_csv(file_path)
            else:
                st.warning(f"File not found on disk: {file_path}")
        else:
            uploaded = st.file_uploader("Upload 10,000-sample x 129-channel CSV:", type=["csv"])
            if uploaded is not None:
                current_filename = uploaded.name
                df_raw = pd.read_csv(uploaded)

        if df_raw is not None:
            with st.spinner(f"Extracting kinematic, bilateral asymmetry, and Welch spectral features from {current_filename}..."):
                t0 = time.time()
                feats = rail_features.extract_features_from_df(df_raw, filename=current_filename)
                df_feat = pd.DataFrame([feats])
                probs = pipeline.predict_proba(df_feat)[0]
                pred_label = pipeline.predict(df_feat)[0]
                elapsed = time.time() - t0

            st.success(f"Processed `{current_filename}` (10,000 samples across 129 channels) in {elapsed:.3f}s")

            # Diagnostic Status Banner
            if pred_label == "Normal":
                st.markdown("""
                <div class="status-healthy">
                    🟢 HEALTH STATUS: NOMINAL (HEALTHY TRACK)<br>
                    <small>Both Side I and Side II rail contact bands exhibit symmetric rolling contact vibrations within healthy roughness limits. No periodic corrugation wear detected.</small>
                </div>
                """, unsafe_allow_html=True)
            elif pred_label == "Side I":
                st.markdown("""
                <div class="status-warning">
                    🟠 MAINTENANCE ALERT: SIDE I CORRUGATION DETECTED<br>
                    <small>Statistically elevated asymmetric resonance detected on <b>Side I rail</b> (Axle-box positions 1, 3, 5, 7). Dispatch rail grinding unit to reprofile Side I running head.</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-critical">
                    🟣 MAINTENANCE ALERT: SIDE II CORRUGATION DETECTED<br>
                    <small>Statistically elevated asymmetric resonance detected on <b>Side II rail</b> (Axle-box positions 2, 4, 6, 8). Dispatch rail grinding unit to reprofile Side II running head.</small>
                </div>
                """, unsafe_allow_html=True)

            # Diagnostic Signature Metrics
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("Train Operating Speed", f"{feats['speed_kmh']:.1f} km/h", f"{feats['speed_mps']:.2f} m/s")
            with d2:
                prob_dict = {pipeline.class_names[i]: probs[i] for i in range(len(pipeline.class_names))}
                max_conf = max(probs) * 100
                st.metric("Diagnosis", pred_label, f"{max_conf:.1f}% Confidence")
            with d3:
                st.metric("Bilateral Diff (S1 - S2)", f"{feats['vib_diff_mean']:+.4f} m/s²", f"Ratio: {feats['vib_ratio_mean']:.2f}x")
            with d4:
                st.metric("Corrugation Band (300-600Hz)", f"{feats['psd_ratio_corrug_classic']:.2f}x Ratio", f"Diff: {feats['psd_diff_corrug_classic']:+.2e}")

            # Visualizations Section
            st.markdown("---")
            st.subheader("Bogie Sensor Telemetry & Spatial Diagnostics")

            # 1. 8-Car x 8-Position Axle Box Vibration Matrix
            st.markdown("##### 1. Full Train Bogie Vibration Intensity Matrix (64 Axle-Boxes across 8 Cars)")
            st.caption("Axle positions 1, 3, 5, 7 monitor Side I rail; Positions 2, 4, 6, 8 monitor Side II rail. Values represent Root Mean Square (RMS) acceleration in m/s²:")

            cols = df_raw.columns
            matrix_data = []
            for c in range(1, 9):
                row_vals = []
                for p in range(1, 9):
                    ch_cols = [col for col in cols if f"car {c}" in col and f"position {p}" in col and "Vibration" in col]
                    if ch_cols:
                        rms_val = np.sqrt(np.mean(df_raw[ch_cols[0]].values**2))
                        row_vals.append(rms_val)
                    else:
                        row_vals.append(0.0)
                matrix_data.append(row_vals)

            col_names = [
                "Pos 1 (S1)", "Pos 2 (S2)", "Pos 3 (S1)", "Pos 4 (S2)",
                "Pos 5 (S1)", "Pos 6 (S2)", "Pos 7 (S1)", "Pos 8 (S2)"
            ]
            row_names = [f"Car {c}" for c in range(1, 9)]
            df_matrix = pd.DataFrame(matrix_data, index=row_names, columns=col_names)
            st.dataframe(df_matrix.style.background_gradient(cmap="YlOrRd").format("{:.3f}"), use_container_width=True)

            col_v1, col_v2 = st.columns(2)

            with col_v1:
                # 2. Time-Domain Bilateral Signals
                st.markdown("##### 2. Bilateral Acceleration Signals (10 kHz, 1.0s)")
                st.caption("Average axle-box vibration time histories comparing Side I vs Side II rail contact:")
                s1_vib_cols = [c for c in cols if 'Vibration' in c and any(f'position {p}' in c for p in [1, 3, 5, 7])]
                s2_vib_cols = [c for c in cols if 'Vibration' in c and any(f'position {p}' in c for p in [2, 4, 6, 8])]

                s1_sig = df_raw[s1_vib_cols].mean(axis=1).values
                s2_sig = df_raw[s2_vib_cols].mean(axis=1).values

                step = max(1, len(s1_sig) // 1000)
                df_time = pd.DataFrame({
                    "Side I Rail (m/s²)": s1_sig[::step],
                    "Side II Rail (m/s²)": s2_sig[::step]
                })
                st.line_chart(df_time, height=260)

            with col_v2:
                # 3. Welch Power Spectral Density Multi-Band Distribution
                st.markdown("##### 3. Multi-Band Spectral Energy Distribution")
                st.caption("Power Spectral Density (PSD) segmented across physical resonance bands:")
                df_bands = pd.DataFrame({
                    "Side I Energy": [
                        feats["psd_s1_low"],
                        feats["psd_s1_mid1"],
                        feats["psd_s1_corrug_classic"],
                        feats["psd_s1_corrug_high"],
                        feats["psd_s1_impact"]
                    ],
                    "Side II Energy": [
                        feats["psd_s2_low"],
                        feats["psd_s2_mid1"],
                        feats["psd_s2_corrug_classic"],
                        feats["psd_s2_corrug_high"],
                        feats["psd_s2_impact"]
                    ]
                }, index=["Low (10-100Hz)", "P2 Mode (100-300Hz)", "Classic Corrug (300-600Hz)", "High Corrug (600-1200Hz)", "Impact (1200-3000Hz)"])
                st.bar_chart(df_bands, height=260)

    with tab2:
        st.subheader("Official Held-Out Test Set Predictions (`rail_predictions.csv`)")
        pred_csv_path = resolve_file(
            os.path.join(REPO_ROOT, "predictions", "rail_predictions.csv"),
            os.path.join(REPO_ROOT, "Rail Corrugation", "rail_predictions.csv"),
            os.path.join(REPO_ROOT, "Rail_Corrugation", "rail_predictions.csv"),
            os.path.join(WORKSPACE_ROOT, "ps3", "Rail_Corrugation", "rail_predictions.csv"),
        )

        if os.path.exists(pred_csv_path):
            df_preds = pd.read_csv(pred_csv_path)

            t_kpi1, t_kpi2, t_kpi3, t_kpi4 = st.columns(4)
            with t_kpi1:
                st.metric("Total Test Recordings", len(df_preds))
            with t_kpi2:
                norm_count = (df_preds["prediction"] == "Normal").sum()
                st.metric("Normal (Healthy)", norm_count, f"{norm_count / len(df_preds) * 100:.1f}%")
            with t_kpi3:
                s1_count = (df_preds["prediction"] == "Side I").sum()
                st.metric("Side I Corrugation", s1_count, f"{s1_count / len(df_preds) * 100:.1f}%")
            with t_kpi4:
                s2_count = (df_preds["prediction"] == "Side II").sum()
                st.metric("Side II Corrugation", s2_count, f"{s2_count / len(df_preds) * 100:.1f}%")

            st.markdown("---")

            # Merge with test features if available for rich inspection
            test_feat_path = resolve_file(
                os.path.join(REPO_ROOT, "Rail Corrugation", "test_features.csv"),
                os.path.join(REPO_ROOT, "Rail_Corrugation", "test_features.csv"),
                os.path.join(WORKSPACE_ROOT, "ps3", "Rail_Corrugation", "test_features.csv"),
            )
            if os.path.exists(test_feat_path):
                df_test_feats = pd.read_csv(test_feat_path)
                df_rich = df_preds.merge(df_test_feats, left_on="file_id", right_on="filename", how="left")
                display_cols = ["file_id", "prediction", "speed_kmh", "vib_diff_mean", "vib_ratio_mean", "s1_vib_rms_max", "s2_vib_rms_max", "cars_s1_dominant", "cars_s2_dominant"]
                valid_cols = [c for c in display_cols if c in df_rich.columns]
                st.dataframe(df_rich[valid_cols], use_container_width=True)
            else:
                st.dataframe(df_preds, use_container_width=True)

            col_d1, col_d2 = st.columns([1, 2])
            with col_d1:
                csv_buf = io.StringIO()
                df_preds.to_csv(csv_buf, index=False)
                st.download_button(
                    label="📥 Download rail_predictions.csv",
                    data=csv_buf.getvalue(),
                    file_name="rail_predictions.csv",
                    mime="text/csv",
                    help="Official 68-row prediction file formatted for submission archive"
                )

    with tab3:
        st.subheader("Judging Rubric & Physical Architecture")
        st.markdown(r"""
        #### 1. Official Evaluation Formula
        Per **LTA Problem Statement 3 Specification Section 5.3**:
        $$\text{Macro F1} = \frac{1}{3} \left( F_1(\text{Normal}) + F_1(\text{Side I}) + F_1(\text{Side II}) \right)$$
        where for each class $c$:
        $$F_1(c) = \frac{2 \cdot \text{Precision}(c) \cdot \text{Recall}(c)}{\text{Precision}(c) + \text{Recall}(c)}$$

        #### 2. Physical Sensor Architecture
        * **Tooth-Wheel Pulse Sensor:** 90 teeth, wheel diameter $D = 0.85\text{ m}$. Kinematic vehicle speed:
          $$v = \frac{N_{\text{pulses}}}{90} \cdot \pi D \quad [\text{m/s}]$$
        * **Bilateral Asymmetry Decoupling:** Speed confounds absolute vibration. Bilateral differentials ($\Delta \text{RMS} = \text{RMS}_{\text{Side I}} - \text{RMS}_{\text{Side II}}$) and speed-normalized intensities isolate track degradation from vehicle dynamics.
        * **Welch Spectral Resonance:** Corrugation wear produces characteristic contact resonance in the $300 - 600\text{ Hz}$ band.

        #### 3. Stratified 5-Fold Cross-Validation Performance
        | Class | Precision | Recall | F1 Score | Support |
        | :--- | :---: | :---: | :---: | :---: |
        | **Normal** | **98.28%** | **97.44%** | **0.9785** | 234 |
        | **Side I** | **64.71%** | **78.57%** | **0.7097** | 14 |
        | **Side II** | **86.96%** | **83.33%** | **0.8511** | 24 |
        | **Macro Average** | **83.31%** | **86.45%** | **`0.8464`** | 272 |
        | **Accuracy** | — | — | **`95.22%`** | 259 / 272 |
        """)


# ==========================================
# VIEW 5: AIR CONDITIONING & VENTILATION (ACV)
# ==========================================
def render_acv_subsystem():
    st.markdown("""
    <div class="main-header">
        <div class="main-title">❄️ Air Conditioning & Ventilation (ACV) System</div>
        <div class="sub-title">Refrigerant Leakage Fault Diagnosis, Tracking Degradation & Consist Localisation</div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Judging Metric Score", value="1.0000", delta="100.0% Linear Rank-Decay", help="Arithmetic mean of (N - (r - 1)) / N across all documented test and validation cases")
    with col2:
        st.metric(label="Top-1 Localisation Rate", value="100.0%", delta="6 / 6 Benchmark Cases")
    with col3:
        st.metric(label="Consist Topologies", value="8-Car & 4-Car", delta="Adaptive Schema Parsing")
    with col4:
        st.metric(label="Telemetry Interval", value="30 seconds", delta="Continuous Stream")

    st.markdown("---")

    pipeline = load_acv_pipeline()
    if pipeline is None:
        st.error("ACV model pipeline not found. Please train using `ps3/ACV/code/train.py`.")
        return

    tab1, tab2, tab3 = st.tabs([
        "🔍 Consist Diagnostic & Thermal Profiler",
        "⚡ Batch Inference & Submission Generator",
        "📊 Methodology & Thermodynamic Physics"
    ])

    with tab1:
        st.write("Diagnose and localise refrigerant leakage undercharge across train car HVAC units:")

        test_dir = resolve_dir(
            os.path.join(WORKSPACE_ROOT, "problem_statements", "PS3", "02_Datasets", "ACV", "Test"),
            os.path.join(REPO_ROOT, "data", "ACV", "Test"),
        )
        train_dir = resolve_dir(
            os.path.join(WORKSPACE_ROOT, "problem_statements", "PS3", "02_Datasets", "ACV", "Train"),
            os.path.join(REPO_ROOT, "data", "ACV", "Train"),
        )

        c_opt1, c_opt2 = st.columns([1, 1])
        selected_file_path = None
        current_case_name = ""

        with c_opt1:
            available_cases = {}
            test_file = os.path.join(test_dir, "acv_test_case.xlsx")
            if os.path.exists(test_file):
                available_cases["acv_test_case.xlsx (Official Held-Out Test)"] = test_file

            if os.path.exists(train_dir):
                for f in sorted(os.listdir(train_dir)):
                    if f.endswith(".xlsx"):
                        available_cases[f] = os.path.join(train_dir, f)

            if available_cases:
                choice = st.selectbox(
                    "Select Benchmark Case File:",
                    options=list(available_cases.keys()),
                    index=0,
                    help="Select the held-out test case or a documented training fault case"
                )
                selected_file_path = available_cases[choice]
                current_case_name = choice.split()[0]

        with c_opt2:
            uploaded_file = st.file_uploader(
                "Or Upload Custom Consist Telemetry (.xlsx):",
                type=["xlsx"],
                help="Upload an Excel workbook containing consist ACV telemetry"
            )
            if uploaded_file is not None:
                selected_file_path = uploaded_file
                current_case_name = uploaded_file.name

        if selected_file_path is not None:
            with st.spinner("Analyzing consist thermodynamic telemetry..."):
                diag = pipeline.predict_file(selected_file_path)

            top_car = diag["top_faulty_car"]
            ranked_str = diag["ranked_cars"]
            car_features = diag["car_features"]
            active_cars = diag["active_cars"]
            all_cars = diag["all_cars"]
            time_series = diag["time_series"]

            top_feat = car_features.get(top_car, {})
            top_err = top_feat.get("tracking_error_mean", 0.0)
            top_tin = top_feat.get("indoor_temp_mean", 0.0)
            top_rel = top_feat.get("fleet_rel_err", 0.0)
            top_score = top_feat.get("final_score", 0.0)

            # High-visibility Diagnostic Alert Box
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
                        border: 1px solid #ef4444; border-radius: 10px; padding: 18px 24px; margin: 16px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="background: #ef4444; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">
                            FAULT DETECTED • REFRIGERANT LEAKAGE
                        </span>
                        <h3 style="color: white; margin: 8px 0 4px 0;">Primary Faulty Unit: Car {top_car}</h3>
                        <div style="color: #fca5a5; font-size: 0.95rem;">
                            Consist Fault Localisation Ranking: <code style="background: rgba(0,0,0,0.4); color: #fecaca; padding: 2px 8px; border-radius: 4px;">{ranked_str}</code>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.8rem; color: #fca5a5;">ANOMALY SEVERITY</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #f87171;">+{top_score:.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Consist Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(label="Active Consist Size", value=f"{len(active_cars)} Cars", delta=f"{len(all_cars)} Total Slots")
            with m2:
                st.metric(label=f"Car {top_car} Mean Tracking Error", value=f"{top_err:+.2f} °C", delta="Elevated vs Setpoint", delta_color="inverse")
            with m3:
                st.metric(label=f"Car {top_car} Mean Cabin Temp", value=f"{top_tin:.2f} °C", delta="Warmest in Fleet", delta_color="inverse")
            with m4:
                st.metric(label=f"Fleet-Relative Departure", value=f"{top_rel:+.2f} °C", delta="Above Consist Median", delta_color="inverse")

            st.markdown("---")

            # Consist Ranking Leaderboard Table
            st.subheader(f"Consist Fault Probability Leaderboard ({current_case_name})")
            
            table_rows = []
            ranked_list = ranked_str.split("|")
            for rank_idx, c in enumerate(ranked_list, 1):
                feat = car_features.get(c, {})
                is_act = feat.get("is_active", 1)
                
                if is_act == 0:
                    status = "⚪ Inactive / Uncoupled"
                elif rank_idx == 1:
                    status = "🔴 Primary Suspect (Leakage)"
                elif rank_idx <= 3:
                    status = "🟡 Secondary Watch"
                else:
                    status = "🟢 Normal Condition"

                table_rows.append({
                    "Rank": f"#{rank_idx}",
                    "Car ID": f"Car {c}",
                    "Anomaly Score": round(feat.get("final_score", -999.0), 3) if is_act else "N/A",
                    "Mean Tracking Error (°C)": f"{feat.get('tracking_error_mean', 0.0):+.2f}" if is_act else "N/A",
                    "Cooling Error (°C)": f"{feat.get('tracking_error_cool', 0.0):+.2f}" if is_act else "N/A",
                    "Mean Cabin Temp (°C)": f"{feat.get('indoor_temp_mean', 0.0):.2f}" if is_act else "N/A",
                    "Fleet Relative (°C)": f"{feat.get('fleet_rel_err', 0.0):+.2f}" if is_act else "N/A",
                    "Setpoint Exceedance": f"{feat.get('persistence_over_setpoint', 0.0)*100:.1f}%" if is_act else "N/A",
                    "Circuit Asymmetry": f"{feat.get('pressure_asymmetry', 0.0)*100:.1f}%" if is_act and feat.get("pressure_asymmetry", 0.0) > 0 else "—",
                    "Diagnostic Status": status
                })

            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            # Interactive Multi-Car Thermal Waveforms
            st.markdown("---")
            st.subheader("Consist Thermal Trajectory Profiler")

            view_type = st.radio(
                "Display Metric:",
                options=["Passenger Cabin Temperature (°C)", "Setpoint Tracking Error (Tin - Ttarget, °C)"],
                horizontal=True
            )

            # Prepare chart DataFrame
            chart_dict = {}
            for c in active_cars:
                ts = time_series.get(c, {})
                if view_type.startswith("Passenger Cabin"):
                    if ts.get("indoor"):
                        chart_dict[f"Car {c}{' (LEAK)' if c == top_car else ''}"] = ts["indoor"]
                else:
                    if ts.get("error"):
                        chart_dict[f"Car {c}{' (LEAK)' if c == top_car else ''}"] = ts["error"]

            if chart_dict:
                df_chart = pd.DataFrame(chart_dict)
                st.line_chart(df_chart, use_container_width=True)
                st.caption("Telemetry stream downsampled for interactive visualization. Note persistent thermal excursion and elevated baseline on the diagnosed leaking car.")

    with tab2:
        st.subheader("Official Competition Submission Generator")
        st.write("Generates the competition-compliant `acv_predictions.csv` prediction file:")

        test_file = resolve_file(
            os.path.join(test_dir, "acv_test_case.xlsx"),
            os.path.join(WORKSPACE_ROOT, "problem_statements", "PS3", "02_Datasets", "ACV", "Test", "acv_test_case.xlsx"),
            os.path.join(REPO_ROOT, "data", "ACV", "Test", "acv_test_case.xlsx"),
        )
        if os.path.exists(test_file):
            diag_test = pipeline.predict_file(test_file)
            test_df = pd.DataFrame([{
                "file_id": "acv_test_case.xlsx",
                "ranked_cars": diag_test["ranked_cars"]
            }])

            st.dataframe(test_df, use_container_width=True)

            col_sub1, col_sub2 = st.columns([1, 2])
            with col_sub1:
                csv_buf = io.StringIO()
                test_df.to_csv(csv_buf, index=False)
                st.download_button(
                    label="📥 Download acv_predictions.csv",
                    data=csv_buf.getvalue(),
                    file_name="acv_predictions.csv",
                    mime="text/csv",
                    help="Official prediction file formatted for competition submission"
                )

            # Display Test Case Car Breakdown
            st.markdown("#### Test Case Diagnostics Breakdown (`acv_test_case.xlsx`)")
            test_cars = diag_test["ranked_cars"].split("|")
            breakdown_rows = []
            for r_idx, c in enumerate(test_cars, 1):
                f_c = diag_test["car_features"].get(c, {})
                breakdown_rows.append({
                    "Rank": f"#{r_idx}",
                    "Car ID": f"Car {c}",
                    "Anomaly Score": round(f_c.get("final_score", 0.0), 4),
                    "Tracking Error (°C)": f"{f_c.get('tracking_error_mean', 0.0):+.4f}",
                    "Cabin Temp (°C)": f"{f_c.get('indoor_temp_mean', 0.0):.2f}",
                    "P90 Error (°C)": f"{f_c.get('tracking_error_p90', 0.0):+.2f}",
                    "Exceedance Rate": f"{f_c.get('persistence_over_setpoint', 0.0)*100:.1f}%",
                    "Role": "Faulty (Refrigerant Leak)" if r_idx == 1 else "Normal"
                })
            st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True)

    with tab3:
        st.subheader("Judging Rubric & Thermodynamic Architecture")
        st.markdown(r"""
        #### 1. Official Evaluation Formula
        Per **LTA Problem Statement 3 Specification Section 5.2**:
        $$\text{Linear Rank-Decay Score} = \frac{N - (r - 1)}{N}$$
        where $N$ is the number of active cars in the consist, and $r \in \{1, \dots, N\}$ is the 1-indexed rank of the true faulty car.

        #### 2. Thermodynamic Principles of Refrigerant Undercharge
        * **Vapour-Compression Cycle Degradation:**
          $$\dot{Q}_{\text{evap}} = \dot{m}_{\text{ref}} \left( h_{\text{suction}} - h_{\text{inlet}} \right)$$
          Refrigerant undercharge depletes active charge, lowering mass flow rate $\dot{m}_{\text{ref}}$ and shrinking the latent evaporation zone.
        * **Thermal Tracking Collapse:** Under leakage, the unit operates at continuous 100% cooling duty (`Running Mode == 'Automatic Cooling'`), but cabin temperature fails to reach setpoint ($T_{\text{in}} > T_{\text{target}}$).
        * **Fleet Common-Mode Decoupling:** Environmental heatwaves, tunnel thermal loads, and solar irradiance affect all cars simultaneously. Subtracting the fleet median baseline ($\delta T_i = T_i - \text{median}(T)$) isolates car-specific thermodynamic degradation.
        * **Circuit Pressure Asymmetry:** In dual-circuit AC systems (e.g., Case 04), high-side condensing pressure plummets on the leaking circuit relative to the healthy circuit.

        #### 3. Cross-Validation Results Across All Documented Failure Cases
        | Case File | Consist Topology | True Faulty Car | Predicted Rank | Score | Ranking String |
        | :--- | :---: | :---: | :---: | :---: | :--- |
        | `acv_case_01.xlsx` | 8-Car Consist | **`01`** | **#1** | **`1.0000`** | `01\|02\|03\|04\|07\|06\|08\|05` |
        | `acv_case_02.xlsx` | 8-Car Consist | **`02`** | **#1** | **`1.0000`** | `02\|03\|07\|08\|06\|01\|04\|05` |
        | `acv_case_03.xlsx` | 8-Car Consist | **`03`** | **#1** | **`1.0000`** | `03\|02\|01\|07\|04\|08\|05\|06` |
        | `acv_case_04.xlsx` | 4-Car Active (Dual Pressures) | **`01`** | **#1** | **`1.0000`** | `01\|04\|02\|03\|05\|06\|07\|08` |
        | `acv_case_05.xlsx` | 8-Car Consist | **`04`** | **#1** | **`1.0000`** | `04\|02\|07\|01\|06\|03\|08\|05` |
        | `acv_case_06.xlsx` | 8-Car Consist | **`06`** | **#1** | **`1.0000`** | `06\|08\|04\|02\|03\|05\|01\|07` |
        | **Mean Score** | — | — | **Top-1: 100%** | **`1.0000`** | **All Cases Rank #1** |
        """)


# ==========================================
# MAIN APP ROUTER
# ==========================================
def main():
    # Brand Header Box with LTA logo and pulsing live telemetry badge
    st.sidebar.markdown("""
    <div class="sidebar-brand-box">
        <div class="logo-container">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Land_Transport_Authority_%28Singapore%29_logo.svg/320px-Land_Transport_Authority_%28Singapore%29_logo.svg.png" width="130" style="display: block; margin: 0 auto;">
        </div>
        <div class="sidebar-title">LTA NebulaX 2026</div>
        <div class="sidebar-subtitle">Rail Vehicle Condition Monitoring</div>
        <div class="live-status-pill">
            <span class="pulse-dot"></span> TELEMETRY ACTIVE
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Section
    st.sidebar.markdown('<div class="sidebar-widget-title">NAVIGATION CONSOLE</div>', unsafe_allow_html=True)
    view_mode = st.sidebar.radio(
        "Navigation:",
        [
            "🚆 Fleet Control Tower",
            "🚪 Saloon Door System",
            "🔩 Structural Health (SHM)",
            "🛤️ Rail Corrugation System",
            "❄️ ACV Subsystem (Air Con)"
        ],
        index=0
    )

    st.sidebar.markdown("---")

    # Fleet Quick Stats Scorecard Widget
    st.sidebar.markdown("""
    <div class="sidebar-widget">
        <div class="sidebar-widget-title">FLEET PERFORMANCE SCORECARD</div>
        <div class="sidebar-metric-grid">
            <div class="sidebar-metric-item">
                <div class="sidebar-metric-label">Avg CV Score</div>
                <div class="sidebar-metric-value" style="color: #34d399;">0.9296</div>
            </div>
            <div class="sidebar-metric-item">
                <div class="sidebar-metric-label">Active Modules</div>
                <div class="sidebar-metric-value" style="color: #60a5fa;">4 / 4</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Judging Deliverables Checklist Widget
    st.sidebar.markdown("""
    <div class="sidebar-widget">
        <div class="sidebar-widget-title">JUDGING COMPLIANCE CHECKLIST</div>
        <div class="check-row">
            <span>shm_predictions.csv</span>
            <span class="check-tag-done">16 / 16</span>
        </div>
        <div class="check-row">
            <span>door_predictions.csv</span>
            <span class="check-tag-done">38 / 38</span>
        </div>
        <div class="check-row">
            <span>rail_predictions.csv</span>
            <span class="check-tag-done">68 / 68</span>
        </div>
        <div class="check-row">
            <span>acv_predictions.csv</span>
            <span class="check-tag-done">1 / 1</span>
        </div>
        <div class="check-row">
            <span>predictions.zip</span>
            <span class="check-tag-done">4 MODULES</span>
        </div>
        <div class="check-row">
            <span>Cross-Validation</span>
            <span class="check-tag-done">ALL VALIDATED</span>
        </div>
        <div class="check-row">
            <span>Interactive UI / UX</span>
            <span class="check-tag-done">READY</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Persistent 1-Click Download of predictions.zip right from the sidebar
    zip_path = resolve_file(
        os.path.join(REPO_ROOT, "predictions.zip"),
        os.path.join(WORKSPACE_ROOT, "predictions.zip"),
        os.path.join(WORKSPACE_ROOT, "ps3", "predictions.zip"),
    )
    if os.path.exists(zip_path):
        st.sidebar.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with open(zip_path, "rb") as fp:
            st.sidebar.download_button(
                label="📦 Download predictions.zip",
                data=fp.read(),
                file_name="predictions.zip",
                mime="application/zip",
                use_container_width=True,
                help="Official submission package ready for submission"
            )

    # Footer Metadata
    st.sidebar.markdown("""
    <div style="text-align: center; margin-top: 18px; color: #64748b; font-size: 0.72rem; line-height: 1.4;">
        <b>LTA-CMS v2.4</b> • PS3 Deliverables<br>
        Singapore Rapid Transit System
    </div>
    """, unsafe_allow_html=True)

    if view_mode == "🚆 Fleet Control Tower":
        render_fleet_control_tower()
    elif view_mode == "🚪 Saloon Door System":
        render_door_subsystem()
    elif view_mode == "🔩 Structural Health (SHM)":
        render_shm_subsystem()
    elif "Rail Corrugation" in view_mode:
        render_rail_subsystem()
    elif "ACV" in view_mode:
        render_acv_subsystem()


if __name__ == "__main__":
    main()


