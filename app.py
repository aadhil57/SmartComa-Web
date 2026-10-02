from pathlib import Path

import os
import zipfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download
# SMARTCOMA_HTML_DEDENT_PATCH
import textwrap

_original_st_markdown = st.markdown

def _smartcoma_markdown(body, *args, **kwargs):
    if isinstance(body, str):
        body = textwrap.dedent(body)
    return _original_st_markdown(body, *args, **kwargs)

st.markdown = _smartcoma_markdown

import plotly.graph_objects as go
from scipy.io import loadmat
from scipy.signal import resample_poly
import tensorflow as tf


# ============================================================
# SMART COMA MONITORING SYSTEM
# Professional AI Medical Dashboard
# EEG + IoT + Clinical Fusion
# ============================================================

st.set_page_config(
    page_title="SmartComa AI | Patient Monitoring",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/Fusion_V1_best_val_auc.keras"

EEG_STATS_PATH = "data/EEG_10sec_Train_Only_Normalization_Stats.csv"

MULTI_STATS_PATH = "data/Multimodal_TrainOnly_Normalization_Stats.csv"

# ============================================================
# PROJECT SETTINGS
# ============================================================

EEG_CHANNELS = [
    "Fp1", "Fp2",
    "F3", "F4",
    "C3", "C4",
    "P3", "P4",
    "O1", "O2",
    "F7", "F8",
    "T3", "T4",
    "T5", "T6",
    "Fz", "Cz", "Pz"
]

TARGET_FS = 100
WINDOW_SIZE = 1000
STEP_SIZE = 500

DECISION_THRESHOLD = 0.68


# ============================================================
# PREMIUM UI
# ============================================================

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%,
            rgba(37,99,235,0.16), transparent 28%),
        radial-gradient(circle at 85% 15%,
            rgba(14,165,233,0.12), transparent 30%),
        radial-gradient(circle at 50% 90%,
            rgba(99,102,241,0.10), transparent 32%),
        #070b16;
    color: #e8eefc;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #080d1c 0%,
            #0b1020 50%,
            #080b15 100%
        );
    border-right: 1px solid rgba(148,163,184,0.12);
}

section[data-testid="stSidebar"] * {
    color: #dce7ff !important;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1.2px;
    margin-bottom: 3px;
}

.subtitle {
    color: #8fa4c7;
    font-size: 14px;
    margin-bottom: 25px;
}

.topbar {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 20px;
    padding: 22px 25px;
    margin-bottom: 20px;
    backdrop-filter: blur(18px);
}

.card {
    background:
        linear-gradient(
            145deg,
            rgba(20,30,53,0.92),
            rgba(9,15,29,0.94)
        );
    border: 1px solid rgba(148,163,184,0.13);
    border-radius: 18px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.20);
}

.card:hover {
    border-color: rgba(59,130,246,0.30);
}

.card-title {
    color: #8fa4c7;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}

.card-value {
    color: #f5f8ff;
    font-size: 30px;
    font-weight: 800;
    margin-top: 7px;
}

.card-small {
    color: #6f85aa;
    font-size: 12px;
    margin-top: 4px;
}

.section-title {
    font-size: 21px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 14px;
}

.section-subtitle {
    color: #8fa4c7;
    font-size: 13px;
    margin-bottom: 15px;
}

.ai-panel {
    background:
        linear-gradient(
            135deg,
            rgba(30,64,175,0.22),
            rgba(14,116,144,0.14)
        );
    border: 1px solid rgba(59,130,246,0.30);
    border-radius: 22px;
    padding: 26px;
    box-shadow:
        0 0 35px rgba(37,99,235,0.08);
}

.good {
    color: #4ade80;
    font-weight: 800;
}

.warning {
    color: #fbbf24;
    font-weight: 800;
}

.poor {
    color: #fb7185;
    font-weight: 800;
}

.status-pill {
    display: inline-block;
    padding: 8px 15px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .5px;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.25);
    color: #4ade80;
}

.info-pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.20);
    color: #93c5fd;
}

.warning-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 14px;
    padding: 15px;
    color: #fcd34d;
}

.disclaimer {
    background: rgba(15,23,42,0.70);
    border: 1px solid rgba(148,163,184,0.13);
    border-radius: 15px;
    padding: 15px;
    color: #8fa4c7;
    font-size: 12px;
    line-height: 1.6;
}

.pipeline {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}

.pipeline-item {
    background: rgba(30,41,59,0.80);
    border: 1px solid rgba(96,165,250,0.15);
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 11px;
    font-weight: 700;
    color: #c7d7f4;
}

.pipeline-arrow {
    color: #3b82f6;
    font-weight: 900;
}

.footer {
    margin-top: 45px;
    padding: 20px;
    text-align: center;
    color: #526987;
    font-size: 11px;
}

div[data-testid="stMetric"] {
    background: rgba(15,23,42,0.70);
    border: 1px solid rgba(148,163,184,0.10);
    padding: 14px;
    border-radius: 15px;
}

button[kind="primary"] {
    border-radius: 12px;
    font-weight: 700;
}

</style>
""")


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_fusion_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
    except Exception:
        return None


@st.cache_data
def load_eeg_stats():

    if not os.path.exists(EEG_STATS_PATH):
        return None

    try:
        df = pd.read_csv(EEG_STATS_PATH)

        if "Channel" in df.columns:
            df = df.set_index("Channel")

        return df

    except Exception:
        return None


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction_probability" not in st.session_state:
    st.session_state.prediction_probability = 0.339639

if "prediction_status" not in st.session_state:
    st.session_state.prediction_status = "GOOD"

if "patient_id" not in st.session_state:
    st.session_state.patient_id = "0299"

if "eeg_data" not in st.session_state:
    st.session_state.eeg_data = None

if "window_probabilities" not in st.session_state:
    st.session_state.window_probabilities = None


model = load_fusion_model()
eeg_stats = load_eeg_stats()


# ============================================================
# FUNCTIONS
# ============================================================

def parse_header(uploaded_header):

    text = uploaded_header.read().decode(
        "utf-8",
        errors="ignore"
    )

    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]

    if not lines:
        raise ValueError("Invalid header file.")

    parts = lines[0].split()

    if len(parts) < 4:
        raise ValueError("Unable to read sampling frequency.")

    channels = int(parts[1])
    fs = float(parts[2])
    samples = int(parts[3])

    return channels, fs, samples


def load_eeg_mat(uploaded_mat):

    content = uploaded_mat.read()

    import io

    mat = loadmat(io.BytesIO(content))

    if "val" not in mat:
        raise ValueError(
            "MAT file does not contain expected variable 'val'."
        )

    eeg = np.asarray(mat["val"], dtype=np.float32)

    if eeg.ndim != 2:
        raise ValueError(
            "EEG data must be a 2-dimensional array."
        )

    return eeg


def preprocess_eeg(
    eeg,
    fs,
    channel_names=None
):

    eeg = np.asarray(eeg, dtype=np.float32)

    # --------------------------------------------------------
    # Channel handling
    # --------------------------------------------------------

    if channel_names is not None:

        channel_map = {
            str(name): i
            for i, name in enumerate(channel_names)
        }

        missing = [
            ch for ch in EEG_CHANNELS
            if ch not in channel_map
        ]

        if missing:
            raise ValueError(
                "Missing EEG channels: "
                + ", ".join(missing)
            )

        indices = [
            channel_map[ch]
            for ch in EEG_CHANNELS
        ]

        eeg = eeg[indices, :]

    else:

        if eeg.shape[0] < 19:
            raise ValueError(
                "EEG file has fewer than 19 channels."
            )

        eeg = eeg[:19, :]

    # --------------------------------------------------------
    # Remove patient/channel DC offset
    # --------------------------------------------------------

    eeg = eeg - np.mean(
        eeg,
        axis=1,
        keepdims=True
    )

    # --------------------------------------------------------
    # Resample to 100 Hz
    # --------------------------------------------------------

    if int(fs) != TARGET_FS:

        eeg = resample_poly(
            eeg,
            TARGET_FS,
            int(fs),
            axis=1
        ).astype(np.float32)

    # --------------------------------------------------------
    # Train-only normalization
    # --------------------------------------------------------

    if eeg_stats is not None:

        mean_col = None
        std_col = None

        for col in ["Mean", "mean", "Channel_Mean"]:
            if col in eeg_stats.columns:
                mean_col = col
                break

        for col in ["Std", "std", "Channel_Std"]:
            if col in eeg_stats.columns:
                std_col = col
                break

        if mean_col and std_col:

            means = []
            stds = []

            for ch in EEG_CHANNELS:

                if ch in eeg_stats.index:

                    means.append(
                        float(eeg_stats.loc[ch, mean_col])
                    )

                    stds.append(
                        float(eeg_stats.loc[ch, std_col])
                    )

                else:
                    means.append(0.0)
                    stds.append(1.0)

            means = np.asarray(
                means,
                dtype=np.float32
            ).reshape(-1, 1)

            stds = np.asarray(
                stds,
                dtype=np.float32
            ).reshape(-1, 1)

            stds[stds == 0] = 1.0

            eeg = (eeg - means) / stds

    # --------------------------------------------------------
    # 10-sec windows / 5-sec step
    # --------------------------------------------------------

    windows = []

    for start in range(
        0,
        eeg.shape[1] - WINDOW_SIZE + 1,
        STEP_SIZE
    ):

        window = eeg[
            :,
            start:start + WINDOW_SIZE
        ]

        if window.shape == (
            19,
            WINDOW_SIZE
        ):
            windows.append(window)

    if not windows:
        raise ValueError(
            "EEG recording is shorter than 10 seconds."
        )

    windows = np.asarray(
        windows,
        dtype=np.float32
    )

    # H5 format was channels x samples.
    # Model expects samples x channels.
    model_windows = np.transpose(
        windows,
        (0, 2, 1)
    )

    return eeg, model_windows


def normalize_iot(
    heart_rate,
    temperature,
    movement
):

    # Train-set patient/window-level IoT statistics
    means = np.array([
        82.842090,
        36.764591,
        0.480710
    ], dtype=np.float32)

    stds = np.array([
        20.291555,
        0.533917,
        0.298328
    ], dtype=np.float32)

    values = np.array([
        heart_rate,
        temperature,
        movement
    ], dtype=np.float32)

    stds[stds == 0] = 1.0

    return (
        (values - means) / stds
    ).astype(np.float32)

    # Train-set preprocessing values.
    # Missing values use train-set medians/modes.

def normalize_clinical(
    age,
    rosc,
    ttm,
    ohca,
    shockable,
    sex
):
    if rosc is None or not np.isfinite(rosc):
        rosc = 20.0

    if ttm is None or not np.isfinite(ttm):
        ttm = 33.0

    if ohca is None or not np.isfinite(ohca):
        ohca = 1.0

    if shockable is None or not np.isfinite(shockable):
        shockable = 1.0

    if sex is None or not np.isfinite(sex):
        sex = 0

    values = np.array([
        age,
        rosc,
        ttm,
        ohca,
        shockable,
        sex
    ], dtype=np.float32)

    means = np.array([
        59.128571,
        21.350000,
        33.428571,
        0.792857,
        0.571429,
        0.321429
    ], dtype=np.float32)

    stds = np.array([
        15.810862,
        11.915122,
        1.053551,
        0.406714,
        0.496649,
        0.468702
    ], dtype=np.float32)

    stds[stds == 0] = 1.0

    return (
        (values - means) / stds
    ).astype(np.float32)


def predict_fusion(
    eeg_windows,
    iot_features,
    clinical_features
):

    if model is None:
        raise RuntimeError(
            "Fusion model could not be loaded."
        )

    n = eeg_windows.shape[0]

    iot_batch = np.repeat(
        iot_features.reshape(1, -1),
        n,
        axis=0
    )

    clinical_batch = np.repeat(
        clinical_features.reshape(1, -1),
        n,
        axis=0
    )

    predictions = model.predict(
        [
            eeg_windows,
            iot_batch,
            clinical_batch
        ],
        verbose=0
    ).reshape(-1)

    probability = float(
        np.mean(predictions)
    )

    return probability, predictions


def status_from_probability(probability):

    if probability >= DECISION_THRESHOLD:
        return "POOR"

    return "GOOD"


def create_eeg_plot(eeg, channel_index=0):

    signal = eeg[channel_index]

    # Show a manageable segment.
    n = min(
        len(signal),
        TARGET_FS * 20
    )

    signal = signal[:n]

    time_axis = (
        np.arange(n) / TARGET_FS
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=signal,
            mode="lines",
            line=dict(
                width=1.2
            ),
            name=EEG_CHANNELS[channel_index]
        )
    )

    fig.update_layout(
        height=380,
        margin=dict(
            l=10,
            r=10,
            t=35,
            b=10
        ),
        title=(
            f"EEG • {EEG_CHANNELS[channel_index]} "
            "• First 20 Seconds"
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Time (seconds)",
        yaxis_title="Normalized amplitude",
        showlegend=False
    )

    return fig


def create_probability_plot(probabilities):

    x = np.arange(
        1,
        len(probabilities) + 1
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=probabilities,
            mode="lines+markers",
            line=dict(
                width=2
            ),
            name="AI probability"
        )
    )

    fig.add_hline(
        y=DECISION_THRESHOLD,
        line_dash="dash",
        annotation_text="Decision threshold"
    )

    fig.update_layout(
        height=340,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=10,
            r=10,
            t=35,
            b=10
        ),
        title="Window-level AI Probability",
        xaxis_title="10-second EEG window",
        yaxis_title="Poor-outcome probability",
        yaxis=dict(
            range=[0, 1]
        )
    )

    return fig


def create_monitor_plot():

    x = np.linspace(
        0,
        8,
        800
    )

    # DEMO ECG-style waveform.
    # This is NOT real patient ECG.
    ecg = (
        0.04 * np.sin(
            2 * np.pi * 1.2 * x
        )
    )

    for beat in np.arange(
        0.5,
        8,
        0.85
    ):

        ecg += (
            1.1 *
            np.exp(
                -((x - beat) ** 2)
                / 0.0015
            )
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=ecg,
            mode="lines",
            line=dict(
                width=2
            ),
            name="Demo monitor"
        )
    )

    fig.update_layout(
        height=260,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=5,
            r=5,
            t=30,
            b=5
        ),
        title="Live Monitor • DEMO SIGNAL",
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False
        )
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================


# ============================================================
# CENTRAL PATIENT SELECTION
# ============================================================

@st.cache_data
def load_all_dashboard_patient_ids():
    try:
        clinical_path = Path(
           "data/SmartComa_Clinical_200_Patients.csv"
        )
        df = pd.read_csv(clinical_path)

        ids = (
            df["Patient_ID"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(4)
            .tolist()
        )

        return sorted(ids)

    except Exception:
        return []


all_dashboard_patient_ids = load_all_dashboard_patient_ids()

if "selected_dashboard_patient" not in st.session_state:
    st.session_state.selected_dashboard_patient = (
        all_dashboard_patient_ids[0]
        if all_dashboard_patient_ids
        else "0299"
    )

with st.sidebar:
    st.markdown("### 👤 ACTIVE PATIENT")

    selected_dashboard_patient = st.selectbox(
        "Select Patient ID",
        all_dashboard_patient_ids,
        index=(
            all_dashboard_patient_ids.index(
                st.session_state.selected_dashboard_patient
            )
            if st.session_state.selected_dashboard_patient
            in all_dashboard_patient_ids
            else 0
        ),
        key="global_patient_selector"
    )

    st.session_state.selected_dashboard_patient = (
        selected_dashboard_patient
    )

    st.markdown(
        f"""
        <div style="
            margin-top:8px;
            padding:10px 12px;
            border-radius:10px;
            background:rgba(0,180,255,0.10);
            border:1px solid rgba(0,180,255,0.25);
        ">
        <div style="font-size:10px;color:#8ea4bd;">
        CURRENT PATIENT
        </div>
        <div style="
            font-size:22px;
            font-weight:700;
            color:#ffffff;
        ">
        {selected_dashboard_patient}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with st.sidebar:

    st.html("""
        <div style="
            font-size:26px;
            font-weight:800;
            margin-bottom:3px;
        ">
        🧠 SmartComa
        </div>

        <div style="
            color:#7890b5;
            font-size:11px;
            margin-bottom:25px;
        ">
        AI PATIENT MONITORING PLATFORM
        </div>
        """)

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "📂 Stored Patient Demo",
            "👤 Patient Analysis",
            "🧠 EEG Analysis",
            "📡 Live Monitoring",
            "🩺 Clinical Information",
            "🤖 AI Prediction",
            "🧑‍⚕️ Physiotherapy",
            "🔬 About Project"
        ]
    )

    st.divider()

    model_status = (
        "ONLINE"
        if model is not None
        else "MODEL ERROR"
    )

    st.html(
        f"""
        <div class="card">
            <div class="card-title">AI ENGINE</div>
            <div class="card-value"
                 style="font-size:21px;">
                 ● {model_status}
            </div>
            <div class="card-small">
                Fusion V1 • EEG + IoT + Clinical
            </div>
        </div>
        """
    )



# ============================================================
# STORED 200-PATIENT DEMO DATA
# ============================================================

STORED_IOT_CSV = "data/synthetic_iot_200_patients.csv"
STORED_HEA_ZIP = "/content/drive/MyDrive/SmartComa_EEG/EEG_200_Metadata.zip"
STORED_CLINICAL_CSV = "data/SmartComa_Clinical_200_Patients.csv"
STORED_MASTER_CSV = "data/SmartComa_Master_200_Patient_Map.csv"

@st.cache_data(show_spinner=False)
def load_stored_patient_ids():
    master = pd.read_csv(STORED_MASTER_CSV)
    ids = master["Patient_ID"].astype(str).str.zfill(4).tolist()
    return sorted(ids)

@st.cache_data(show_spinner=False)
def load_stored_clinical():
    df = pd.read_csv(STORED_CLINICAL_CSV)
    df["Patient_ID"] = df["Patient_ID"].astype(str).str.zfill(4)
    return df


@st.cache_data(show_spinner=False)
def load_stored_iot():
    df = pd.read_csv(STORED_IOT_CSV)
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(4)

    agg = (
        df.groupby("patient_id")
        .agg(
            heart_rate_mean=("heart_rate", "mean"),
            body_temperature_mean=("body_temperature", "mean"),
            movement_mean=("movement", "mean")
        )
        .reset_index()
    )

    return agg

HF_REPO_ID = "aadhil57786/SmartComa-EEG-200"
HF_REPO_TYPE = "dataset"


@st.cache_data(show_spinner=False)
def get_hf_eeg_files():
    api = HfApi(token=st.secrets["HF_TOKEN"])
    files = api.list_repo_files(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE
    )
    return files

HF_EEG_ZIP = "EEG_200_Patients.zip"
HF_HEA_ZIP = "EEG_200_Metadata.zip"


@st.cache_data(show_spinner=False)
def get_hf_zip(zip_name):
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=zip_name,
        token=st.secrets["HF_TOKEN"]
    )


@st.cache_data(show_spinner=False)
def find_hf_eeg_file(patient_id, extension):
    patient_id = str(patient_id).zfill(4)
    extension = extension.lower()

    zip_name = HF_EEG_ZIP if extension == ".mat" else HF_HEA_ZIP
    zip_path = get_hf_zip(zip_name)

    with zipfile.ZipFile(zip_path, "r") as z:
        matches = [
            name for name in z.namelist()
            if name.lower().endswith(extension)
            and patient_id in Path(name).name
        ]

        if not matches:
            raise FileNotFoundError(
                f"No {extension} EEG file found for patient {patient_id} "
                f"inside {zip_name}"
            )

        file_name = matches[0]
        file_bytes = z.read(file_name)

    return file_bytes, Path(file_name).name

@st.cache_data(show_spinner=False)
def load_stored_eeg(patient_id):
    patient_id = str(patient_id).zfill(4)

    raw_bytes, mat_name = find_hf_eeg_file(patient_id, ".mat")
    hea_bytes, hea_name = find_hf_eeg_file(patient_id, ".hea")

    return raw_bytes, hea_bytes, mat_name, hea_name
# ============================================================
# DASHBOARD
# ============================================================



# ============================================================
# AUTOMATIC ACTIVE-PATIENT DATA LOADER
# ============================================================

def auto_load_active_patient():

    patient_id = st.session_state.get(
        "selected_dashboard_patient",
        "0299"
    )

    # Already loaded for this patient
    if st.session_state.get(
        "_auto_loaded_patient"
    ) == patient_id:
        return

    try:

        # ----------------------------------------------------
        # LOAD STORED CLINICAL + IoT DATA
        # ----------------------------------------------------

        clinical_df = load_stored_clinical()
        iot_df = load_stored_iot()

        clinical_df["Patient_ID"] = (
            clinical_df["Patient_ID"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(4)
        )

        # IoT helper returns patient_id
        iot_df["patient_id"] = (
            iot_df["patient_id"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(4)
        )

        c_match = clinical_df[
            clinical_df["Patient_ID"] == patient_id
        ]

        i_match = iot_df[
            iot_df["patient_id"] == patient_id
        ]

        if c_match.empty:
            raise ValueError(
                f"Clinical data not found for {patient_id}"
            )

        if i_match.empty:
            raise ValueError(
                f"IoT data not found for {patient_id}"
            )

        c_row = c_match.iloc[0]
        i_row = i_match.iloc[0]

        # ----------------------------------------------------
        # CLINICAL VALUES
        # ----------------------------------------------------

        clinical_values = {
            "Age": float(c_row["Age"])
            if pd.notna(c_row["Age"]) else np.nan,

            "ROSC": float(c_row["ROSC"])
            if pd.notna(c_row["ROSC"]) else np.nan,

            "TTM": float(c_row["TTM"])
            if pd.notna(c_row["TTM"]) else np.nan,

            "OHCA": float(c_row["OHCA"])
            if pd.notna(c_row["OHCA"]) else np.nan,

            "Shockable_Rhythm":
                float(c_row["Shockable_Rhythm"])
                if pd.notna(c_row["Shockable_Rhythm"])
                else np.nan,

            "Sex_Encoded":
                float(c_row["Sex_Encoded"])
                if "Sex_Encoded" in c_row.index
                and pd.notna(c_row["Sex_Encoded"])
                else (
                    1.0
                    if str(c_row.get("Sex", "")).strip().lower()
                    == "female"
                    else 0.0
                )
        }

        # ----------------------------------------------------
        # IoT VALUES
        # ----------------------------------------------------

        iot_values = {
            "heart_rate_mean":
                float(i_row["heart_rate_mean"]),

            "body_temperature_mean":
                float(i_row["body_temperature_mean"]),

            "movement_mean":
                float(i_row["movement_mean"])
        }

        # ----------------------------------------------------
        # LOAD EEG
        # ----------------------------------------------------

        mat_bytes, hea_bytes, mat_name, hea_name = (
            load_stored_eeg(patient_id)
        )

        import tempfile
        from scipy.io import loadmat

        with tempfile.TemporaryDirectory() as tmp:

            mat_path = Path(tmp) / Path(mat_name).name
            hea_path = Path(tmp) / Path(hea_name).name

            mat_path.write_bytes(mat_bytes)
            hea_path.write_bytes(hea_bytes)

            mat_data = loadmat(str(mat_path))

            if "val" not in mat_data:
                raise ValueError(
                    f"EEG variable 'val' not found for {patient_id}"
                )

            eeg_array = mat_data["val"]

            # -----------------------------------------------
            # READ HEADER
            # -----------------------------------------------

            header_text = hea_bytes.decode(
                "utf-8",
                errors="ignore"
            )

            header_lines = [
                x.strip()
                for x in header_text.splitlines()
                if x.strip()
            ]

            first_line = header_lines[0].split()

            fs = float(first_line[2])

            channel_names = []

            for line in header_lines[1:]:

                parts = line.split()

                if len(parts) >= 9:

                    ch = parts[8]

                    if ch.startswith("#"):
                        ch = ch[1:]

                    channel_names.append(ch)

            # -----------------------------------------------
            # SAME EEG PREPROCESSING
            # -----------------------------------------------

            processed_eeg, eeg_windows = preprocess_eeg(
                eeg_array,
                fs,
                channel_names
            )

        if eeg_windows is None:
            raise ValueError(
                f"No complete EEG windows generated for {patient_id}"
            )

        eeg_windows = np.asarray(
            eeg_windows,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # NORMALIZE CLINICAL + IoT
        # ----------------------------------------------------

        clinical_norm = normalize_clinical(
            clinical_values["Age"],
            clinical_values["ROSC"],
            clinical_values["TTM"],
            clinical_values["OHCA"],
            clinical_values["Shockable_Rhythm"],
            clinical_values["Sex_Encoded"]
        )

        iot_norm = normalize_iot(
            iot_values["heart_rate_mean"],
            iot_values["body_temperature_mean"],
            iot_values["movement_mean"]
        )

        clinical_norm = np.asarray(
            clinical_norm,
            dtype=np.float32
        ).reshape(1, -1)

        iot_norm = np.asarray(
            iot_norm,
            dtype=np.float32
        ).reshape(1, -1)

        # ----------------------------------------------------
        # REPEAT PATIENT FEATURES FOR EACH EEG WINDOW
        # ----------------------------------------------------

        n_windows = len(eeg_windows)

        clinical_batch = np.repeat(
            clinical_norm,
            n_windows,
            axis=0
        )

        iot_batch = np.repeat(
            iot_norm,
            n_windows,
            axis=0
        )

        # ----------------------------------------------------
        # FUSION V1 PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            [
                eeg_windows,
                iot_batch,
                clinical_batch
            ],
            verbose=0
        ).reshape(-1)

        patient_probability = float(
            np.mean(predictions)
        )

        threshold = float(
            DECISION_THRESHOLD
            if "DECISION_THRESHOLD" in globals()
            else 0.68
        )

        prediction_label = int(
            patient_probability >= threshold
        )

        prediction_status = (
            "POOR"
            if prediction_label == 1
            else "GOOD"
        )

        # ----------------------------------------------------
        # STORE EVERYTHING IN SESSION STATE
        # ----------------------------------------------------

        st.session_state.eeg_data = processed_eeg

        st.session_state.eeg_windows = eeg_windows

        st.session_state.window_probabilities = predictions

        st.session_state.prediction_probability = (
            patient_probability
        )

        st.session_state.prediction_status = (
            prediction_status
        )

        st.session_state.prediction_label = (
            prediction_label
        )

        st.session_state.active_patient_clinical = (
            clinical_values
        )

        st.session_state.active_patient_iot = (
            iot_values
        )

        st.session_state.active_patient_id = (
            patient_id
        )

        st.session_state._auto_loaded_patient = (
            patient_id
        )

        st.session_state._auto_patient_error = None

    except Exception as e:

        import traceback

        error_text = (
            f"{type(e).__name__}: {str(e)}"
        )

        st.session_state._auto_patient_error = error_text
        st.session_state._auto_loaded_patient = None

        print("=" * 80)
        print("ACTIVE PATIENT LOADER ERROR")
        print("=" * 80)
        print("Patient:", patient_id)
        print("Error  :", error_text)
        print()
        traceback.print_exc()
        print("=" * 80)



# ============================================================
# ACTIVE PATIENT CONTEXT
# ============================================================

ACTIVE_PATIENT_ID = st.session_state.get(
    "selected_dashboard_patient",
    "0299"
)

st.markdown(
    f"""
    <div style="
        margin:8px 0 18px 0;
        padding:10px 16px;
        border-radius:10px;
        background:linear-gradient(
            90deg,
            rgba(0,180,255,0.12),
            rgba(120,70,255,0.10)
        );
        border:1px solid rgba(0,180,255,0.22);
    ">
        <span style="
            color:#8ea4bd;
            font-size:11px;
        ">
        ACTIVE PATIENT
        </span>
        <span style="
            color:#ffffff;
            font-size:18px;
            font-weight:700;
            margin-left:12px;
        ">
        {ACTIVE_PATIENT_ID}
        </span>
    </div>
    """,
    unsafe_allow_html=True
)



# ============================================================
# RUN AUTOMATIC ACTIVE-PATIENT LOADER
# ============================================================

auto_load_active_patient()

if st.session_state.get("_auto_patient_error"):
    st.warning(
        "Active patient data loading issue: "
        + str(st.session_state.get("_auto_patient_error"))
    )


if page == "🏠 Dashboard":

    st.html("""
        <div class="topbar">
            <div class="main-title">
                SmartComa AI Monitoring Center
            </div>
            <div class="subtitle">
                Multimodal clinical decision-support dashboard
                for coma patient monitoring
            </div>
        </div>
        """)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.html("""
            <div class="card">
                <div class="card-title">Patient</div>
                <div class="card-value">0299</div>
                <div class="card-small">
                    Active demonstration case
                </div>
            </div>
            """)

    with c2:
        st.html("""
            <div class="card">
                <div class="card-title">Heart Rate</div>
                <div class="card-value">76 BPM</div>
                <div class="card-small">
                    IoT demonstration data
                </div>
            </div>
            """)

    with c3:
        st.html("""
            <div class="card">
                <div class="card-title">Temperature</div>
                <div class="card-value">36.56 °C</div>
                <div class="card-small">
                    IoT demonstration data
                </div>
            </div>
            """)

    with c4:
        st.html("""
            <div class="card">
                <div class="card-title">AI Status</div>
                <div class="card-value good">
                    GOOD
                </div>
                <div class="card-small">
                    Demo probability 33.96%
                </div>
            </div>
            """)

    st.html(
        '<div class="section-title">System Overview</div>'
    )

    left, right = st.columns(
        [1.45, 1]
    )

    with left:

        st.plotly_chart(
            create_monitor_plot(),
            use_container_width=True
        )

        st.html("""
            <div class="disclaimer">
            <b>DEMO MONITOR:</b>
            The waveform above is a simulated ECG-style
            visualization for interface demonstration.
            It is not a real patient ECG measurement.
            </div>
            """)

    with right:

        st.html("""
            <div class="ai-panel">
                <div class="card-title">
                    MULTIMODAL AI
                </div>

                <div style="
                    font-size:40px;
                    font-weight:800;
                    margin-top:8px;
                ">
                    33.96%
                </div>

                <div class="good"
                     style="font-size:18px;">
                    GOOD OUTCOME CLASS
                </div>

                <br>

                <span class="info-pill">
                    Threshold: 68%
                </span>

                <span class="info-pill">
                    Fusion V1
                </span>

                <br><br>

                EEG + IoT + Clinical information
                are integrated before prediction.
            </div>
            """)

    st.html(
        '<div class="section-title">AI Processing Pipeline</div>'
    )

    st.html("""
        <div class="pipeline">

            <div class="pipeline-item">
                📥 EEG
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                🧹 Preprocessing
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                🧠 EEG Encoder
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                📡 IoT Features
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                🩺 Clinical Features
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                🤖 Fusion AI
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                📊 Patient Result
            </div>

        </div>
        """)


# ============================================================
# PATIENT ANALYSIS
# ============================================================


elif page == "📂 Stored Patient Demo":

    st.html("""
        <div class="topbar">
            <div class="main-title">
                Stored Patient Demo
            </div>
            <div class="subtitle">
                Select a stored patient and automatically load
                EEG + IoT + Clinical information
            </div>
        </div>
    """)

    st.html("""
        <div class="warning-box">
        ⚠️ These are existing dataset patients used for
        academic demonstration and model evaluation.
        They are not unseen new patients.
        </div>
    """)

    patient_ids = load_stored_patient_ids()

    selected_patient = st.selectbox(
        "Select Patient ID",
        patient_ids,
        index=patient_ids.index("0299") if "0299" in patient_ids else 0
    )

    clinical_df = load_stored_clinical()
    iot_df = load_stored_iot()

    c_row = clinical_df[
        clinical_df["Patient_ID"] == selected_patient
    ]

    i_row = iot_df[
        iot_df["patient_id"] == selected_patient
    ]

    if len(c_row) == 0 or len(i_row) == 0:
        st.error("Stored clinical / IoT data not found.")
        st.stop()

    c = c_row.iloc[0]
    i = i_row.iloc[0]

    st.html("""
        <div class="section-title">
            Automatically Loaded Patient Information
        </div>
    """)

    a, b, ccol, d = st.columns(4)

    with a:
        st.metric("Patient ID", selected_patient)

    with b:
        st.metric("Age", str(c["Age"]))

    with ccol:
        st.metric("Heart Rate", f"{i['heart_rate_mean']:.2f} BPM")

    with d:
        st.metric(
            "Temperature",
            f"{i['body_temperature_mean']:.2f} °C"
        )

    st.write("")

    e, f, g = st.columns(3)

    with e:
        st.metric("Movement", f"{i['movement_mean']:.3f}")

    with f:
        st.metric("OHCA", str(c["OHCA"]))

    with g:
        st.metric("TTM", str(c["TTM"]))

    st.write("")

    st.html("""
        <div class="section-title">
            EEG + IoT + Clinical → Fusion V1
        </div>
    """)

    if st.button(
        "🚀 ANALYZE STORED PATIENT",
        use_container_width=True,
        type="primary"
    ):

        try:
            with st.spinner(
                "Loading stored EEG and running Fusion V1..."
            ):

                mat_bytes, hea_bytes, mat_name, hea_name = (
                    load_stored_eeg(selected_patient)
                )

                # ------------------------------------------------
                # Save temporarily so the existing preprocessing
                # pipeline can be reused.
                # ------------------------------------------------
                import tempfile

                with tempfile.TemporaryDirectory() as tmp:

                    mat_path = Path(tmp) / Path(mat_name).name
                    hea_path = Path(tmp) / Path(hea_name).name

                    mat_path.write_bytes(mat_bytes)
                    hea_path.write_bytes(hea_bytes)

                    # ------------------------------------------------
                    # Load MAT EEG data
                    # preprocess_eeg() expects:
                    #   EEG array + sampling rate + channel names
                    # ------------------------------------------------
                    mat_data = loadmat(str(mat_path))

                    if "val" not in mat_data:
                        raise ValueError(
                            "MAT file does not contain EEG variable 'val'."
                        )

                    raw_eeg = mat_data["val"]

                    # ------------------------------------------------
                    # Read sampling rate + channel names from HEA
                    # ------------------------------------------------
                    hea_text = hea_path.read_text(
                        encoding="utf-8",
                        errors="ignore"
                    )

                    hea_lines = [
                        line.strip()
                        for line in hea_text.splitlines()
                        if line.strip()
                    ]

                    if not hea_lines:
                        raise ValueError("HEA file is empty.")

                    # First header line:
                    # record_name channels sampling_rate samples
                    header_parts = hea_lines[0].split()

                    if len(header_parts) < 4:
                        raise ValueError(
                            "Invalid HEA first line."
                        )

                    fs = float(header_parts[2])

                    # Channel names are normally the last token
                    # of each signal-definition line.
                    channel_names = []

                    for line in hea_lines[1:]:
                        parts = line.split()

                        if len(parts) >= 1:
                            channel_name = parts[-1]

                            # Avoid comment/header-like lines
                            if channel_name and not channel_name.startswith("#"):
                                channel_names.append(channel_name)

                    # Keep only the expected EEG channel count
                    if len(channel_names) < len(EEG_CHANNELS):
                        raise ValueError(
                            f"HEA contains only {len(channel_names)} "
                            f"channel names; expected at least "
                            f"{len(EEG_CHANNELS)}."
                        )

                    # ------------------------------------------------
                    # Existing dashboard preprocessing
                    # ------------------------------------------------
                    processed_eeg, eeg_windows = preprocess_eeg(
                        raw_eeg,
                        fs,
                        channel_names
                    )

                # Clinical values
                sex_value = str(c["Sex"])

                clinical_values = {
                    "Age": float(c["Age"]),
                    "ROSC": float(c["ROSC"])
                    if pd.notna(c["ROSC"]) else 20.0,
                    "TTM": float(c["TTM"])
                    if pd.notna(c["TTM"]) else 33.0,
                    "OHCA": float(c["OHCA"])
                    if pd.notna(c["OHCA"]) else 1.0,
                    "Shockable_Rhythm":
                        float(c["Shockable_Rhythm"])
                        if pd.notna(c["Shockable_Rhythm"])
                        else 1.0,
                    "Sex_Encoded":
                        0.0 if sex_value.lower() == "male" else 1.0
                }

                iot_values = {
                    "heart_rate_mean":
                        float(i["heart_rate_mean"]),
                    "body_temperature_mean":
                        float(i["body_temperature_mean"]),
                    "movement_mean":
                        float(i["movement_mean"])
                }

                # Existing normalization
                clinical_norm = normalize_clinical(
                    clinical_values["Age"],
                    clinical_values["ROSC"],
                    clinical_values["TTM"],
                    clinical_values["OHCA"],
                    clinical_values["Shockable_Rhythm"],
                    clinical_values["Sex_Encoded"]
                )

                iot_norm = normalize_iot(
                    iot_values["heart_rate_mean"],
                    iot_values["body_temperature_mean"],
                    iot_values["movement_mean"]
                )

                # ------------------------------------------------
                # Fusion prediction
                # ------------------------------------------------
                probs = []

                for window in eeg_windows:

                    eeg_input = np.asarray(
                        window,
                        dtype=np.float32
                    )

                    if eeg_input.ndim == 2:
                        if eeg_input.shape == (19, 1000):
                            eeg_input = eeg_input.T

                    eeg_input = np.expand_dims(
                        eeg_input,
                        axis=0
                    )

                    iot_input = np.asarray(
                        iot_norm,
                        dtype=np.float32
                    ).reshape(1, -1)

                    clinical_input = np.asarray(
                        clinical_norm,
                        dtype=np.float32
                    ).reshape(1, -1)

                    pred = model.predict(
                        {
                            "eeg_input": eeg_input,
                            "iot_input": iot_input,
                            "clinical_input": clinical_input
                        },
                        verbose=0
                    )[0][0]

                    probs.append(float(pred))

                probability = float(np.mean(probs))

                status = (
                    "POOR"
                    if probability >= DECISION_THRESHOLD
                    else "GOOD"
                )

                st.session_state["prediction"] = probability
                st.session_state["patient_id"] = selected_patient
                st.session_state["patient_status"] = status
                st.session_state["window_probabilities"] = probs

                st.success(
                    f"Patient {selected_patient} analyzed successfully."
                )

                st.metric(
                    "AI Probability",
                    f"{probability * 100:.2f}%"
                )

                st.metric(
                    "AI Status",
                    status
                )

                st.caption(
                    f"Decision threshold: "
                    f"{DECISION_THRESHOLD:.2f}"
                )

        except Exception as e:
            st.error(
                f"Stored patient analysis failed: {e}"
            )


elif page == "👤 Patient Analysis":

    st.html("""
        <div class="topbar">
            <div class="main-title">
                Patient Analysis
            </div>
            <div class="subtitle">
                Upload patient EEG and enter corresponding
                IoT and clinical information.
            </div>
        </div>
        """)

    eeg_file = st.file_uploader(
        "Upload EEG MAT file",
        type=["mat"]
    )

    hea_file = st.file_uploader(
        "Upload EEG Header file",
        type=["hea"]
    )

    st.html(
        '<div class="section-title">Patient Information</div>'
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        patient_id = st.text_input(
            "Patient ID",
            value="NEW_PATIENT"
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45
        )

        sex = st.selectbox(
            "Sex",
            [
                "Male",
                "Female"
            ]
        )

    with c2:

        rosc_option = st.selectbox(
            "ROSC time available?",
            [
                "Yes",
                "No"
            ]
        )

        if rosc_option == "Yes":

            rosc = st.number_input(
                "ROSC time",
                min_value=0.0,
                value=20.0
            )

        else:
            rosc = None

        ttm = st.number_input(
            "TTM",
            min_value=25.0,
            max_value=40.0,
            value=33.0
        )

    with c3:

        ohca = st.selectbox(
            "OHCA",
            [
                "Yes",
                "No"
            ]
        )

        shockable = st.selectbox(
            "Shockable Rhythm",
            [
                "Yes",
                "No"
            ]
        )

    st.html(
        '<div class="section-title">IoT Measurements</div>'
    )

    i1, i2, i3 = st.columns(3)

    with i1:
        heart_rate = st.number_input(
            "Heart Rate (BPM)",
            min_value=30.0,
            max_value=180.0,
            value=76.0
        )

    with i2:
        temperature = st.number_input(
            "Body Temperature (°C)",
            min_value=30.0,
            max_value=42.0,
            value=36.5
        )

    with i3:
        movement = st.number_input(
            "Movement Index",
            min_value=0.0,
            max_value=1.0,
            value=0.42
        )

    st.html("<br>")

    analyze = st.button(
        "🚀 ANALYZE PATIENT",
        type="primary",
        use_container_width=True
    )

    if analyze:

        if eeg_file is None:

            st.error(
                "Please upload an EEG .mat file."
            )

        elif hea_file is None:

            st.error(
                "Please upload the matching .hea file."
            )

        elif model is None:

            st.error(
                "Fusion model could not be loaded."
            )

        else:

            try:

                channels, fs, samples = parse_header(
                    hea_file
                )

                raw_eeg = load_eeg_mat(
                    eeg_file
                )

                processed_eeg, windows = preprocess_eeg(
                    raw_eeg,
                    fs
                )

                sex_encoded = (
                    0
                    if sex == "Male"
                    else 1
                )

                clinical = normalize_clinical(
                    age,
                    rosc,
                    ttm,
                    1 if ohca == "Yes" else 0,
                    1 if shockable == "Yes" else 0,
                    sex_encoded
                )

                iot = normalize_iot(
                    heart_rate,
                    temperature,
                    movement
                )

                probability, window_probs = predict_fusion(
                    windows,
                    iot,
                    clinical
                )

                status = status_from_probability(
                    probability
                )

                st.session_state.prediction_done = True
                st.session_state.prediction_probability = probability
                st.session_state.prediction_status = status
                st.session_state.patient_id = patient_id
                st.session_state.eeg_data = processed_eeg
                st.session_state.window_probabilities = window_probs

                st.success(
                    "Patient analysis completed successfully."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Analysis failed: {str(e)}"
                )


# ============================================================
# EEG ANALYSIS
# ============================================================

elif page == "🧠 EEG Analysis":

    st.html("""
        <div class="topbar">
            <div class="main-title">
                EEG Analysis
            </div>
            <div class="subtitle">
                19-channel EEG preprocessing and visualization
            </div>
        </div>
        """)

    if st.session_state.eeg_data is None:

        st.info(
            "Analyze a patient first to display EEG results."
        )

    else:

        eeg = st.session_state.eeg_data

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Channels",
                "19"
            )

        with c2:
            st.metric(
                "Sampling Rate",
                "100 Hz"
            )

        with c3:
            st.metric(
                "Window",
                "10 sec"
            )

        channel = st.selectbox(
            "Select EEG Channel",
            EEG_CHANNELS
        )

        channel_index = EEG_CHANNELS.index(
            channel
        )

        st.plotly_chart(
            create_eeg_plot(
                eeg,
                channel_index
            ),
            use_container_width=True
        )

        st.html("""
            <div class="info-pill">
                10-second windows
            </div>

            <div class="info-pill">
                5-second step
            </div>

            <div class="info-pill">
                50% overlap
            </div>

            <div class="info-pill">
                Train-only normalization
            </div>
            """)


# ============================================================
# LIVE MONITORING
# ============================================================

elif page == "📡 Live Monitoring":

    st.html("""
        <div class="topbar">
            <div class="main-title">
                Live Patient Monitoring
            </div>
            <div class="subtitle">
                Real-time style monitoring interface
                for prototype demonstration
            </div>
        </div>
        """)

    c1, c2, c3, c4 = st.columns(4)

 patient_id = st.session_state.get(
    "active_patient_id",
    st.session_state.get(
        "selected_dashboard_patient",
        "0299"
    )
)

patient_iot = st.session_state.get(
    "active_patient_iot",
    {}
) or {}

heart_rate = patient_iot.get(
    "heart_rate_mean",
    None
)

temperature = patient_iot.get(
    "body_temperature_mean",
    None
)

movement = patient_iot.get(
    "movement_mean",
    None
)

metrics = [
    (
        "❤️ Heart Rate",
        f"{heart_rate:.2f} BPM"
        if heart_rate is not None
        else "N/A",
        "Stored IoT data"
    ),
    (
        "🌡️ Temperature",
        f"{temperature:.2f} °C"
        if temperature is not None
        else "N/A",
        "Stored IoT data"
    ),
    (
        "📡 Movement",
        f"{movement:.3f}"
        if movement is not None
        else "N/A",
        "Stored IoT data"
    ),
    (
        "🧠 EEG",
        "ACTIVE",
        "19 channels"
    )
]

    for col, data in zip(
        [c1, c2, c3, c4],
        metrics
    ):

        with col:

            st.html(
                f"""
                <div class="card">
                    <div class="card-title">
                        {data[0]}
                    </div>
                    <div class="card-value">
                        {data[1]}
                    </div>
                    <div class="card-small">
                        {data[2]}
                    </div>
                </div>
                """
            )

    st.plotly_chart(
        create_monitor_plot(),
        use_container_width=True
    )

    st.html("""
        <div class="warning-box">
        ⚠️ This live-monitoring page is a prototype
        visualization. IoT values shown here are
        demonstration values unless connected to
        the actual sensor system.
        </div>
        """)


# ============================================================
# CLINICAL INFORMATION
# ============================================================

elif page == "🩺 Clinical Information":

    st.html("""
        <div class="topbar">
            <div class="main-title">
                Clinical Information
            </div>
            <div class="subtitle">
                Patient factors used by the multimodal AI model
            </div>
        </div>
        """)

    c1, c2, c3 = st.columns(3)

    with c1:

        st.html("""
            <div class="card">
                <div class="card-title">Age</div>
                <div class="card-value">45</div>
                <div class="card-small">
                    Patient demographic
                </div>
            </div>
            """)

    with c2:

        st.html("""
            <div class="card">
                <div class="card-title">OHCA</div>
                <div class="card-value">YES</div>
                <div class="card-small">
                    Out-of-hospital cardiac arrest
                </div>
            </div>
            """)

    with c3:

        st.html("""
            <div class="card">
                <div class="card-title">Shockable Rhythm</div>
                <div class="card-value">YES</div>
                <div class="card-small">
                    Clinical feature
                </div>
            </div>
            """)

    st.html(
        '<div class="section-title">Clinical Features</div>'
    )

    clinical_table = pd.DataFrame({
        "Feature": [
            "Age",
            "Sex",
            "ROSC",
            "OHCA",
            "Shockable Rhythm",
            "TTM"
        ],
        "Value": [
            "45 years",
            "Male",
            "20.0",
            "Yes",
            "Yes",
            "33 °C"
        ],
        "AI Role": [
            "Demographic",
            "Demographic",
            "Resuscitation",
            "Clinical",
            "Clinical",
            "Temperature management"
        ]
    })

    st.dataframe(
        clinical_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# AI PREDICTION
# ============================================================

elif page == "🤖 AI Prediction":

    st.html("""
        <div class="topbar">
            <div class="main-title">
                AI Prediction Center
            </div>
            <div class="subtitle">
                Multimodal Fusion V1 prediction and
                explainable monitoring view
            </div>
        </div>
        """)

    probability = st.session_state.prediction_probability
    status = st.session_state.prediction_status

    left, right = st.columns(
        [1, 1.5]
    )

    with left:

        st.html(
            f"""
            <div class="ai-panel">

                <div class="card-title">
                    PATIENT
                </div>

                <div style="
                    font-size:26px;
                    font-weight:800;
                    margin-top:8px;
                ">
                    {st.session_state.patient_id}
                </div>

                <br>

                <div class="card-title">
                    AI PROBABILITY
                </div>

                <div style="
                    font-size:52px;
                    font-weight:800;
                ">
                    {probability * 100:.2f}%
                </div>

                <div class="
                    {'poor' if status == 'POOR' else 'good'}
                    "
                    style="font-size:20px;">
                    {status}
                </div>

                <br>

                <span class="info-pill">
                    Threshold {DECISION_THRESHOLD:.2f}
                </span>

            </div>
            """
        )

    with right:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={
                    "suffix": "%"
                },
                title={
                    "text":
                    "Poor-outcome probability"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "threshold": {
                        "line": {
                            "width": 4
                        },
                        "value":
                        DECISION_THRESHOLD * 100
                    }
                }
            )
        )

        fig.update_layout(
            height=320,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if st.session_state.window_probabilities is not None:

        st.html(
            '<div class="section-title">'
            'Window-level Analysis'
            '</div>'
        )

        st.plotly_chart(
            create_probability_plot(
                st.session_state.window_probabilities
            ),
            use_container_width=True
        )

    st.html("""
        <div class="disclaimer">
        <b>Clinical decision-support notice:</b>
        The AI output is an experimental research result
        and must not be treated as an autonomous medical
        diagnosis or treatment decision. Final interpretation
        must be performed by qualified healthcare professionals.
        </div>
        """)


# ============================================================
# PHYSIOTHERAPY
# ============================================================

elif page == "🧑‍⚕️ Physiotherapy":

    # ========================================================
    # ACTIVE PATIENT DATA
    # ========================================================

    patient_id = st.session_state.get(
        "active_patient_id",
        st.session_state.get(
            "selected_dashboard_patient",
            "0299"
        )
    )

    prediction_probability = st.session_state.get(
        "prediction_probability",
        None
    )

    prediction_status = st.session_state.get(
        "prediction_status",
        None
    )

    patient_iot = st.session_state.get(
        "active_patient_iot",
        {}
    ) or {}

    patient_clinical = st.session_state.get(
        "active_patient_clinical",
        {}
    ) or {}

    movement_value = patient_iot.get(
        "movement_mean",
        None
    )

    heart_rate = patient_iot.get(
        "heart_rate_mean",
        None
    )

    temperature = patient_iot.get(
        "body_temperature_mean",
        None
    )

    age = patient_clinical.get(
        "Age",
        None
    )

    # ========================================================
    # HEADER
    # ========================================================

    st.html("""
        <div class="topbar">
            <div class="main-title">
                Physiotherapy Support
            </div>
            <div class="subtitle">
                Patient-specific AI-assisted rehabilitation
                monitoring for clinician review
            </div>
        </div>
    """)

    st.html(f"""
        <div class="warning-box">
        ⚠️ This module provides prototype decision-support
        information for clinician review. It does not provide
        autonomous medical prescriptions or treatment orders.
        <br><br>
        <b>Active Patient:</b> {patient_id}
        </div>
    """)

    # ========================================================
    # PATIENT AI SUMMARY
    # ========================================================

    st.html(
        '<div class="section-title">'
        'Patient Rehabilitation Context'
        '</div>'
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if prediction_probability is not None:
            st.metric(
                "AI Probability",
                f"{prediction_probability * 100:.1f}%"
            )
        else:
            st.metric("AI Probability", "N/A")

    with c2:
        st.metric(
            "AI Status",
            prediction_status if prediction_status else "N/A"
        )

    with c3:
        if movement_value is not None:
            st.metric(
                "Movement",
                f"{movement_value:.2f}"
            )
        else:
            st.metric("Movement", "N/A")

    with c4:
        if heart_rate is not None:
            st.metric(
                "Heart Rate",
                f"{heart_rate:.1f} BPM"
            )
        else:
            st.metric("Heart Rate", "N/A")

    st.html("<br>")

    # ========================================================
    # PATIENT CLINICAL / MONITORING CONTEXT
    # ========================================================

    p1, p2 = st.columns(2)

    with p1:

        st.html("""
            <div class="card">
                <div class="card-title">
                    PATIENT CONTEXT
                </div>
        """)

        if age is not None:
            st.write(f"**Age:** {age:.0f} years")

        if temperature is not None:
            st.write(
                f"**Body Temperature:** "
                f"{temperature:.2f} °C"
            )

        if patient_clinical.get("ROSC") is not None:
            st.write(
                f"**ROSC:** "
                f"{patient_clinical.get('ROSC')}"
            )

        if patient_clinical.get("OHCA") is not None:
            st.write(
                f"**OHCA:** "
                f"{patient_clinical.get('OHCA')}"
            )

        if patient_clinical.get("Shockable_Rhythm") is not None:
            st.write(
                f"**Shockable Rhythm:** "
                f"{patient_clinical.get('Shockable_Rhythm')}"
            )

        st.html("</div>")

    with p2:

        st.html("""
            <div class="card">
                <div class="card-title">
                    MOVEMENT MONITORING
                </div>
        """)

        if movement_value is not None:

            if movement_value < 0.30:
                movement_note = (
                    "Low movement level detected in the "
                    "stored monitoring data."
                )
            elif movement_value < 0.70:
                movement_note = (
                    "Moderate movement level observed "
                    "in the stored monitoring data."
                )
            else:
                movement_note = (
                    "Higher movement level observed "
                    "in the stored monitoring data."
                )

            st.write(
                f"**Movement Mean:** "
                f"{movement_value:.3f}"
            )

            st.write(movement_note)

        else:
            st.write(
                "Movement data is not currently available."
            )

        st.html("</div>")

    st.html("<br>")

    # ========================================================
    # CLINICIAN REVIEW PROTOCOL
    # ========================================================

    st.html(
        '<div class="section-title">'
        'Clinician Review Protocol'
        '</div>'
    )

    r1, r2 = st.columns(2)

    with r1:

        st.html("""
            <div class="card">
                <div class="card-title">
                    POSITIONING
                </div>

                <div style="
                    font-size:20px;
                    font-weight:800;
                    margin-top:8px;
                ">
                    Pressure-relief assessment
                </div>

                <div class="card-small">
                    Review patient positioning, pressure
                    points and bed positioning according to
                    the responsible clinical protocol.
                </div>
            </div>
        """)

        st.html("<br>")

        st.html("""
            <div class="card">
                <div class="card-title">
                    PASSIVE MOVEMENT
                </div>

                <div style="
                    font-size:20px;
                    font-weight:800;
                    margin-top:8px;
                ">
                    Range-of-motion assessment
                </div>

                <div class="card-small">
                    Clinician may assess passive range of
                    motion and determine whether appropriate
                    rehabilitation activity is indicated.
                </div>
            </div>
        """)

    with r2:

        st.html("""
            <div class="card">
                <div class="card-title">
                    MOVEMENT RESPONSE
                </div>

                <div style="
                    font-size:20px;
                    font-weight:800;
                    margin-top:8px;
                ">
                    Monitor movement behaviour
                </div>

                <div class="card-small">
                    Compare movement measurements across
                    monitoring windows and review changes
                    with the patient's clinical condition.
                </div>
            </div>
        """)

        st.html("<br>")

        # Dynamic escalation card
        if (
            prediction_status == "POOR"
            or (
                prediction_probability is not None
                and prediction_probability >= 0.68
            )
        ):

            escalation_title = (
                "Enhanced clinician review"
            )

            escalation_text = (
                "AI output indicates a higher-risk classification "
                "according to the frozen project threshold. "
                "The responsible clinical team should review "
                "the patient before rehabilitation decisions."
            )

        else:

            escalation_title = (
                "Routine clinician review"
            )

            escalation_text = (
                "AI output is below the project decision threshold. "
                "Continue clinical assessment and monitoring before "
                "making rehabilitation decisions."
            )

        st.html(f"""
            <div class="card">
                <div class="card-title">
                    CLINICAL ESCALATION
                </div>

                <div style="
                    font-size:20px;
                    font-weight:800;
                    margin-top:8px;
                ">
                    {escalation_title}
                </div>

                <div class="card-small">
                    {escalation_text}
                </div>
            </div>
        """)

    st.html("<br>")

    # ========================================================
    # IMPORTANT PROJECT DISCLAIMER
    # ========================================================

    st.html("""
        <div class="warning-box">
        🧑‍⚕️ <b>Clinician Review Required</b><br><br>

        The physiotherapy module is an AI-assisted monitoring
        component of the Smart Coma Patient Monitoring System.
        It is intended to help organize patient monitoring
        information for clinical review.

        Rehabilitation exercises, positioning, passive ROM,
        frequency, intensity and escalation decisions must be
        determined by qualified healthcare professionals based
        on the complete clinical assessment.
        </div>
    """)


# ============================================================
# ABOUT
# ============================================================

elif page == "🔬 About Project":

    st.html("""
        <div class="topbar">
            <div class="main-title">
                ARMedisense / SmartComa
            </div>
            <div class="subtitle">
                AI-based multimodal monitoring system
                for intelligent coma-patient monitoring
            </div>
        </div>
        """)

    st.html("""
        <div class="ai-panel">

        <h3>System Architecture</h3>

        <div class="pipeline">

            <div class="pipeline-item">
                EEG Dataset
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                EEG Preprocessing
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                CNN + BiLSTM
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                IoT Features
            </div>

            <div class="pipeline-arrow">+</div>

            <div class="pipeline-item">
                Clinical Features
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                Fusion V1
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                Patient-level AI Result
            </div>

        </div>

        </div>
        """)

    st.html(
        '<div class="section-title">'
        'Technologies'
        '</div>'
    )

    tech = pd.DataFrame({
        "Layer": [
            "EEG",
            "Machine Learning",
            "IoT",
            "Dashboard",
            "AR"
        ],
        "Technology": [
            "PhysioNet I-CARE EEG",
            "TensorFlow / Keras",
            "Heart Rate / Temperature / Movement",
            "Python / Streamlit / Plotly",
            "Unity / ARCore"
        ]
    })

    st.dataframe(
        tech,
        use_container_width=True,
        hide_index=True
    )

    st.html("""
        <div class="disclaimer">

        <b>Research prototype:</b>
        This system is intended for academic demonstration
        and AI-assisted clinical decision support.
        It is not a certified medical device and does not
        replace professional diagnosis, monitoring or treatment.

        </div>
        """)


# ============================================================
# FOOTER
# ============================================================

st.html("""
    <div class="footer">
        SMARTCOMA AI • Multimodal Patient Monitoring System
        <br>
        EEG + IoT + Clinical Intelligence • Academic Research Prototype
    </div>
    """)
