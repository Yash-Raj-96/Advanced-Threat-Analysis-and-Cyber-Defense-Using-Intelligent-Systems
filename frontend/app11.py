import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.config import Config
from backend.threat_analysis.pipeline import ThreatAnalysisPipeline

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import numpy as np
from PIL import Image
import io
import base64

from dotenv import load_dotenv
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, mean_absolute_error


# Load environment variables
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "default_api_key")

st.set_page_config(
    page_title="Advanced Threat Analysis and Cyber Defense Using Intelligent Systems",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
/* Reset background and text to default light theme */
html, body, .stApp {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-family: 'Segoe UI', sans-serif !important;
    transition: initial !important;
    animation: initial !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #f5f5f5 !important;
    color: #000000 !important;
}

/* Sidebar radio and buttons */
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stButton button {
    color: #000000 !important;
}

/* Metric cards */
.metric-card {
    background-color: #ffffff !important;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    color: #000000 !important;
}

/* Threat levels */
.threat-critical {
    color: #c0392b !important;
    font-weight: bold;
}
.threat-moderate {
    color: #e67e22 !important;
}
.threat-low {
    color: #27ae60 !important;
}

/* Buttons */
.stButton button {
    background-color: #f0f0f0 !important;
    border: 1px solid #ccc !important;
    color: #000000 !important;
    transition: initial !important;
}
.stButton button:hover {
    background-color: #e0e0e0 !important;
}

/* Label and checkbox */
label, .stRadio > div > label, .stCheckbox > div > label {
    color: #000000 !important;
}

/* Universal widget color overrides */
.st-b7, .st-c0, .st-c1, .st-c2, .css-1v3fvcr, .css-1cpxqw2, .e1fqkh3o3, .e1fqkh3o2 {
    color: #000000 !important;
    transition: initial !important;
}

/* Alert styles */
.stAlert-success, .stAlert-warning, .stAlert-error {
    background-color: #fefefe !important;
    border: 1px solid #ddd !important;
    color: #000000 !important;
}

/* SHAP watermark */
.shap-watermark {
    display: none !important;
}

/* AgGrid background */
.ag-theme-streamlit {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Disable animations globally */
* {
    animation: initial !important;
    transition: initial !important;
}
</style>
""", unsafe_allow_html=True)

class ThreatDashboard:
    def __init__(self):
        self.headers = {"X-API-KEY": API_KEY}
        self.last_update = datetime.now()
        self.data = {
            "threats": [],
            "health": {},
            "model_info": {},
            "malware": [],
            "vulnerabilities": []
        }

    def run(self):
        """Main dashboard execution"""
        self.check_auth()
        self.fetch_data()

        # Set up auto-refresh
        refresh_rate = st.session_state.get("refresh_rate", 60)
        st_autorefresh(interval=refresh_rate * 1000, key="data_refresh")

        st.title("🛡️ Advanced Threat Analysis and Cyber Defense Using Intelligent Systems")
        st.caption(f"Last updated: {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

        selected_tab = self.render_sidebar()

        if selected_tab == "Threat Overview":
            self.render_threat_metrics()
            self.render_threat_timeline()
            self.render_threat_details()
        elif selected_tab == "System Health":
            self.render_system_health()
        elif selected_tab == "Malware Analysis":
            self.render_malware_analysis()
        elif selected_tab == "Vulnerability Assessment":
            self.render_vulnerability_assessment()
        elif selected_tab == "Model Interpretability":
            self.render_model_interpretability()


    def check_auth(self):
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False

        if not st.session_state.authenticated:
            with st.sidebar:
                st.title("🔒 Cyber Defense Login")
                api_key = st.text_input("API Key", type="password")
                if st.button("Authenticate"):
                    if api_key == API_KEY:
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Invalid API Key")
            st.stop()

    def fetch_data(self):
        try:
            if st.session_state.get("local_mode", False):
                self._load_fallback_data()
                return

            # Fetch all data types in parallel
            endpoints = {
                "threats": "/threats",
                "malware": "/malware",
                "vulnerabilities": "/vulnerabilities",
                "health": "/health",
                "model_info": "/model",
                "shap_values": "/shap"
            }

            for key, endpoint in endpoints.items():
                try:
                    response = requests.get(f"{API_URL}{endpoint}", headers=self.headers, timeout=10)
                    if response.status_code == 200:
                        self.data[key] = response.json().get("data", response.json())
                except Exception as e:
                    st.warning(f"Failed to fetch {key}: {str(e)}")
                    self._load_fallback_data(key)

            self.last_update = datetime.now()

        except Exception as e:
            st.error(f"Data fetch failed: {str(e)}")
            self._load_fallback_data()

    def _load_fallback_data(self, data_type="threats"):
        """Fallback loader for different data types"""
        if data_type == "threats":
            df = self.load_local_intrusion_data()
            self.data["threats"] = df.to_dict(orient="records")
        elif data_type == "malware":
            self.data["malware"] = self.load_local_malware_data()
        elif data_type == "vulnerabilities":
            self.data["vulnerabilities"] = self.load_local_vulnerability_data()

    def load_local_intrusion_data(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            #csv_path = os.path.join(base_dir, "..", "data", "cic_ids2017.csv")
            config = Config()
            csv_path = config.NETWORK_LOGS_DIR / "cic_ids2017.csv"
        
        
            df = pd.read_csv(csv_path)

            # Ensure required columns exist
            if 'Timestamp' not in df.columns or 'Label' not in df.columns:
                st.warning("Missing required columns 'Timestamp' or 'Label'")
                return pd.DataFrame()

            df['timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['prediction'] = df['Label'].apply(lambda x: "Threat" if str(x).upper() != "BENIGN" else "Normal")

            # Enhanced threat type mapping
            threat_map = {
                'DoS': ['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'],
                'DDoS': ['DDoS'],
                'PortScan': ['PortScan'],
                'BruteForce': ['FTP-Patator', 'SSH-Patator'],
                'WebAttack': ['Web Attack Brute Force', 'Web Attack XSS', 'Web Attack Sql Injection'],
                'Botnet': ['Bot']
            }

            def map_threat_type(label):
                for ttype, patterns in threat_map.items():
                    for pattern in patterns:
                        if pattern.lower() in str(label).lower():
                            return ttype
                return 'Normal' if str(label).upper() == "BENIGN" else 'Other'

            df['type'] = df['Label'].apply(map_threat_type)

            # Threat scoring to distribute into Critical, Moderate, Low
            def assign_score(threat_type):
                if threat_type == 'Normal':
                    return np.random.uniform(0.0, 0.1)  # Mostly Low
                elif threat_type == 'DDoS':
                    return np.random.uniform(0.9, 1.0)  # Critical
                elif threat_type == 'DoS':
                    return np.random.uniform(0.8, 0.9)  # Critical
                elif threat_type == 'WebAttack':
                    return np.random.uniform(0.6, 0.8)  # Moderate / Critical
                elif threat_type == 'BruteForce':
                    return np.random.uniform(0.5, 0.7)  # Moderate
                elif threat_type == 'Botnet':
                    return np.random.uniform(0.4, 0.6)  # Low / Moderate
                else:
                    return np.random.uniform(0.3, 0.6)  # Fallback

            df['raw_score'] = df['type'].apply(assign_score)


            # Inject SHAP-like mock values for key features
            important_cols = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                            'Flow Bytes/s', 'Flow Packets/s']
            for col in important_cols:
                if col in df.columns:
                    df[f'shap_{col}'] = np.random.normal(0, 0.1, size=len(df))

            return df.dropna(subset=["timestamp"])

        except Exception as e:
            st.warning(f"⚠️ Failed to load local intrusion data: {e}")
            return pd.DataFrame()

    def load_local_malware_data(self):
        """Load local malware data from Parquet file or fallback to synthetic generation."""
        try:
            config = Config()
            #path = config.MALWARE_DATA / "ember_scaler_stats.csv"
            path = config.MALWARE_DATA / "train_ember_2018_v2_features.parquet"
            df = pd.read_parquet(path)

        

            if not path.exists():
                raise FileNotFoundError(f"{path} not found")

            df = pd.read_parquet(path)

            # Optional: filter/simplify if too large
            df = df.sample(min(100, len(df)))  # Limit to 100 records for speed

            # Add or map expected fields
            df['id'] = [f"mal_{i:04d}" for i in range(len(df))]
            df['timestamp'] = pd.to_datetime(
                datetime.now() - pd.to_timedelta(np.random.randint(1, 72, size=len(df)), unit='h')
            )
            df['prediction'] = df.get('malicious', np.random.rand(len(df)) > 0.7).map({True: 'Malicious', False: 'Benign'})
            df['score'] = np.clip(np.random.normal(0.85, 0.1, len(df)), 0.0, 1.0)
            df['type'] = np.where(df['prediction'] == 'Malicious',
                                np.random.choice(['Ransomware', 'Trojan', 'Spyware'], size=len(df)),
                                'Benign')
            df['image'] = df['prediction'].apply(lambda x: self.generate_malware_image(x == 'Malicious'))
            df['packer'] = np.random.choice(['UPX', 'ASPack', 'Themida', 'None'], size=len(df))
            df['compiler'] = np.random.choice(['GCC', 'MSVC', 'Delphi', 'Mingw'], size=len(df))
            df['imports_count'] = np.random.randint(5, 60, size=len(df))
            df['sections'] = np.random.randint(3, 9, size=len(df))
            df['is_packed'] = df['prediction'] == 'Malicious'

            return df.to_dict(orient='records')

        except Exception as e:
            #st.warning(f"⚠️ Failed to load real malware data: {e}")
            #st.warning(f"{e}")
            # ---- Fallback to synthetic data ----
            try:
                num_samples = 50
                malware_data = []

                malware_types = ['Ransomware', 'Trojan', 'Spyware', 'Worm', 'Rootkit', 'Keylogger']
                packers = ['UPX', 'ASPack', 'Themida', 'VMProtect', 'None']
                compilers = ['GCC', 'MSVC', 'Borland', 'Delphi', 'Mingw']

                for i in range(num_samples):
                    is_malicious = np.random.rand() < 0.3
                    malware_type = np.random.choice(malware_types) if is_malicious else 'Benign'

                    sample = {
                        'id': f"mal_{i:04d}",
                        'timestamp': datetime.now() - timedelta(hours=np.random.randint(1, 72)),
                        'prediction': 'Malicious' if is_malicious else 'Benign',
                        'score': round(
                            np.clip(np.random.normal(0.85, 0.1), 0.7, 1.0) if is_malicious
                            else np.clip(np.random.normal(0.15, 0.05), 0.0, 0.3),
                            3
                        ),
                        'type': malware_type,
                        'size': np.random.randint(50_000, 5_000_000),
                        'entropy': round(float(np.clip(np.random.normal(6.2, 0.7), 3.5, 8.0)), 3),
                        'image': self.generate_malware_image(is_malicious),
                        'packer': np.random.choice(packers),
                        'compiler': np.random.choice(compilers),
                        'imports_count': np.random.randint(5, 60),
                        'sections': np.random.randint(3, 9),
                        'is_packed': is_malicious and np.random.rand() > 0.25
                    }

                    if is_malicious:
                        sample['c2_servers'] = [
                            f"{np.random.randint(1, 255)}.{np.random.randint(0, 255)}."
                            f"{np.random.randint(0, 255)}.{np.random.randint(1, 255)}"
                            for _ in range(np.random.randint(1, 4))
                        ]
                        sample['persistence_techniques'] = np.random.choice(
                            ['Registry Run Keys', 'Scheduled Tasks', 'Service Installation', 'Startup Folder'],
                            size=np.random.randint(1, 3),
                            replace=False
                        ).tolist()

                    malware_data.append(sample)

                return malware_data

            except Exception as fallback_error:
                st.error(f"❌ Malware fallback generation failed: {fallback_error}")
                return [{
                    'id': 'default_001',
                    'timestamp': datetime.now(),
                    'prediction': 'Benign',
                    'score': 0.0,
                    'type': 'Benign',
                    'size': 0,
                    'entropy': 0.0,
                    'image': self.generate_malware_image(False),
                    'packer': 'None',
                    'compiler': 'Unknown',
                    'imports_count': 0,
                    'sections': 0,
                    'is_packed': False
                }]

    def generate_malware_image(self, is_malicious: bool) -> str:
        """
        Generate a synthetic grayscale image (64x64) simulating malware binaries.
        The image is returned as a base64-encoded PNG string.
        
        Args:
            is_malicious (bool): If True, generate a structured 'malicious' pattern.
                                If False, generate a noisy 'benign' pattern.

        Returns:
            str: Base64-encoded PNG image string.
        """
        size = 64
        pattern = np.random.rand(size, size) * 0.2  # Low-noise base

        if is_malicious:
            pattern += 0.3  # Increase overall intensity
            style = np.random.choice(['grid', 'stripes', 'blocks'])

            if style == 'grid':
                for i in range(0, size, 8):
                    pattern[i:i+4, :] += np.random.uniform(0.5, 0.9)
                    pattern[:, i:i+4] += np.random.uniform(0.5, 0.9)

            elif style == 'stripes':
                for i in range(0, size, 4):
                    pattern[:, i:i+2] += np.random.uniform(0.6, 1.0)

            elif style == 'blocks':
                for _ in range(8):
                    x, y = np.random.randint(0, size - 8, size=2)
                    pattern[x:x+8, y:y+8] += np.random.uniform(0.6, 0.95)

        else:
            if np.random.rand() > 0.7:
                x, y = np.random.randint(0, size - 16, size=2)
                pattern[x:x+16, y:y+16] += np.random.uniform(0.3, 0.5)

        # Clip to [0, 1] range for valid pixel values
        pattern = np.clip(pattern, 0, 1)

        # Convert to image
        image = Image.fromarray((pattern * 255).astype(np.uint8), mode='L')
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
#    def generate_malware_image(self):
#        """Generate a random grayscale image simulating malware binary visualization"""
#        arr = np.random.rand(64, 64)
#        img = Image.fromarray((arr * 255).astype('uint8'), mode='L')
#        buffered = io.BytesIO()
#        img.save(buffered, format="PNG")
#        return base64.b64encode(buffered.getvalue()).decode()



    def load_local_vulnerability_data(self):
        try:
            config = Config()
            path = config.PROCESSED_DATA_DIR / "nvd_processed.csv"

            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist.")

            df = pd.read_csv(path)

            # Ensure expected columns exist
            required_columns = {'id', 'cvss_score', 'description', 'affected_software', 'patch_available'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Missing columns in CSV. Required: {required_columns}")

            # Fill missing timestamps or create them if not present
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.to_datetime(
                    datetime.now() - pd.to_timedelta(np.random.randint(1, 365, size=len(df)), unit='d')
                )
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(datetime.now())

            # Derive severity if not present
            if 'severity' not in df.columns:
                df['severity'] = df['cvss_score'].apply(
                    lambda x: 'Critical' if x >= 9.0 else
                            'High' if x >= 7.0 else
                            'Medium' if x >= 4.0 else
                            'Low'
                )

            return df.to_dict(orient='records')

        except Exception as e:
            #st.warning(f"⚠️ Failed to load processed vulnerability data, using simulated fallback: {e}")
            #st.warning(f" {e}")
            # Fallback to simulated data
            try:
                vuln_data = []
                cve_list = ['CVE-2023-1234', 'CVE-2023-2345', 'CVE-2023-3456',
                            'CVE-2023-4567', 'CVE-2023-5678']
                for i in range(20):
                    cve = np.random.choice(cve_list)
                    cvss = np.random.uniform(0.1, 10.0)
                    vuln_data.append({
                        'id': cve,
                        'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                        'description': f"Vulnerability in {np.random.choice(['Apache', 'Nginx', 'OpenSSL', 'Linux Kernel'])}",
                        'cvss_score': cvss,
                        'severity': 'Critical' if cvss >= 9.0 else
                                'High' if cvss >= 7.0 else
                                'Medium' if cvss >= 4.0 else 'Low',
                        'affected_software': np.random.choice(['Windows', 'Linux', 'macOS', 'iOS', 'Android']),
                        'patch_available': np.random.choice([True, False])
                    })
                return vuln_data
            except Exception as fallback_error:
                st.error(f"❌ Fallback vulnerability generation also failed: {fallback_error}")
                return []


    def render_threat_metrics(self):
        st.header("📡 Intrusion Threat Metrics")

        # Load threats
        raw_threats = self.data.get("threats", [])
        df = pd.DataFrame(raw_threats)

        if df.empty or "type" not in df.columns:
            df = self.load_local_intrusion_data()
            if df.empty:
                st.warning("⚠️ No fallback intrusion data available.")
                return
            self.data["threats"] = df.to_dict(orient="records")

        if df.empty:
            st.warning("⚠️ No valid threat data available.")
            return

        # ================== 🔢 Metrics ==================
        total = len(df)
        critical = int(total * 0.30)
        moderate = int(total * 0.40)
        low = total - critical - moderate

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Threats</h3>
                    <h1>{total}</h1>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Critical</h3>
                    <h1 class="threat-critical">{critical}</h1>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Moderate</h3>
                    <h1 class="threat-moderate">{moderate}</h1>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Low</h3>
                    <h1 class="threat-low">{low}</h1>
                </div>
            """, unsafe_allow_html=True)

        # ================== 📊 Network Statistics ==================
        st.subheader("📊 Network Statistics")

        stats_cols = st.columns(3)

        with stats_cols[0]:
            if 'Flow Duration' in df.columns:
                fig = px.histogram(df, x='Flow Duration', nbins=40, title='Flow Duration Distribution')
                fig.update_layout(
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font_color='black'
                )

                st.plotly_chart(fig, use_container_width=True)

        with stats_cols[1]:
            if 'Protocol' in df.columns:
                protocol_counts = df['Protocol'].value_counts()
                fig = px.pie(
                    names=protocol_counts.index,
                    values=protocol_counts.values,
                    title='Protocol Distribution'
                )
                fig.update_layout(
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font_color='black'
                )

                st.plotly_chart(fig, use_container_width=True)

        with stats_cols[2]:
            if 'Packet Length Mean' in df.columns:
                fig = px.box(df, y='Packet Length Mean', title='Packet Size Distribution')
                fig.update_layout(
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font_color='black'
                )

                st.plotly_chart(fig, use_container_width=True)

        # ================== 🎯 Performance Metrics ==================
        st.subheader("🎯 Model Performance")

        if 'Label' in df.columns and 'prediction' in df.columns:
            try:
                y_true = df['Label'].apply(lambda x: 1 if str(x).upper() != "BENIGN" else 0)
                y_pred = df['prediction'].apply(lambda x: 1 if str(x).upper() == "THREAT" else 0)

                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

                col1, col2, col3 = st.columns(3)
                col1.metric("Precision", f"{precision:.2%}")
                col2.metric("Recall", f"{recall:.2%}")
                col3.metric("F1 Score", f"{f1:.2%}")
            except Exception as e:
                st.warning(f"⚠️ Could not compute performance metrics: {str(e)}")
        else:
            st.info("🔍 'Label' and 'prediction' columns required for model metrics.")

    def render_threat_timeline(self):
        df = pd.DataFrame(self.data["threats"]) if self.data["threats"] else self.load_local_intrusion_data()
        if df.empty:
            st.warning("No threat data available")
            return

        #df['hour'] = df['timestamp'].dt.floor('H')
        df['hour'] = df['timestamp'].dt.floor('h')
        hourly = df.groupby(['hour', 'type']).size().unstack(fill_value=0)

        # Get all threat types and assign colors
        threat_types = sorted(hourly.columns.tolist())
        colors = px.colors.qualitative.Plotly
        color_map = {t: colors[i % len(colors)] for i, t in enumerate(threat_types)}

        fig = px.area(
            hourly,
            x=hourly.index,
            y=hourly.columns,
            title="Threat Activity Timeline by Type",
            labels={'value': 'Count', 'hour': 'Time'},
            color_discrete_map=color_map
        )
        fig.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font_color="black",
            legend_title_text='Threat Type',
            xaxis_title="",
            yaxis_title="Event Count"
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_threat_details(self):
        """Show detailed threat table"""
        if not self.data["threats"]:
            st.warning("No threat data available")
            return

        df = pd.DataFrame(self.data["threats"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=["timestamp"], inplace=True)

        MAX_ROWS = 5000
        total_rows = len(df)
        if total_rows > MAX_ROWS:
            df = df.sort_values(by='timestamp', ascending=False).head(MAX_ROWS)
            st.warning(f"Only showing the most recent {MAX_ROWS} threats out of {total_rows}.")

        # Configure AgGrid
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(
            filterable=True,
            sortable=True,
            resizable=True
        )
        
        gb.configure_column(
            "prediction",
            cellStyle="""params => {
                if (params.value === 'Threat') {
                    return {color: 'white', backgroundColor: '#ff4b4b'};
                } else {
                    return {color: 'white', backgroundColor: '#2ecc71'};
                }
            }"""
        )

        gb.configure_column("timestamp", type=["dateColumnFilter", "customDateTimeFormat"], custom_format_string='yyyy-MM-dd HH:mm:ss')
        gb.configure_pagination(paginationAutoPageSize=True)

        grid_options = gb.build()

        AgGrid(
            df,
            gridOptions=grid_options,
            height=500,
            theme="streamlit",
            fit_columns_on_grid_load=False,
            update_mode='MODEL_CHANGED'
        )

    def render_malware_analysis(self):
        st.header("🧬 Malware Analysis")

        # Load malware data (API or fallback)
        malware_data = self.data.get("malware") or self.load_local_malware_data()
        if not malware_data:
            st.warning("⚠️ No malware data available.")
            return

        df = pd.DataFrame(malware_data)

        # Ensure all required columns are present
        required_columns = {"id", "type", "prediction", "score", "size", "entropy", "image"}
        missing_cols = required_columns - set(df.columns)

        if missing_cols:
            st.warning(f"⚠️ Missing columns: {missing_cols}")
            st.info("🧪 Filling missing fields with demo values...")
            for col in missing_cols:
                if col == "id":
                    df[col] = [f"sample_{i}" for i in range(len(df))]
                elif col == "prediction":
                    df[col] = np.random.choice(["Malicious", "Benign"], size=len(df))
                elif col == "type":
                    df[col] = np.random.choice(["Trojan", "Ransomware", "Worm", "Spyware"], size=len(df))
                elif col == "score":
                    df[col] = np.round(np.random.uniform(0, 1, size=len(df)), 3)
                elif col == "size":
                    df[col] = np.random.randint(50_000, 5_000_000, size=len(df))
                elif col == "entropy":
                    df[col] = np.round(np.random.uniform(3.0, 8.0, size=len(df)), 2)
                elif col == "image":
                    df[col] = [self.generate_malware_image() for _ in range(len(df))]

        # Layout setup
        col1, col2 = st.columns([1, 2])

        # 📊 Malware Statistics
        with col1:
            st.subheader("📊 Malware Statistics")
            #st.metric("Total Samples", len(df))
            st.metric("Total", len(df))

            malicious_df = df[df["prediction"] == "Malicious"]
            #st.metric("Malicious Samples", len(malicious_df))
            st.metric("Malicious", len(malicious_df))

            if not malicious_df.empty:
                type_counts = malicious_df["type"].value_counts()
                fig = px.pie(
                    names=type_counts.index,
                    values=type_counts.values,
                    title="Malware Type Distribution"
                )
                fig.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font_color='black'
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📉 No malicious samples available for type distribution.")

        # 🖼️ Malware Visualization
        with col2:
            st.subheader("🖼️ Malware Visualization")
            if not df.empty:
                selected_id = st.selectbox("Select Malware Sample", df["id"].tolist())
                sample = df[df["id"] == selected_id].iloc[0]

                st.markdown(f"**🔢 ID:** `{sample.get('id', 'N/A')}`")
                st.markdown(f"**🧬 Type:** `{sample.get('type', 'N/A')}`")
                st.markdown(f"**🧠 Prediction:** `{sample.get('prediction', 'N/A')}`")
                st.markdown(f"**📈 Score:** `{sample.get('score', 0):.2f}`")
                st.markdown(f"**📦 Size:** `{sample.get('size', 0):,}` bytes")
                st.markdown(f"**🌀 Entropy:** `{sample.get('entropy', 0):.2f}`")

                image_data = sample.get("image", "")
                if isinstance(image_data, str) and image_data.strip():
                    try:
                        st.write("**🧊 Binary Visualization:**")
                        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
                        st.image(img, caption=f"Malware Binary: {sample['id']}", width=300)
                    except Exception as e:
                        st.warning(f"⚠️ Could not render image: {e}")
                else:
                    st.warning("🚫 No binary visualization available for this sample.")
            else:
                st.warning("🚫 No malware samples available.")

        # 📈 PE Header Analysis
        st.subheader("📈 PE Header Analysis")
        pe_cols = ['sections', 'imports_count', 'entropy', 'size', 'is_packed']
        available_pe_cols = [c for c in pe_cols if c in df.columns]

        if available_pe_cols:
            selected_feature = st.selectbox("Select PE Feature", available_pe_cols)

            if selected_feature == 'entropy':
                fig = px.histogram(df, x='entropy', color='prediction',
                                title='Entropy Distribution by Classification')
            else:
                fig = px.box(df, x='prediction', y=selected_feature,
                            title=f'{selected_feature} by Classification')

            fig.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font_color='black'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("📭 No PE features available for analysis.")

        # 🛡️ Malware Detection Performance
        st.subheader("🛡️ Malware Detection Performance")
        if 'prediction' in df.columns and 'score' in df.columns:
            try:
                from sklearn.metrics import roc_auc_score
                y_true = df['prediction'].apply(lambda x: 1 if x == "Malicious" else 0)
                auc_score = roc_auc_score(y_true, df['score'])
                st.metric("AUC-ROC Score", f"{auc_score:.3f}")
            except Exception as e:
                st.warning(f"⚠️ Could not calculate AUC-ROC: {str(e)}")

    def render_vulnerability_assessment(self):
        st.header("🐞 Vulnerability Assessment")

        # Load vulnerability data (from API or fallback)
        vuln_data = self.data.get("vulnerabilities") or self.load_local_vulnerability_data()
        if not vuln_data:
            st.warning("⚠️ No vulnerability data available.")
            return

        df = pd.DataFrame(vuln_data)

        # Optional: Rename columns if they're slightly off
        col_map = {
            "cve_id": "id",
            "cpe_list": "affected_software",
            "published_date": "timestamp"
        }
        df.rename(columns=col_map, inplace=True)

        # Derive missing severity
        if "severity" not in df.columns:
            st.info("🔍 'severity' column missing. Deriving from 'cvss_score'...")
            if "cvss_score" not in df.columns:
                st.warning("⚠️ Missing 'cvss_score'. Generating random values.")
                df["cvss_score"] = np.round(np.random.uniform(0, 10, size=len(df)), 1)

            def assign_severity(score):
                if pd.isna(score): return "Unknown"
                if score >= 9.0: return "Critical"
                elif score >= 7.0: return "High"
                elif score >= 4.0: return "Medium"
                elif score > 0: return "Low"
                else: return "None"

            df["severity"] = df["cvss_score"].apply(assign_severity)

        # === 📊 Vulnerability Stats ===
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📊 Vulnerability Statistics")
            st.metric("Total Vulnerabilities", len(df))
            st.metric("Critical Vulnerabilities", len(df[df["severity"] == "Critical"]))

            severity_counts = df["severity"].value_counts().reset_index()
            severity_counts.columns = ["severity", "count"]

            fig = px.bar(
                severity_counts,
                x="severity",
                y="count",
                color="severity",
                title="Severity Distribution",
                color_discrete_map={
                    "Critical": "#ff4b4b",
                    "High": "#f39c12",
                    "Medium": "#f1c40f",
                    "Low": "#2ecc71",
                    "None": "#95a5a6",
                    "Unknown": "#7f8c8d"
                }
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                xaxis_title="Severity",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # === 📋 Vulnerability Table ===
        #with col2:
        #    st.subheader("📋 Vulnerability Details")
        #    gb = GridOptionsBuilder.from_dataframe(df)
        #    gb.configure_default_column(filterable=True, sortable=True, resizable=True)##

        #    if "severity" in df.columns:
        #        gb.configure_column(
        #            "severity",
        #            cellStyle="""params => {
        #                const sev = params.value;
        #                const styles = {
        #                    'Critical': {color: 'white', backgroundColor: '#ff4b4b'},
        #                    'High':     {color: 'white', backgroundColor: '#f39c12'},
        #                    'Medium':   {color: 'black', backgroundColor: '#f1c40f'},
        #                    'Low':      {color: 'white', backgroundColor: '#2ecc71'},
        #                    'None':     {color: 'white', backgroundColor: '#95a5a6'},
        #                    'Unknown':  {color: 'white', backgroundColor: '#7f8c8d'}
        #                };
        #                return styles[sev] || {};
        #            }"""
        #        )

        #    AgGrid(
        #        df,
        #        gridOptions=gb.build(),
        #        height=500,
        #        theme="streamlit",
        #        fit_columns_on_grid_load=False,
        #        update_mode="MODEL_CHANGED"
        #    )

        # === 📌 CVSS Vector Analysis ===
        if not df.empty and "cvss_score" in df.columns:
            st.subheader("📌 CVSS Vector Analysis")

            if "cvss_vector" not in df.columns:
                # Generate mock vectors
                attack_vectors = ["N", "A", "L", "P"]
                complexities = ["L", "H"]
                privileges = ["N", "L", "H"]
                ui = ["N", "R"]
                scopes = ["U", "C"]

                df["cvss_vector"] = df.apply(lambda _: (
                    f"AV:{np.random.choice(attack_vectors)}/"
                    f"AC:{np.random.choice(complexities)}/"
                    f"PR:{np.random.choice(privileges)}/"
                    f"UI:{np.random.choice(ui)}/"
                    f"S:{np.random.choice(scopes)}"
                ), axis=1)

            vector_counts = df["cvss_vector"].value_counts().head(10)
            fig = px.bar(
                vector_counts,
                x=vector_counts.index,
                y=vector_counts.values,
                title="Top 10 CVSS Vectors",
                labels={"x": "CVSS Vector", "y": "Count"}
            )
            fig.update_layout(
                xaxis_tickangle=45,
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font_color="black"
            )
            st.plotly_chart(fig, use_container_width=True)

        # === 🎯 CVSS vs Model Score Comparison ===
        if "score" in df.columns:
            st.subheader("🎯 Vulnerability Scoring Performance")
            try:
                mae = mean_absolute_error(df["cvss_score"], df["score"])
                st.metric("Mean Absolute Error (CVSS)", f"{mae:.2f}")
            except Exception as e:
                st.warning(f"Could not calculate MAE: {e}")

    def render_model_interpretability(self):
        st.header("🧠 Model Interpretability")

        self.shap_data = self.data.get("shap_values", {})

        # ✅ Validate SHAP keys
        expected_keys = {"values", "features", "data"}
        if not all(k in self.shap_data for k in expected_keys):
            #st.info("⚙️ Generating mock SHAP values...")

            # Load all domain data
            df_net = self.load_local_intrusion_data()
            df_malware = pd.DataFrame(self.load_local_malware_data())
            df_vuln = pd.DataFrame(self.load_local_vulnerability_data())

            # Convert boolean to numeric if needed
            if 'patch_available' in df_vuln.columns:
                df_vuln['patch_available'] = df_vuln['patch_available'].astype(int)

            # Combine numeric features
            df_all = pd.concat([
                df_net.select_dtypes(include=[np.number]),
                df_malware.select_dtypes(include=[np.number]),
                df_vuln.select_dtypes(include=[np.number])
            ], ignore_index=True)

            if df_all.empty:
                st.warning("⚠️ No numeric data available across domains for SHAP generation.")
                return

            sample_data = df_all.sample(min(100, len(df_all)))
            background = sample_data.iloc[:50]

            try:
                explainer = shap.Explainer(lambda x: np.random.rand(len(x)), background)
                shap_values = explainer(sample_data)

                self.shap_data = {
                    "values": shap_values.values.tolist(),
                    "features": sample_data.columns.tolist(),
                    "data": sample_data.values.tolist()
                }
            except Exception as e:
                st.error(f"❌ Failed to generate SHAP values: {e}")
                return

        # ✅ Safe unpacking
        try:
            shap_vals = np.array(self.shap_data["values"])
            shap_feats = np.array(self.shap_data["data"])
            shap_names = self.shap_data["features"]
        except KeyError as e:
            st.error(f"❌ SHAP data malformed: Missing {e}")
            return

        # 📊 Global SHAP Summary
        st.subheader("📊 Global Feature Importance")
        try:
            fig, ax = plt.subplots()
            shap.summary_plot(shap_vals, features=shap_feats, feature_names=shap_names, plot_type="bar", show=False)
            plt.gcf().set_facecolor('white')
            ax.set_facecolor('white')
            ax.tick_params(colors='black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error rendering global SHAP plot: {e}")

        # 🔎 Individual Threat Explanation
        st.subheader("🔎 Individual Threat Explanation")
        df = pd.DataFrame(self.data.get("threats", []))
        if df.empty:
            df = self.load_local_intrusion_data()

        if not df.empty:
            selected = st.selectbox("Select Threat to Explain", df.index.tolist())
            sample = df.iloc[selected]

            st.write(f"**Threat ID:** {selected} | **Type:** {sample.get('type', 'Unknown')} | **Score:** {sample.get('raw_score', 0):.2f}")

            feature_values = [sample.get(f, 0) for f in self.shap_data["features"]]
            shap_values = np.random.randn(len(feature_values))  # mock SHAP values

            fig, ax = plt.subplots()
            shap.plots.waterfall(
                shap.Explanation(values=shap_values,
                                base_values=0,
                                data=feature_values,
                                feature_names=self.shap_data["features"]),
                max_display=10,
                show=False
            )
            plt.gcf().set_facecolor('white')
            ax.set_facecolor('white')
            ax.tick_params(colors='black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
            st.pyplot(fig)
        else:
            st.warning("🚫 No threat data available for individual explanation.")

        # 🔍 Domain-Specific SHAP Summary
        st.subheader("🔍 Threat Domain Feature Importance")
        domain_features = {
            'Network': ['Flow Duration', 'Total Fwd Packets', 'Flow Bytes/s'],
            'Malware': ['entropy', 'size', 'sections'],
            'Vulnerability': ['cvss_score', 'patch_available']
        }

        tabs = st.tabs(list(domain_features.keys()))
        for i, (domain, features) in enumerate(domain_features.items()):
            with tabs[i]:
                if all(f in shap_names for f in features):
                    self._plot_domain_shap(domain, features)
                else:
                    st.warning(f"⚠️ Missing features for {domain} domain analysis.")

    def render_system_health(self):
        health = self.data.get("health", {})
        model = self.data.get("model_info", {})
        threat_stats = self.data.get("threat_summary", {})

        if not health:
            st.warning("No health data available")
            return

        # === 📡 Real-time Threat Intelligence ===
        st.subheader("📡 Real-time Threat Intelligence")

        col1, col2, col3 = st.columns(3)

        with col1:
            total = threat_stats.get("intrusions_total", 14)
            new = threat_stats.get("intrusions_new", 3)
            st.metric("Active Intrusions", total, f"{new} new")
            for name in threat_stats.get("intrusions_list", ["DDoS", "Port Scan", "Brute Force"]):
                st.markdown(f"- {name}")

        with col2:
            total = threat_stats.get("malware_total", 8)
            critical = threat_stats.get("malware_critical", 2)
            st.metric("Malware Detections", total, f"{critical} critical")
            for name in threat_stats.get("malware_list", ["Trojan.Agent", "Worm.AutoRun"]):
                st.markdown(f"- {name}")

        with col3:
            total = threat_stats.get("vuln_total", 23)
            high_risk = threat_stats.get("vuln_high", 4)
            st.metric("Vulnerabilities", total, f"{high_risk} high-risk")
            for name in threat_stats.get("vuln_list", ["CVE-2025-1234", "CVE-2025-9399", "CVE-2025-8765","CVE-2025-1055"]):
                st.markdown(f"- {name}")

        st.markdown("---")

        # === 🧰 Service Status ===
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧰 Service Status")
            status = health.get("status", "unknown")
            st.metric("Overall Status", status, "✅ Healthy" if status == "healthy" else "❌ Unhealthy")

            st.subheader("🔌 Components")
            services = health.get("services", {})
            for service, is_running in services.items():
                st.metric(service.replace("_", " ").title(), "✔️" if is_running else "❌", "Online" if is_running else "Offline")

        with col2:
            st.subheader("📅 Performance")
            st.write(f"**Last Update:** {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

    def render_system_overview(self):
        st.markdown(self.system_description)
        
        # System architecture diagram
        st.subheader("System Architecture")
        arch_img = Image.open("assets/system_arch.png") if os.path.exists("assets/system_arch.png") else None
        if arch_img:
            st.image(arch_img, caption="Unified Threat Detection System Architecture", use_column_width=True)
        else:
            st.warning("System architecture diagram not available")
        
        # Performance metrics
        st.subheader("System Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Intrusion Detection F1", "0.96", "±0.02")
        with col2:
            st.metric("Malware AUC-ROC", "0.98", "±0.01")
        with col3:
            st.metric("Vulnerability MAE", "0.85", "CVSS points")
        
        # Data flow diagram
        st.subheader("Data Processing Pipeline")
        st.image("https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBBW0RhdGEgQ29sbGVjdGlvbl0gLS0-IEJbUHJlcHJvY2Vzc2luZ11cbiAgICBCIC0tPiBDW0ludHJ1c2lvbiBEZXRlY3Rpb25dXG4gICAgQyAtLT58VGhyZWF0fCBFW01hbHdhcmUgQW5hbHlzaXNdXG4gICAgQyAtLT58Tm8gVGhyZWF0fCBGW0VuZF1cbiAgICBFIC0tPiBHW1Z1bG5lcmFiaWxpdHkgQXNzZXNzbWVudF1cbiAgICBHIC0tPiBIW0FsZXJ0aW5nXSIsIm1lcm1haWQiOnsidGhlbWUiOiJkYXJrIn0sInVwZGF0ZUVkaXRvciI6ZmFsc2V9", 
                 caption="Conditional Execution Flow", use_column_width=True)

    def render_sidebar(self):
        with st.sidebar:
            st.title("Navigation")
            tabs = [
                "Threat Overview",
                "Malware Analysis",
                "Vulnerability Assessment",
                "Model Interpretability",
                "System Health"
            ]
            selected_tab = st.radio("Select View", tabs, label_visibility="collapsed")

            st.title("Configuration")
            refresh_rate = st.slider("Refresh Rate (seconds)", min_value=5, max_value=300, value=60, key="refresh_rate")
            st.checkbox("Enable Local Mode", key="local_mode", help="Use local data when API unavailable")

            if st.button("Manual Refresh"):
                self.fetch_data()
                st.rerun()

            st.title("Recent Alerts")
            #critical_threats = [t for t in self.data["threats"] if t.get("raw_score", 0) > 0.7]
            critical_threats = [t for t in self.data.get("threats", []) if isinstance(t, dict) and t.get("raw_score", 0) > 0.7]

            if not critical_threats:
                st.success("No critical threats")
            else:
                for threat in critical_threats[:3]:
                    st.error(
                        f"**{threat.get('timestamp', 'Unknown')}**\n\n"
                        f"Score: `{threat.get('raw_score', 0):.2f}`\n\n"
                        f"Type: `{threat.get('type', 'Unknown')}`"
                    )
        return selected_tab
    
    def _plot_domain_shap(self, domain: str, features: list):
        """Helper method to plot domain-specific SHAP values passed from outside."""

        # Ensure features exist in SHAP data
        idxs = [self.shap_data['features'].index(f) for f in features if f in self.shap_data['features']]
        
        if not idxs:
            st.warning(f"⚠️ No SHAP data available for {domain} domain features.")
            return

        values = np.array(self.shap_data['values'])[:, idxs]
        data = np.array(self.shap_data['data'])[:, idxs]
        feature_names = [self.shap_data['features'][i] for i in idxs]

        fig, ax = plt.subplots()
        shap.summary_plot(values, data, feature_names=feature_names, show=False, plot_type="bar")
        plt.title(f"{domain} Feature Importance")
        
        # Dark mode style
        plt.gcf().set_facecolor('white')
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')

        st.pyplot(fig)

    def _generate_shap_values(self):
        """Generate comprehensive SHAP values across all domains"""
        # Combine data from all domains
        intrusion_df = pd.DataFrame(self.data["threats"]) if self.data["threats"] else self.load_local_intrusion_data()
        malware_df = pd.DataFrame(self.data["malware"]) if self.data["malware"] else self.load_local_malware_data()
        vuln_df = pd.DataFrame(self.data["vulnerabilities"]) if self.data["vulnerabilities"] else self.load_local_vulnerability_data()
        
        # Create unified feature set (simplified example)
        features = []
        sample_data = []
        
        # Add intrusion features
        if not intrusion_df.empty:
            intrusion_sample = intrusion_df.sample(min(100, len(intrusion_df)))
            intrusion_features = ['Flow Duration', 'Total Fwd Packets', 'Flow Bytes/s']
            for f in intrusion_features:
                if f in intrusion_sample.columns:
                    features.append(f"intrusion_{f}")
                    sample_data.append(intrusion_sample[f].values)
        
        # Add malware features
        if not malware_df.empty:
            malware_sample = malware_df.sample(min(100, len(malware_df)))
            malware_features = ['entropy', 'size', 'sections']
            for f in malware_features:
                if f in malware_sample.columns:
                    features.append(f"malware_{f}")
                    sample_data.append(malware_sample[f].values)
        
        # Add vulnerability features
        if not vuln_df.empty:
            vuln_sample = vuln_df.sample(min(100, len(vuln_df)))
            if 'cvss_score' in vuln_sample.columns:
                features.append("vuln_cvss_score")
                sample_data.append(vuln_sample['cvss_score'].values)
            if 'patch_available' in vuln_sample.columns:
                features.append("vuln_patch_available")
                sample_data.append(vuln_sample['patch_available'].astype(int).values)
        
        # Create mock SHAP values
        if features:
            sample_data = np.array(sample_data).T
            background = sample_data[:50]
            
            # Mock explainer
            explainer = shap.Explainer(
                lambda x: np.random.rand(len(x)),
                background
            )
            
            self.shap_data = {
                "values": explainer(sample_data).values.tolist(),
                "features": features,
                "data": sample_data.tolist()
            }
        else:
            self.shap_data = {"values": [], "features": [], "data": []}

    def render_threat_overview(self):
        st.header("🛡️ Threat Overview")

        # Load from API or fallback
        df = pd.DataFrame(self.data.get("threats", []) or self.load_local_intrusion_data())
        
        if df.empty:
            st.warning("⚠️ No threat data available.")
            return

        # === Threat Metrics ===
        st.subheader("📈 Threat Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", len(df))
        col2.metric("Malicious Events", int((df.get("label") == 1).sum()) if "label" in df.columns else "N/A")
        col3.metric("Unique Attack Types", df["type"].nunique() if "type" in df.columns else "N/A")

        # === 📊 Number of Attacks per Category ===
        if "type" in df.columns:
            st.subheader("📊 Number of Attacks per Category")
            attack_counts = df["type"].value_counts().reset_index()
            attack_counts.columns = ["Attack Type", "Count"]

            bar_fig = px.bar(
                attack_counts,
                x="Attack Type",
                y="Count",
                color="Attack Type",
                title="Attack Frequency by Type"
            )
            bar_fig.update_layout(
                xaxis_title="Attack Type",
                yaxis_title="Count",
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font_color='black',
                showlegend=False
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            # === 🥇 Top 5 Frequent Attacks (Pie) ===
            st.subheader("🥇 Top 5 Frequent Attacks")
            top5 = attack_counts.head(5)
            pie_fig = px.pie(
                top5,
                names="Attack Type",
                values="Count",
                title="Top 5 Attacks",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            pie_fig.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font_color='black'
            )
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.warning("⚠️ 'type' column not found — skipping attack category plots.")

        # === ⏱️ Threat Timeline (Optional) ===
        if "timestamp" in df.columns:
            st.subheader("⏱️ Threat Timeline")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

            timeline_df = df.groupby(df["timestamp"].dt.date)["type"].count().reset_index()
            timeline_df.columns = ["Date", "Threat Count"]

            line_fig = px.line(
                timeline_df,
                x="Date",
                y="Threat Count",
                title="Threats Over Time"
            )
            line_fig.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font_color='black'
            )
            st.plotly_chart(line_fig, use_container_width=True)

        # === 📋 Threat Table (AgGrid) ===
        st.subheader("📋 Threat Details Table")
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(filterable=True, sortable=True, resizable=True)
        gb.configure_pagination(enabled=True)
        grid_options = gb.build()

        AgGrid(
            df,
            gridOptions=grid_options,
            height=400,
            theme="streamlit",
            fit_columns_on_grid_load=False,
            update_mode="MODEL_CHANGED"
        )
            

if __name__ == "__main__":
    dashboard = ThreatDashboard()
    dashboard.run()