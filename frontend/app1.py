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
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #1a1d24;
    }
    .metric-card {
        background-color: #1a1d24;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .threat-critical {
        color: #ff4b4b !important;
        font-weight: bold;
    }
    .threat-moderate {
        color: #f39c12 !important;
    }
    .threat-low {
        color: #2ecc71 !important;
    }
    .st-b7, .st-c0, .st-c1, .st-c2 {
        color: #ffffff !important;
    }
    .shap-watermark {
        display: none !important;
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
            csv_path = os.path.join(base_dir, "..", "data", "cic_ids2017.csv")
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['prediction'] = df['Label'].apply(lambda x: "Threat" if str(x).upper() != "BENIGN" else "Normal")
            
            # Enhanced threat types based on CIC-IDS2017
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
            
            # Enhanced scoring based on threat type
            def assign_score(threat_type):
                if threat_type == 'Normal':
                    return np.random.uniform(0.0, 0.2)
                elif threat_type == 'DDoS':
                    return np.random.uniform(0.9, 1.0)
                elif threat_type in ['DoS', 'WebAttack']:
                    return np.random.uniform(0.7, 0.9)
                elif threat_type in ['BruteForce', 'Botnet']:
                    return np.random.uniform(0.5, 0.7)
                else:
                    return np.random.uniform(0.3, 0.5)
            
            df['raw_score'] = df['type'].apply(assign_score)
            
            # Add SHAP values for interpretability
            for col in ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                       'Flow Bytes/s', 'Flow Packets/s']:
                df[f'shap_{col}'] = np.random.normal(0, 0.1, len(df))
            
            return df.dropna(subset=["timestamp"])
        except Exception as e:
            st.warning(f"⚠️ Failed to load local intrusion data: {e}")
            return pd.DataFrame()

    def load_local_malware_data(self):
        try:
            # Simulate malware data with EMBER features
            num_samples = 50
            malware_data = []
            
            for i in range(num_samples):
                is_malicious = np.random.choice([True, False], p=[0.3, 0.7])
                malware_data.append({
                    'id': f"sample_{i}",
                    'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 72)),
                    'prediction': 'Malicious' if is_malicious else 'Benign',
                    'score': np.random.uniform(0.8, 1.0) if is_malicious else np.random.uniform(0.0, 0.2),
                    'type': np.random.choice(['Ransomware', 'Trojan', 'Spyware', 'Worm']) if is_malicious else 'Benign',
                    'size': np.random.randint(1000, 5000000),
                    'entropy': np.random.uniform(4.0, 7.0),
                    'image': self.generate_malware_image()
                })
            return malware_data
        except Exception as e:
            st.warning(f"⚠️ Failed to load local malware data: {e}")
            return []

    def load_local_vulnerability_data(self):
        try:
            # Simulate vulnerability data
            vuln_data = []
            cve_list = [
                'CVE-2023-1234', 'CVE-2023-2345', 'CVE-2023-3456',
                'CVE-2023-4567', 'CVE-2023-5678'
            ]
            
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
        except Exception as e:
            st.warning(f"⚠️ Failed to load local vulnerability data: {e}")
            return []

    def generate_malware_image(self):
        """Generate a random grayscale image simulating malware binary visualization"""
        arr = np.random.rand(64, 64)
        img = Image.fromarray((arr * 255).astype('uint8'), mode='L')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def render_threat_metrics(self):
        threats = self.data["threats"]
        malware = self.data["malware"]
        vulnerabilities = self.data["vulnerabilities"]

        if not threats:
            st.info("Loading local fallback data for threat metrics...")
            df = self.load_local_intrusion_data()
            threats = df.to_dict(orient="records")
        else:
            df = pd.DataFrame(threats)

        if df.empty:
            st.warning("No data available to render metrics.")
            return

        # Calculate threat levels
        critical = moderate = low = 0
        for t in threats:
            score = t.get("raw_score", 0)
            if score > 0.8:
                critical += 1
            elif score > 0.5:
                moderate += 1
            elif score > 0.0:
                low += 1

        # Calculate malware stats
        malware_df = pd.DataFrame(malware) if malware else pd.DataFrame(self.load_local_malware_data())
        malware_count = len(malware_df[malware_df['prediction'] == 'Malicious']) if not malware_df.empty else 0

        # Calculate vulnerability stats
        vuln_df = pd.DataFrame(vulnerabilities) if vulnerabilities else pd.DataFrame(self.load_local_vulnerability_data())
        critical_vuln = len(vuln_df[vuln_df['severity'] == 'Critical']) if not vuln_df.empty else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card"><h3>Total Threats</h3><h1>{len(threats)}</h1></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card"><h3>Critical</h3><h1 class="threat-critical">{critical}</h1></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card"><h3>Malware</h3><h1 class="threat-moderate">{malware_count}</h1></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card"><h3>Critical Vulns</h3><h1 class="threat-critical">{critical_vuln}</h1></div>""", unsafe_allow_html=True)

    def render_threat_timeline(self):
        df = pd.DataFrame(self.data["threats"]) if self.data["threats"] else self.load_local_intrusion_data()
        if df.empty:
            st.warning("No threat data available")
            return

        df['hour'] = df['timestamp'].dt.floor('H')
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
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="white",
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
        st.header("Malware Analysis")
        
        malware_data = self.data["malware"] if self.data["malware"] else self.load_local_malware_data()
        if not malware_data:
            st.warning("No malware data available")
            return

        df = pd.DataFrame(malware_data)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Malware Statistics")
            st.metric("Total Samples", len(df))
            st.metric("Malicious Samples", len(df[df['prediction'] == 'Malicious']))
            
            # Malware type distribution
            type_counts = df[df['prediction'] == 'Malicious']['type'].value_counts()
            fig = px.pie(
                type_counts,
                names=type_counts.index,
                values=type_counts.values,
                title="Malware Type Distribution"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color="white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Malware Visualization")
            selected = st.selectbox("Select Malware Sample", df['id'].tolist())
            sample = df[df['id'] == selected].iloc[0]
            
            st.write(f"**Type:** {sample['type']}")
            st.write(f"**Prediction:** {sample['prediction']}")
            st.write(f"**Score:** {sample['score']:.2f}")
            st.write(f"**Size:** {sample['size']:,} bytes")
            st.write(f"**Entropy:** {sample['entropy']:.2f}")
            
            # Display malware image
            if 'image' in sample:
                st.write("**Binary Visualization:**")
                img_bytes = base64.b64decode(sample['image'])
                img = Image.open(io.BytesIO(img_bytes))
                st.image(img, caption=f"Malware Binary: {sample['id']}", width=300)
            else:
                st.warning("No visualization available for this sample")

    def render_vulnerability_assessment(self):
        st.header("Vulnerability Assessment")
        
        vuln_data = self.data["vulnerabilities"] if self.data["vulnerabilities"] else self.load_local_vulnerability_data()
        if not vuln_data:
            st.warning("No vulnerability data available")
            return

        df = pd.DataFrame(vuln_data)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Vulnerability Statistics")
            st.metric("Total Vulnerabilities", len(df))
            st.metric("Critical Vulnerabilities", len(df[df['severity'] == 'Critical']))
            
            # Severity distribution
            severity_counts = df['severity'].value_counts()
            fig = px.bar(
                severity_counts,
                x=severity_counts.index,
                y=severity_counts.values,
                title="Vulnerability Severity Distribution",
                color=severity_counts.index,
                color_discrete_map={
                    'Critical': '#ff4b4b',
                    'High': '#f39c12',
                    'Medium': '#f1c40f',
                    'Low': '#2ecc71'
                }
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color="white",
                xaxis_title="Severity",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Vulnerability Details")
            
            # Configure AgGrid for vulnerabilities
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_default_column(
                filterable=True,
                sortable=True,
                resizable=True
            )
            
            gb.configure_column(
                "severity",
                cellStyle="""params => {
                    if (params.value === 'Critical') {
                        return {color: 'white', backgroundColor: '#ff4b4b'};
                    } else if (params.value === 'High') {
                        return {color: 'white', backgroundColor: '#f39c12'};
                    } else if (params.value === 'Medium') {
                        return {color: 'white', backgroundColor: '#f1c40f'};
                    } else {
                        return {color: 'white', backgroundColor: '#2ecc71'};
                    }
                }"""
            )
            
            grid_options = gb.build()
            
            AgGrid(
                df,
                gridOptions=grid_options,
                height=500,
                theme="streamlit",
                fit_columns_on_grid_load=False,
                update_mode='MODEL_CHANGED'
            )

    def render_model_interpretability(self):
        st.header("Model Interpretability")
        
        # Get SHAP values from API or generate sample
        shap_data = self.data.get("shap_values", {})
        
        if not shap_data:
            st.info("Generating sample SHAP values...")
            df = self.load_local_intrusion_data()
            if df.empty:
                st.warning("No data available for interpretability analysis")
                return
            
            # Generate sample SHAP values for demonstration
            features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                      'Flow Bytes/s', 'Flow Packets/s', 'Packet Length Mean']
            
            # Filter only numeric columns for SHAP
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            sample_data = df[numeric_cols].sample(min(100, len(df)))
            
            # Create a mock explainer
            background = sample_data.iloc[:50]
            explainer = shap.Explainer(
                lambda x: np.random.rand(len(x)),  # Mock prediction function
                background
            )
            
            shap_values = explainer(sample_data)
            shap_data = {
                "values": shap_values.values.tolist(),
                "features": sample_data.columns.tolist(),
                "data": sample_data.values.tolist()
            }
        
        st.subheader("Feature Importance")
        
        # Global feature importance
        fig, ax = plt.subplots()
        shap.summary_plot(
            np.array(shap_data["values"]),
            features=np.array(shap_data["data"]),
            feature_names=shap_data["features"],
            plot_type="bar",
            show=False
        )
        plt.gcf().set_facecolor('none')
        ax.set_facecolor('none')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        st.pyplot(fig)
        
        st.subheader("Individual Threat Explanation")
        
        # Select a sample to explain
        df = pd.DataFrame(self.data["threats"]) if self.data["threats"] else self.load_local_intrusion_data()
        if not df.empty:
            selected = st.selectbox("Select Threat to Explain", df.index.tolist())
            sample = df.iloc[selected]
            
            # Show SHAP force plot
            st.write(f"**Threat ID:** {selected} | **Type:** {sample.get('type', 'Unknown')} | **Score:** {sample.get('raw_score', 0):.2f}")
            
            # Create force plot
            force_fig, ax = plt.subplots()
            shap.plots._waterfall.waterfall_legacy(
                expected_value=0,
                #shap_values=np.random.rand(len(shap_data["features"]) - 0.5,  # Mock values
                shap_values = np.random.rand(len(shap_data["features"])))

                #features=sample[shap_data["features"]].values,
                #feature_names=shap_data["features"],
                #show=False
            #))
            plt.gcf().set_facecolor('none')
            ax.set_facecolor('none')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
            st.pyplot(force_fig)
        else:
            st.warning("No threat data available for individual explanations")

    def render_system_health(self):
        health = self.data["health"]
        model = self.data["model_info"]

        if not health:
            st.warning("No health data available")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Service Status")
            status = health.get("status", "unknown")
            st.metric("Overall Status", status,
                      "✅ Healthy" if status == "healthy" else "❌ Unhealthy")

            st.subheader("Components")
            services = health.get("services", {})
            for service, status in services.items():
                st.metric(service.replace("_", " ").title(), "✔️" if status else "❌", "Online" if status else "Offline")

        with col2:
            st.subheader("Model Information")
            if model:
                st.write(f"**Type:** {model.get('model_type', 'N/A')}")
                st.write(f"**Version:** {model.get('version', 'N/A')}")
                st.write(f"**Last Updated:** {model.get('last_updated', 'N/A')}")
                st.write(f"**Input Features:** {model.get('input_dimensions', 'N/A')}")
                
                # Model architecture visualization
                st.write("**Architecture:**")
                arch_img = Image.open("assets/model_arch.png") if os.path.exists("assets/model_arch.png") else None
                if arch_img:
                    st.image(arch_img, caption="Multi-Modal Deep Learning Architecture", use_column_width=True)
                else:
                    st.warning("Model architecture diagram not available")
            else:
                st.warning("No model information available")

            st.subheader("Performance")
            st.write(f"**Last Update:** {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

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
            critical_threats = [t for t in self.data["threats"] if t.get("raw_score", 0) > 0.7]
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

if __name__ == "__main__":
    dashboard = ThreatDashboard()
    dashboard.run()