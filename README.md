# 🔒 Advanced Threat Analysis and Cyber Defense Using Intelligent Systems

This repository contains the implementation of a **unified cyber defense framework** that integrates **intrusion detection, malware analysis, and vulnerability assessment** into a single system using **multi-modal deep learning** and **explainable AI (XAI)**.

---

## 🚀 Features
- **Unified Threat Detection**
  - Intrusion Detection (CIC-IDS2017)
  - Malware Analysis (EMBER)
  - Vulnerability Assessment (NVD CVEs)
- **Multi-Modal Deep Learning**
  - LSTM for network intrusion data
  - CNN for malware binaries
  - Transformer for vulnerability text reports
- **Explainability**
  - SHAP & LIME-based interpretations for security analysts
- **End-to-End Pipeline**
  - Data loading, preprocessing, training, evaluation, and threat analysis
- **System Monitoring**
  - MLflow for experiment tracking
  - Performance monitoring & logging
- **Interactive Dashboard**
  - Streamlit-based frontend for real-time visualization
  - FastAPI backend for threat predictions

---

## 📂 Project Structure
```bash
/cyber_defense_system
│
├── backend/
│   ├── data_processing/       # Data loading & preprocessing
│   ├── models/                # CNN, LSTM, Transformer & trainer
│   ├── threat_analysis/       # IDS, malware, vulnerability modules
│   ├── utils/                 # Logging, validation, interpretability
│   ├── api/                   # FastAPI routes
│   ├── config.py              # Configurations
│   └── main.py                # AdvancedThreatAnalyzer (main pipeline)
│
├── frontend/
│   ├── static/                # CSS/JS files
│   ├── templates/             # HTML templates
│   └── app.py                 # Streamlit dashboard
│
├── tests/                     # Unit & integration tests
├── docs/                      # Architecture & API specs
├── data/                      # Raw & processed datasets
├── requirements.txt           # Dependencies
└── README.md
````

---

## ⚡ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/cyber-defense-system.git
   cd cyber-defense-system
   ```

2. **Create a virtual environment & install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   ```bash
   cp .env.example .env
   ```

4. **Run the backend (FastAPI)**

   ```bash
   uvicorn backend.main:app --reload
   ```

5. **Run the frontend (Streamlit dashboard)**

   ```bash
   streamlit run frontend/app.py
   ```

---

## 📊 Usage

* **Training & Threat Analysis**

  ```bash
  python backend/main.py
  ```

* **Interactive Dashboard**
  Access at: [http://localhost:8501](http://localhost:8501)

* **FastAPI Endpoints**
  Access API docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📖 Publication

This project is submitted to:
**9th International Conference on Computational System and Information Technology for Sustainable Solutions (CSITSS-2025), RV College of Engineering, Bengaluru.**

---

## 👨‍💻 Author

**Yashvanth P S**
M.Tech Project – *Advanced Threat Analysis and Cyber Defense Using Intelligent Systems*

---

## ⭐ Acknowledgements

* CIC-IDS2017 Dataset
* EMBER Malware Dataset
* NVD CVE Database
* PyTorch, FastAPI, Streamlit, MLflow

---

