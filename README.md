## 📂 Project Structure

/cyber_defense_system-1-copy
│
├── backend/
│ ├── data_processing/
│ │ ├── data_loader.py
│ │ ├── preprocessor.py
│ │ ├── feature_extractor.py
│ │ ├── prepare_cic.py
│ │ ├── prepare_ember.py
│ │ └── prepare_nvd.py
│ │
│ ├── models/
│ │ ├── multi_modal_model.py
│ │ ├── cnn_module.py
│ │ ├── lstm_module.py
│ │ ├── transformer_module.py
│ │ ├── joint_module.py
│ │ └── model_trainer.py
│ │
│ ├── threat_analysis/
│ │ ├── pipeline.py
│ │ ├── intrusion_detector.py
│ │ ├── malware_analyzer.py
│ │ ├── vulnerability_assessor.py
│ │ ├── validation.py
│ │ └── visuals.py
│ │
│ ├── utils/
│ │ ├── interpretability.py
│ │ ├── logger.py
│ │ ├── config_manager.py
│ │ ├── data_validator.py
│ │ └── performance_monitor.py
│ │
│ ├── api/
│ │ └── fastapi_app.py
│ │
│ ├── config.py
│ ├── app_core.py
│ └── main.py
│
├── frontend/
│ ├── static/
│ ├── templates/
│ │ ├── index.html
│ │ ├── dashboard.html
│ │ └── threat_details.html
│ └── app.py
│
├── tests/
│ ├── test_data_processing.py
│ ├── test_models.py
│ └── test_threat_analysis.py
│
├── docs/
│ ├── architecture.md
│ └── api_spec.md
│
├── requirements.txt
├── README.md
├── .env.example
│
└── data/
├── raw/
│ ├── cve/
│ ├── malware/
│ └── network_logs/
│
└── processed/
