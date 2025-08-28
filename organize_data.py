import os
import shutil
import json
import hashlib
from pathlib import Path
import pandas as pd
import joblib

class DataOrganizer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.config = {
            'paths': {},
            'file_checksums': {}
        }
        
    def create_directory_structure(self):
        """Create the standard directory structure for the project"""
        dirs = [
            'data/raw/cve',
            'data/raw/malware',
            'data/raw/network_logs',
            'data/processed',
            'data/temp',
            'models/preprocessors',
            'models/scalers',
            'models/encoders'
        ]
        
        for dir_path in dirs:
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {full_path}")

    def move_and_validate_files(self):
        """Organize files into the proper directory structure"""
        file_mapping = {
            # Raw data files
            'cic_ids2017.csv': 'data/raw/network_logs/cic_ids2017.csv',
            'nvdove-2.0-2025.json': 'data/raw/cve/nvdove-2.0-2025.json',
            'train_ember_2018_v2_features.parquet': 'data/raw/malware/train_ember_2018_v2_features.parquet',
            'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv': 'data/raw/network_logs/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            
            # Processed data files
            'nvd_processed.csv': 'data/processed/nvd_processed.csv',
            'x_train.parquet': 'data/processed/x_train.parquet',
            'x_test.parquet': 'data/processed/x_test.parquet',
            'y_train.parquet': 'data/processed/y_train.parquet',
            'y_test.parquet': 'data/processed/y_test.parquet',
            
            # Preprocessor files
            'network_preprocessor.joblib': 'models/preprocessors/network_preprocessor.joblib',
            'cve_imputer.joblib': 'models/preprocessors/cve_imputer.joblib',
            'malware_imputer.joblib': 'models/preprocessors/malware_imputer.joblib',
            
            # Scaler files
            'cve_scaler.joblib': 'models/scalers/cve_scaler.joblib',
            'malware_scaler.joblib': 'models/scalers/malware_scaler.joblib',
            'ember_scaler.pkl': 'models/scalers/ember_scaler.pkl',
            'ember_scaler_stats.csv': 'models/scalers/ember_scaler_stats.csv',
            
            # Encoder files
            'label_encoder.joblib': 'models/encoders/label_encoder.joblib',
            
            # Vectorizer files
            'nvd_vectorizer.pkl': 'models/vectorizers/nvd_vectorizer.pkl'
        }

        # Create additional directories needed for file mapping
        additional_dirs = {'models/vectorizers'}
        for dir_path in additional_dirs:
            (self.base_path / dir_path).mkdir(parents=True, exist_ok=True)

        # Move files and record checksums
        for src_file, dest_path in file_mapping.items():
            src_path = self.base_path / src_file
            dest_full_path = self.base_path / dest_path
            
            if src_path.exists():
                # Calculate checksum before moving
                checksum = self.calculate_checksum(src_path)
                
                # Move the file
                shutil.move(str(src_path), str(dest_full_path))
                print(f"Moved: {src_file} -> {dest_path}")
                
                # Store path and checksum in config
                self.config['paths'][src_file] = dest_path
                self.config['file_checksums'][dest_path] = checksum
            else:
                print(f"Warning: Source file not found - {src_file}")

    def calculate_checksum(self, file_path, chunk_size=8192):
        """Calculate SHA256 checksum of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()

    def verify_file_integrity(self):
        """Verify that files haven't been corrupted"""
        for rel_path, expected_checksum in self.config['file_checksums'].items():
            file_path = self.base_path / rel_path
            if not file_path.exists():
                print(f"Error: File missing - {rel_path}")
                continue
                
            current_checksum = self.calculate_checksum(file_path)
            if current_checksum != expected_checksum:
                print(f"Warning: Checksum mismatch for {rel_path}")
            else:
                print(f"Verified: {rel_path}")

    def save_config(self):
        """Save the configuration to a JSON file"""
        config_path = self.base_path / 'data_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to {config_path}")

    def load_sample_data(self):
        """Load sample data to verify everything works"""
        try:
            # Load a small sample from each major dataset
            print("\nLoading sample data for verification:")
            
            # Network data
            network_path = self.base_path / self.config['paths']['cic_ids2017.csv']
            network_sample = pd.read_csv(network_path, nrows=5)
            print(f"Network data sample:\n{network_sample.head(1)}")
            
            # Malware data
            malware_path = self.base_path / self.config['paths']['train_ember_2018_v2_features.parquet']
            malware_sample = pd.read_parquet(malware_path, nrows=5)
            print(f"\nMalware data sample:\n{malware_sample.head(1)}")
            
            # CVE data
            cve_path = self.base_path / self.config['paths']['nvdove-2.0-2025.json']
            with open(cve_path, 'r') as f:
                cve_sample = json.load(f)
            print(f"\nCVE data sample (first item):\n{cve_sample['CVE_Items'][0]['cve']['CVE_data_meta']['ID']}")
            
            # Load a preprocessor to verify
            preprocessor_path = self.base_path / self.config['paths']['network_preprocessor.joblib']
            preprocessor = joblib.load(preprocessor_path)
            print(f"\nPreprocessor loaded successfully: {type(preprocessor).__name__}")
            
        except Exception as e:
            print(f"\nError during sample loading: {str(e)}")

if __name__ == "__main__":
    # Set this to your actual project path
    project_path = r"C:\Users\yashu\Downloads\cyber_defense_system-1 - Copy - Copy - Copy"
    
    organizer = DataOrganizer(project_path)
    
    print("Creating directory structure...")
    organizer.create_directory_structure()
    
    print("\nOrganizing files...")
    organizer.move_and_validate_files()
    
    print("\nVerifying file integrity...")
    organizer.verify_file_integrity()
    
    print("\nSaving configuration...")
    organizer.save_config()
    
    # Verify everything works
    organizer.load_sample_data()
    
    print("\nData organization complete!")