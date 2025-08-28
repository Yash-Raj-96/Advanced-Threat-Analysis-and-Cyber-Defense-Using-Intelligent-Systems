import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Type
from pydantic import BaseModel, ValidationError, validator
from datetime import datetime
import json
import os
from datetime import datetime


logger = logging.getLogger(__name__)

class DataValidator:
    """
    Comprehensive data validation utility for cybersecurity datasets with:
    - Schema validation
    - Data quality checks
    - Type conversion
    - Anomaly detection
    - Custom validation rules
    """
    def normalize_network_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure correct types for protocol, label, and timestamp
        """
        if 'protocol' in df.columns:
            df['protocol'] = df['protocol'].astype(str)

        if 'label' in df.columns:
            df['label'] = df['label'].astype(str)

        if 'timestamp' in df.columns:
            print("📅 Timestamp column sample BEFORE any change:", df['timestamp'].head())
            
            # Convert to string first if it's category
            df['timestamp'] = df['timestamp'].astype(str)
            
            # Now apply datetime conversion
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            print("✅ Timestamp dtype AFTER conversion:", df['timestamp'].dtype)

        return df

        

#    def normalize_malware_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename EMBER malware dataset columns to match schema
        """
#        print("📋 Malware DataFrame columns before normalization:", df.columns.tolist())
#        if 'sha256' in df.columns:
#            df = df.rename(columns={'sha256': 'hash'})
            
#        if 'feature_vector' in df.columns and 'features' not in df.columns:
#            df = df.rename(columns={'feature_vector': 'features'})
            
#            print("✅ Malware DataFrame columns after normalization:", df.columns.tolist())

#        return df

    def normalize_malware_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimized malware normalization:
        - Select only valid float features
        - Avoid memory overload
        - Rename & generate required schema
        """
        import gc

        print("🔍 Malware columns before normalization:", df.columns.tolist())

        # 1. Detect feature columns
        feature_cols = [col for col in df.columns if col.startswith("F")]
        print(f"🎯 Found {len(feature_cols)} feature columns")

        # 2. Limit number of features to avoid memory crash
        MAX_FEATURES = 2000
        if len(feature_cols) > MAX_FEATURES:
            print(f"⚠️ Capping feature columns to {MAX_FEATURES} to avoid memory error")
            feature_cols = feature_cols[:MAX_FEATURES]

        # 3. Drop unused columns to reduce memory
        keep_cols = feature_cols + ['Label'] if 'Label' in df.columns else feature_cols
        df = df[keep_cols].copy()

        # 4. Convert features to float32 in-place
        df[feature_cols] = df[feature_cols].astype(np.float32)

        # 5. Pack features into list of lists efficiently
        df['features'] = df[feature_cols].apply(lambda row: row.to_list(), axis=1)

        # 6. Rename Label to label (if needed)
        if 'Label' in df.columns:
            df = df.rename(columns={'Label': 'label'})
            df['label'] = df['label'].astype(int)

        # 7. Add dummy hashes
        df['hash'] = [f'dummy_{i}' for i in range(len(df))]

        # 8. Keep only hash, features, label
        df = df[['hash', 'features', 'label']]

        print("✅ Malware columns after normalization:", df.columns.tolist())

        # 9. Manual garbage collection to reclaim memory
        gc.collect()

        return df
        
    
    def normalize_cve_json(self, data: list) -> list:
        """
        Normalize CVE JSON fields to match schema
        """
        for item in data:
            item['cve_id'] = item.get('id') or item.get('cve_id')
            item['published_date'] = pd.to_datetime(item.get('published') or item.get('published_date'), errors='coerce')
            item['description'] = item.get('summary') or item.get('description')
            item['severity'] = float(item.get('cvss_score') or item.get('severity') or 0.0)
        return data
    # Common predefined schemas
    
    NETWORK_SCHEMA = {
        'required': ['timestamp', 'source_ip', 'destination_ip', 'protocol', 'label'],
        'types': {
            'timestamp': 'datetime',
            'source_ip': 'str',
            'destination_ip': 'str',
            'protocol': 'str',
            'label': 'str'
        }
    }
    
    MALWARE_SCHEMA = {
        'required': ['hash', 'features', 'label'],
        'types': {
            'hash': 'str',
            'features': 'list',
            'label': 'int'
        }
    }


    CVE_SCHEMA = {
        'required': ['cve_id', 'published_date', 'description', 'severity'],
        'types': {
            'cve_id': 'str',
            'published_date': 'datetime',
            'description': 'str',
            'severity': 'float'
        }
    }

    def __init__(self, strict_mode: bool = True):
        """
        Initialize the data validator
        
        Args:
            strict_mode: If True, raises exceptions on validation failures
                         If False, logs warnings but continues
        """
        self.strict_mode = strict_mode

    def validate_network_data(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate network traffic data against predefined schema
        
        Args:
            data: DataFrame containing network traffic data
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
    
        
        data = self.normalize_network_columns(data)
        return self._validate_dataframe(data, self.NETWORK_SCHEMA, 'network')

#    def validate_malware_data(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate malware data against predefined schema
        
        Args:
            data: DataFrame containing malware features and labels
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
#        data = self.normalize_malware_columns(data)
#        return self._validate_dataframe(data, self.MALWARE_SCHEMA, 'malware')
    
    def validate_malware_data(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        df = self.normalize_malware_columns(df)  # 🔥 This must be here
        return self._validate_dataframe(df, self.MALWARE_SCHEMA, 'malware')


    def validate_cve_data(self, data: List[Dict]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate CVE vulnerability data against predefined schema
        
        Args:
            data: List of dictionaries containing CVE items
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        data = self.normalize_cve_json(data)
        return self._validate_json(data, self.CVE_SCHEMA, 'cve')

    def _validate_dataframe(self, 
                            df: pd.DataFrame, 
                            schema: Dict, 
                            data_type: str) -> Tuple[bool, Dict[str, Any]]:
            """
            Internal method to validate DataFrame against schema
            
            Args:
                df: DataFrame to validate
                schema: Validation schema
                data_type: Type of data for reporting
                
            Returns:
                Tuple of (is_valid, validation_report)
            """
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            report = {
                'data_type': data_type,
                'timestamp': datetime.now().isoformat(),
                'rows_checked': len(df),
                'missing_columns': [],
                'type_violations': {},
                'empty_values': {},
                'anomalies': {},
                'is_valid': True
            }
           

            try:
                # Check for missing columns
                missing_cols = [col for col in schema['required'] if col not in df.columns]
                if missing_cols:
                    report['missing_columns'] = missing_cols
                    report['is_valid'] = False
                    msg = f"Missing required columns in {data_type} data: {missing_cols}"
                    logger.error(msg)
                    self._handle_validation_failure(msg)

                # Check data types
                type_violations = {}
                for col, expected_type in schema['types'].items():
                    if col not in df.columns:
                        continue

                    col_series = df[col].dropna()
                    if expected_type == 'datetime':
                        if not pd.api.types.is_datetime64_any_dtype(col_series):
                            type_violations[col] = 'Not datetime'
                    elif expected_type == 'str':
                        if not pd.api.types.is_string_dtype(col_series):
                            type_violations[col] = f"Actual type: {df[col].dtype}"
                    elif expected_type == 'int':
                        if not pd.api.types.is_integer_dtype(col_series):
                            type_violations[col] = f"Actual type: {df[col].dtype}"
                    elif expected_type == 'float':
                        if not pd.api.types.is_float_dtype(col_series):
                            type_violations[col] = f"Actual type: {df[col].dtype}"
                    elif expected_type == 'list':
                        if not all(isinstance(x, list) for x in col_series):
                            type_violations[col] = "Not all elements are lists"

                if type_violations:
                    report['type_violations'] = type_violations
                    report['is_valid'] = False
                    msg = f"Type violations in {data_type} data: {type_violations}"
                    logger.error(msg)
                    self._handle_validation_failure(msg)

                # Check for empty values in required columns
                empty_values = {}
                for col in schema['required']:
                    if col in df.columns and df[col].isnull().any():
                        empty_values[col] = int(df[col].isnull().sum())

                if empty_values:
                    report['empty_values'] = empty_values
                    report['is_valid'] = False
                    msg = f"Empty values found in {data_type} data: {empty_values}"
                    logger.error(msg)
                    self._handle_validation_failure(msg)

                # Additional data-specific checks
                if data_type == 'network':
                    anomaly_report = self._check_network_anomalies(df)
                    if anomaly_report:
                        report['anomalies'] = anomaly_report
                        report['is_valid'] = False

                elif data_type == 'malware':
                    anomaly_report = self._check_malware_anomalies(df)
                    if anomaly_report:
                        report['anomalies'] = anomaly_report
                        report['is_valid'] = False

                return report['is_valid'], report

            except Exception as e:
                logger.error(f"Validation error for {data_type} data: {str(e)}", exc_info=True)
                report['error'] = str(e)
                report['is_valid'] = False
                return False, report

    def _validate_json(self, 
                    data: List[Dict], 
                    schema: Dict, 
                    data_type: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Internal method to validate JSON data against schema
        
        Args:
            data: List of dictionaries to validate
            schema: Validation schema
            data_type: Type of data for reporting
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            'data_type': data_type,
            'timestamp': datetime.now().isoformat(),
            'items_checked': len(data),
            'missing_fields': [],
            'type_violations': {},
            'invalid_items': [],
            'is_valid': True
        }

        try:
            for i, item in enumerate(data):
                item_report = {}

                # Check for missing fields
                missing_fields = [field for field in schema['required'] if field not in item]
                if missing_fields:
                    item_report['missing_fields'] = missing_fields
                    report['is_valid'] = False

                # Check data types
                type_violations = {}
                for field, expected_type in schema['types'].items():
                    if field not in item:
                        continue

                    value = item[field]
                    if expected_type == 'datetime':
                        if not self._is_valid_datetime(value):
                            type_violations[field] = f"Invalid datetime: {value}"
                    elif expected_type == 'str':
                        if not isinstance(value, str):
                            type_violations[field] = f"Actual type: {type(value)}"
                    elif expected_type == 'int':
                        if not isinstance(value, int):
                            type_violations[field] = f"Actual type: {type(value)}"
                    elif expected_type == 'float':
                        if not isinstance(value, (float, int)):
                            type_violations[field] = f"Actual type: {type(value)}"

                if type_violations:
                    item_report['type_violations'] = type_violations
                    report['is_valid'] = False

                if item_report:
                    report['invalid_items'].append({
                        'item_index': i,
                        'issues': item_report
                    })

            if not report['is_valid']:
                msg = f"Validation failed for {data_type} data. See report for details."
                logger.error(msg)
                self._handle_validation_failure(msg)

            return report['is_valid'], report

        except Exception as e:
            logger.error(f"Validation error for {data_type} data: {str(e)}", exc_info=True)
            report['error'] = str(e)
            report['is_valid'] = False
            return False, report


    def _check_network_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for anomalies in network data"""
        anomalies = {}
        
        # Check for invalid IP addresses
        ip_cols = [col for col in df.columns if '_ip' in col]
        invalid_ips = {}
        for col in ip_cols:
            invalid = df[~df[col].str.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', na=False)]
            if not invalid.empty:
                invalid_ips[col] = len(invalid)
        if invalid_ips:
            anomalies['invalid_ips'] = invalid_ips
        
        # Check for unusual protocols
        if 'protocol' in df.columns:
            protocol_counts = df['protocol'].value_counts()
            if len(protocol_counts) > 20:  # Too many unique protocols
                anomalies['unusual_protocols'] = {
                    'unique_count': len(protocol_counts),
                    'most_common': protocol_counts.head(5).to_dict()
                }
        
        return anomalies

    def _check_malware_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for anomalies in malware data"""
        anomalies = {}
        
        # Check for duplicate hashes
        if 'hash' in df.columns:
            duplicates = df['hash'].duplicated().sum()
            if duplicates > 0:
                anomalies['duplicate_hashes'] = duplicates
        
        # Check feature distributions
        if 'features' in df.columns:
            feature_lengths = df['features'].apply(len)
            if feature_lengths.nunique() > 1:
                anomalies['inconsistent_feature_lengths'] = {
                    'min': int(feature_lengths.min()),
                    'max': int(feature_lengths.max()),
                    'std': float(feature_lengths.std())
                }
        
        return anomalies

    def _is_valid_datetime(self, value: Any) -> bool:
        """Check if value is a valid datetime"""
        if isinstance(value, (datetime, pd.Timestamp)):
            return True
        if isinstance(value, str):
            try:
                pd.to_datetime(value)
                return True
            except:
                return False
        return False


    def _handle_validation_failure(self, message: str) -> None:
        """
        Handle validation failures based on strict mode setting.

        If strict_mode is True, raises a ValueError. Otherwise, logs a warning.
        """
        if self.strict_mode:
            logger.error(f"Validation failed (strict mode): {message}")
            raise ValueError(message)
        else:
            logger.warning(f"Validation warning: {message}")

    def validate_with_schema(self, 
                            data: Union[pd.DataFrame, List[Dict]], 
                            schema: Union[Dict[str, Any], Type[BaseModel]]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate data using either a dict-based schema or a Pydantic model.

        Args:
            data: The data to validate (either a DataFrame or list of dicts).
            schema: A dictionary schema or Pydantic BaseModel class.

        Returns:
            A tuple of (is_valid: bool, validation_report: dict).
        """
        try:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                if not isinstance(data, list):
                    raise ValueError("Pydantic schema requires list of dictionaries as input data.")
                return self._validate_with_pydantic(data, schema)

            elif isinstance(schema, dict):
                if isinstance(data, pd.DataFrame):
                    return self._validate_dataframe(data, schema, 'custom')
                elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    return self._validate_json(data, schema, 'custom')
                else:
                    raise ValueError("Dictionary schema requires either a DataFrame or list of dictionaries.")

            else:
                raise TypeError("Schema must be a dictionary or a Pydantic BaseModel class.")

        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}", exc_info=True)
            return False, {'error': str(e), 'is_valid': False}


    def _validate_with_pydantic(self, 
                              data: Union[pd.DataFrame, List[Dict]], 
                              model: BaseModel) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate data using Pydantic model
        
        Args:
            data: Data to validate
            model: Pydantic BaseModel class
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            'data_type': model.__name__,
            'timestamp': datetime.now().isoformat(),
            'valid_items': 0,
            'invalid_items': [],
            'is_valid': True
        }

        try:
            if isinstance(data, pd.DataFrame):
                data = data.to_dict('records')

            for i, item in enumerate(data):
                try:
                    model.parse_obj(item)
                    report['valid_items'] += 1
                except ValidationError as e:
                    report['invalid_items'].append({
                        'item_index': i,
                        'errors': json.loads(e.json())
                    })
                    report['is_valid'] = False

            if not report['is_valid']:
                msg = f"Pydantic validation failed for {model.__name__}. {len(report['invalid_items'])} invalid items."
                self._handle_validation_failure(msg)

            return report['is_valid'], report

        except Exception as e:
            logger.error(f"Pydantic validation error: {str(e)}", exc_info=True)
            report['error'] = str(e)
            report['is_valid'] = False
            return False, report



    def save_results_to_csv(df: pd.DataFrame, result_type: str, output_dir: str = "outputs/analysis_results"):
        """
        Save analysis results to a timestamped CSV file.

        Args:
            df (pd.DataFrame): The results DataFrame to save.
            result_type (str): One of 'network', 'malware', or 'cve'.
            output_dir (str): Path to the output directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result_type}_results_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)

        df.to_csv(filepath, index=False)
        print(f"[✔] Saved {result_type} results to {filepath}")

    @staticmethod
    def generate_schema_from_data(data: Union[pd.DataFrame, List[Dict]]) -> Dict:
        """
        Generate a validation schema from sample data
        
        Args:
            data: Sample data to analyze
            
        Returns:
            Dictionary representing inferred schema
        """
        if isinstance(data, pd.DataFrame):
            schema = {
                'required': list(data.columns),
                'types': {
                    col: DataValidator._infer_pandas_type(data[col].dtype)
                    for col in data.columns
                }
            }
        else:
            if not data:
                return {'required': [], 'types': {}}
            
            sample = data[0]
            schema = {
                'required': list(sample.keys()),
                'types': {
                    field: DataValidator._infer_python_type(value)
                    for field, value in sample.items()
                }
            }
        
        return schema

    @staticmethod
    def _infer_pandas_type(dtype) -> str:
        """Infer schema type from pandas dtype"""
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return 'datetime'
        elif pd.api.types.is_string_dtype(dtype):
            return 'str'
        elif pd.api.types.is_integer_dtype(dtype):
            return 'int'
        elif pd.api.types.is_float_dtype(dtype):
            return 'float'
        elif pd.api.types.is_bool_dtype(dtype):
            return 'bool'
        else:
            return 'object'

    @staticmethod
    def _infer_python_type(value) -> str:
        """Infer schema type from Python value"""
        if isinstance(value, str):
            return 'str'
        elif isinstance(value, bool):
            return 'bool'
        elif isinstance(value, int):
            return 'int'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, (datetime, pd.Timestamp)):
            return 'datetime'
        elif isinstance(value, list):
            return 'list'
        elif isinstance(value, dict):
            return 'dict'
        else:
            return 'object'