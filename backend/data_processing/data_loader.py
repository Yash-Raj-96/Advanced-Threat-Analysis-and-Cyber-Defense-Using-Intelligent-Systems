import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Union, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
import pyarrow as pa
import ast
from backend.config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    """High-performance data loader with parallel processing and optimized memory usage"""

    def _validate_file_path(self, path: Union[str, Path]) -> Path:
        """Ensure path exists and is a file"""
        path = Path(path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Invalid file path: {path}")
        return path

    def __init__(self):
        self.config = Config()
        self._cache = {}
        self._init_column_mappings()
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info("Optimized DataLoader initialized")

    def _init_column_mappings(self):
        """Predefined column mappings for fast lookup"""
        self.network_column_map = {
            'src ip': 'source_ip', 'srcip': 'source_ip', 'ip_src': 'source_ip',
            'sip': 'source_ip', 'srcaddr': 'source_ip', 'src_address': 'source_ip',
            
            'dst ip': 'destination_ip', 'dstip': 'destination_ip', 'ip_dst': 'destination_ip',
            'dip': 'destination_ip', 'dstaddr': 'destination_ip', 'dst_address': 'destination_ip',

            'start_time': 'timestamp', 'time': 'timestamp', 'date_time': 'timestamp',
            'datetime': 'timestamp', 'endtime': 'timestamp', 'date': 'timestamp',
            
            'attack_type': 'label', 'attack_cat': 'label', 'class': 'label',
            'target': 'label', 'is_attack': 'label', 'binarylabel': 'label'
        }

    def _fast_normalize_columns(self, df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
        cols = df.columns.str.lower().str.strip()
        df.columns = [column_map.get(col, col) for col in cols]
        return df

    def _parallel_load_network_files(self, files: List[Path]) -> List[pd.DataFrame]:
        def load_file(file: Path):
            try:
                df = pd.read_csv(
                    file,
                    engine='c',
                    low_memory=False
                )
                df = self._fast_normalize_columns(df, self.network_column_map)
                return self._clean_network_data(df)
            except Exception as e:
                logger.warning(f"Skipping {file.name}: {str(e)}")
                return None

        results = list(self.executor.map(load_file, files))
        return [df for df in results if df is not None]

    def _clean_network_data(self, df: pd.DataFrame) -> pd.DataFrame:
        ip_cols = [c for c in ['source_ip', 'dest_ip'] if c in df.columns]
        for col in ip_cols:
            df[col] = df[col].astype(str).str.strip().str.extract(r'(\d+\.\d+\.\d+\.\d+)')[0]

        if 'label' in df.columns:
            df['label'] = (
                df['label']
                .astype(str)
                .str.lower()
                .str.strip()
                .map({'normal': 0, 'attack': 1, 'benign': 0, '0': 0, '1': 1})
                .fillna(0)
                .astype(np.int8)
            )

        return df.dropna(subset=ip_cols) if ip_cols else df

    def load_network_data(self, use_cache=True) -> pd.DataFrame:
        cache_key = "network_data"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        files = list(self.config.NETWORK_LOGS_DIR.glob('*.csv'))
        if not files:
            raise FileNotFoundError(f"No CSV files in {self.config.NETWORK_LOGS_DIR}")

        dfs = self._parallel_load_network_files(files)
        if not dfs:
            raise ValueError("No valid network files processed")

        combined = pd.concat(dfs, ignore_index=True)

        for col in combined.select_dtypes(include=['object']):
            combined[col] = combined[col].astype('category')

        self._cache[cache_key] = combined
        logger.info(f"Loaded network data: {combined.shape}")
        return combined

    def load_malware_data(self, use_cache=True) -> pd.DataFrame:
        cache_key = "malware_data"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            table = pq.read_table(self.config.MALWARE_DATA)
            df = table.to_pandas()

            df.rename(columns={
                'sha256': 'hash',
                'feature_vector': 'features',
                'class': 'label'
            }, inplace=True)

            if 'features' in df.columns and df['features'].dtype == 'object':
                df['features'] = df['features'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

            if 'label' in df.columns:
                df['label'] = df['label'].astype(np.int8)

            for col in df.select_dtypes(include=['float']):
                df[col] = pd.to_numeric(df[col], downcast='float')

            self._cache[cache_key] = df
            return df

        except Exception as e:
            logger.error(f"Malware data load failed: {str(e)}")
            return pd.DataFrame()

    def load_cve_data(self, file_path=None) -> List[Dict]:
        cache_key = "cve_data"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._validate_file_path(file_path or self.config.CVE_DATA)

        try:
            import rapidjson
            with open(path, 'r', encoding='utf-8') as f:
                data = rapidjson.load(f)
        except ImportError:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        cve_items = []
        for item in data.get('CVE_Items', data.get('vulnerabilities', [])):
            cve = item.get('cve', item)
            cve_items.append({
                'id': cve.get('CVE_data_meta', {}).get('ID', ''),
                'published': cve.get('publishedDate', ''),
                'description': cve.get('description', {}).get('description_data', [{}])[0].get('value', ''),
                'severity': float(cve.get('impact', {}).get('baseMetricV3', {}).get('cvssV3', {}).get('baseScore', 0))
            })

        self._cache[cache_key] = cve_items
        return cve_items

    def load_all_data(self, use_cache=True) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
        network_future = self.executor.submit(self.load_network_data, use_cache)
        malware_future = self.executor.submit(self.load_malware_data, use_cache)
        cve_future = self.executor.submit(self.load_cve_data)

        return (
            network_future.result(),
            malware_future.result(),
            cve_future.result()
        )

    def __del__(self):
        self.executor.shutdown(wait=False)
