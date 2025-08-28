import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
import numpy as np
import psutil
import os
from datetime import datetime
from collections import defaultdict
import json
from typing import Union, Dict


logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Advanced performance monitoring system with resource tracking and reporting.
    
    Features:
    - Operation timing with context managers
    - Memory usage tracking
    - CPU utilization monitoring
    - Disk I/O tracking
    - Network usage monitoring (optional)
    - Comprehensive reporting
    - Threshold-based alerts
    """
    
    def __init__(self, track_memory: bool = True, track_cpu: bool = True):
        """
        Initialize the performance monitor.
        
        Args:
            track_memory: Whether to track memory usage
            track_cpu: Whether to track CPU utilization
        """
        self.metrics = defaultdict(dict)
        self.current_operation = None
        self.track_memory = track_memory
        self.track_cpu = track_cpu
        self._baseline_memory = None
        self._baseline_cpu = None
        
        # Initialize process tracker
        self.process = psutil.Process(os.getpid())
        
        # Set baseline measurements
        if self.track_memory:
            self._baseline_memory = self._get_memory_usage()
        if self.track_cpu:
            self._baseline_cpu = self._get_cpu_usage()

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB"""
        mem = self.process.memory_info()
        return {
            'rss_mb': mem.rss / (1024 ** 2),
            'vms_mb': mem.vms / (1024 ** 2),
            'percent': self.process.memory_percent()
        }

    def _get_cpu_usage(self) -> Dict[str, float]:
        """Get current CPU usage"""
        return {
            'process_percent': self.process.cpu_percent(interval=0.1),
            'system_percent': psutil.cpu_percent(interval=0.1)
        }

    def _get_disk_io(self) -> Dict[str, float]:
        """Get disk I/O statistics"""
        io = self.process.io_counters()
        return {
            'read_bytes': io.read_bytes,
            'write_bytes': io.write_bytes,
            'read_count': io.read_count,
            'write_count': io.write_count
        }

    @contextmanager
    def track(self, operation_name: str, alert_threshold: Optional[float] = None):
        """
        Context manager for timing operations and collecting performance metrics.
        
        Args:
            operation_name: Name of the operation being tracked
            alert_threshold: Optional time threshold in seconds to trigger alerts
            
        Example:
            with monitor.track("data_loading"):
                load_data()
        """
        if self.current_operation:
            logger.warning(f"Nested operation detected: {operation_name} within {self.current_operation}")

        # Store previous operation
        prev_operation = self.current_operation
        self.current_operation = operation_name
        
        # Get initial metrics
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage() if self.track_memory else None
        start_cpu = self._get_cpu_usage() if self.track_cpu else None
        start_disk = self._get_disk_io()
        
        try:
            yield
            
        except Exception as e:
            logger.error(f"Operation {operation_name} failed during monitoring: {str(e)}")
            raise
            
        finally:
            # Calculate duration
            duration = time.perf_counter() - start_time
            
            # Record metrics
            self.metrics[operation_name]['duration_sec'] = duration
            self.metrics[operation_name]['timestamp'] = datetime.now().isoformat()
            
            if self.track_memory:
                end_memory = self._get_memory_usage()
                self.metrics[operation_name]['memory'] = {
                    'start': start_memory,
                    'end': end_memory,
                    'delta': {
                        'rss_mb': end_memory['rss_mb'] - start_memory['rss_mb'],
                        'vms_mb': end_memory['vms_mb'] - start_memory['vms_mb'],
                        'percent_change': end_memory['percent'] - start_memory['percent']
                    }
                }
            
            if self.track_cpu:
                end_cpu = self._get_cpu_usage()
                self.metrics[operation_name]['cpu'] = {
                    'start': start_cpu,
                    'end': end_cpu,
                    'delta': {
                        'process_percent': end_cpu['process_percent'] - start_cpu['process_percent'],
                        'system_percent': end_cpu['system_percent'] - start_cpu['system_percent']
                    }
                }
            
            # Disk I/O metrics
            end_disk = self._get_disk_io()
            self.metrics[operation_name]['disk'] = {
                'start': start_disk,
                'end': end_disk,
                'delta': {
                    'read_bytes': end_disk['read_bytes'] - start_disk['read_bytes'],
                    'write_bytes': end_disk['write_bytes'] - start_disk['write_bytes'],
                    'read_count': end_disk['read_count'] - start_disk['read_count'],
                    'write_count': end_disk['write_count'] - start_disk['write_count']
                }
            }
            
            # Check threshold
            if alert_threshold and duration > alert_threshold:
                logger.warning(
                    f"⚠️ Performance alert: Operation '{operation_name}' took {duration:.2f}s "
                    f"(threshold: {alert_threshold:.2f}s)"
                )
            
            # Log basic metrics
            logger.info(
                f"⏱️ Operation '{operation_name}' completed in {duration:.3f}s | "
                f"Memory: +{self.metrics[operation_name].get('memory', {}).get('delta', {}).get('rss_mb', 0):.1f}MB"
            )
            
            # Restore previous operation
            self.current_operation = prev_operation

    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for a specific operation or all operations.
        
        Args:
            operation_name: Name of the operation to get metrics for
            
        Returns:
            Dictionary of performance metrics
        """
        if operation_name:
            return self.metrics.get(operation_name, {})
        return dict(self.metrics)

    def generate_report(self, format: str = 'dict') -> Union[Dict, str, pd.DataFrame]:
        """
        Generate a comprehensive performance report.
        
        Args:
            format: Output format ('dict', 'json', 'dataframe')
            
        Returns:
            Performance report in requested format
        """
        report = {
            'operations': dict(self.metrics),
            'system': {
                'baseline': {
                    'memory': self._baseline_memory,
                    'cpu': self._baseline_cpu
                },
                'current': {
                    'memory': self._get_memory_usage() if self.track_memory else None,
                    'cpu': self._get_cpu_usage() if self.track_cpu else None
                }
            },
            'summary': self._generate_summary()
        }
        
        if format == 'json':
            return json.dumps(report, indent=2)
        elif format == 'dataframe':
            return self._generate_dataframe(report)
        return report

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from collected metrics"""
        if not self.metrics:
            return {}
        
        durations = [op['duration_sec'] for op in self.metrics.values()]
        memory_deltas = [
            op.get('memory', {}).get('delta', {}).get('rss_mb', 0)
            for op in self.metrics.values()
        ]
        
        return {
            'total_operations': len(self.metrics),
            'total_time_sec': sum(durations),
            'avg_time_sec': np.mean(durations),
            'max_time_sec': max(durations),
            'min_time_sec': min(durations),
            'total_memory_growth_mb': sum(memory_deltas),
            'avg_memory_growth_mb': np.mean(memory_deltas),
            'max_memory_growth_mb': max(memory_deltas),
            'operations_by_duration': sorted(
                self.metrics.keys(),
                key=lambda k: self.metrics[k]['duration_sec'],
                reverse=True
            )
        }

    def _generate_dataframe(self, report: Dict) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame"""
        rows = []
        for op_name, metrics in report['operations'].items():
            row = {
                'operation': op_name,
                'duration_sec': metrics['duration_sec'],
                'memory_rss_mb_start': metrics.get('memory', {}).get('start', {}).get('rss_mb'),
                'memory_rss_mb_end': metrics.get('memory', {}).get('end', {}).get('rss_mb'),
                'cpu_process_percent_start': metrics.get('cpu', {}).get('start', {}).get('process_percent'),
                'cpu_process_percent_end': metrics.get('cpu', {}).get('end', {}).get('process_percent'),
                'disk_read_bytes': metrics.get('disk', {}).get('delta', {}).get('read_bytes'),
                'disk_write_bytes': metrics.get('disk', {}).get('delta', {}).get('write_bytes')
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df['duration_rank'] = df['duration_sec'].rank(ascending=False)
        return df

    def reset(self) -> None:
        """Reset all collected metrics"""
        self.metrics.clear()
        logger.info("Performance metrics reset")

    def get_current_operation(self) -> Optional[str]:
        """Get the name of the currently tracked operation"""
        return self.current_operation

    def log_summary(self) -> None:
        """Log a summary of performance metrics"""
        summary = self._generate_summary()
        if not summary:
            logger.info("No performance metrics collected yet")
            return
        
        logger.info("📊 Performance Summary:")
        logger.info(f"  Total operations: {summary['total_operations']}")
        logger.info(f"  Total time: {summary['total_time_sec']:.2f}s")
        logger.info(f"  Average operation time: {summary['avg_time_sec']:.2f}s")
        logger.info(f"  Longest operation: {summary['max_time_sec']:.2f}s ({summary['operations_by_duration'][0]})")
        logger.info(f"  Memory growth: {summary['total_memory_growth_mb']:.1f}MB total")