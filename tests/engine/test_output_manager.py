"""
Unit tests for OutputManager component.

Tests the centralized output management system that eliminates
duplicate file writes and provides consistent output handling.

Author: Senior Python Developer
Date: 2025
"""

import pytest
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open
from engine.output_manager import OutputManager, get_output_manager, reset_output_manager


class TestOutputManager:
    """Test suite for OutputManager component."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def output_config(self, temp_dir):
        """Create output configuration for testing."""
        return {
            'results_dir': str(temp_dir / 'results'),
            'reports_dir': str(temp_dir / 'reports'),
            'logs_dir': str(temp_dir / 'logs'),
            'data_dir': str(temp_dir / 'data'),
            'detailed_subdir': 'detailed',
            'validation_subdir': 'validation'
        }
    
    @pytest.fixture
    def output_manager(self, output_config):
        """Create OutputManager instance for testing."""
        return OutputManager(output_config)
    
    @pytest.fixture
    def sample_fills_df(self):
        """Create sample fills DataFrame for testing."""
        return pd.DataFrame({
            'timestamp': ['2025-01-01 10:00:00', '2025-01-01 11:00:00'],
            'symbol': ['TEST3', 'TEST4'],
            'side': ['BUY', 'SELL'],
            'quantity': [100, 200],
            'price': [25.50, 30.25],
            'order_type': ['MARKET', 'LIMIT']
        })
    
    def test_initialization_normal_mode(self, output_config):
        """Test OutputManager initialization in normal mode."""
        with patch('os.getenv', return_value='0'):  # Not audit mode
            om = OutputManager(output_config)
            
            assert om.config == output_config
            assert not om.audit_only
            assert isinstance(om._written_files, set)
    
    def test_initialization_audit_mode(self, output_config):
        """Test OutputManager initialization in audit mode."""
        with patch('os.getenv', return_value='1'):  # Audit mode
            om = OutputManager(output_config)
            
            assert om.audit_only
    
    def test_initialization_default_config(self):
        """Test OutputManager initialization with default config."""
        with patch('os.getenv', return_value='0'):
            om = OutputManager()
            
            assert 'results_dir' in om.config
            assert om.config['results_dir'] == 'results'
    
    def test_write_unified_fills_success(self, output_manager, sample_fills_df, temp_dir):
        """Test successful unified fills writing."""
        artifacts = output_manager.write_unified_fills(sample_fills_df)
        
        # Check return artifacts
        assert 'csv' in artifacts
        assert 'json' in artifacts
        
        # Check files were created
        csv_path = Path(artifacts['csv'])
        json_path = Path(artifacts['json'])
        
        assert csv_path.exists()
        assert json_path.exists()
        
        # Verify CSV content
        written_df = pd.read_csv(csv_path)
        assert len(written_df) == len(sample_fills_df)
        assert list(written_df.columns) == list(sample_fills_df.columns)
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        assert len(json_data) == len(sample_fills_df)
    
    def test_write_unified_fills_audit_mode(self, output_config, sample_fills_df):
        """Test unified fills writing in audit mode."""
        with patch('os.getenv', return_value='1'):  # Audit mode
            om = OutputManager(output_config)
            artifacts = om.write_unified_fills(sample_fills_df)
            
            # Should return empty artifacts in audit mode
            assert artifacts == {}
    
    def test_write_unified_fills_empty_data(self, output_manager):
        """Test unified fills writing with empty DataFrame."""
        empty_df = pd.DataFrame()
        artifacts = output_manager.write_unified_fills(empty_df)
        
        assert artifacts == {}
    
    def test_write_unified_fills_duplicate_prevention(self, output_manager, sample_fills_df, temp_dir):
        """Test that duplicate writes are prevented."""
        # First write
        artifacts1 = output_manager.write_unified_fills(sample_fills_df)
        assert 'csv' in artifacts1
        
        # Second write should be prevented
        artifacts2 = output_manager.write_unified_fills(sample_fills_df)
        assert artifacts2 == {}  # No new files written
    
    def test_write_signal_data(self, output_manager, temp_dir):
        """Test signal data writing."""
        signal_data = pd.DataFrame({
            'date': ['2025-01-01', '2025-01-02'],
            'symbol': ['TEST3', 'TEST3'],
            'signal': ['BUY', 'SELL'],
            'score': [1.8, -2.1]
        })
        
        filepath = output_manager.write_signal_data(signal_data)
        
        assert filepath is not None
        written_path = Path(filepath)
        assert written_path.exists()
        
        # Verify content
        written_df = pd.read_csv(written_path)
        assert len(written_df) == len(signal_data)
    
    def test_write_performance_report(self, output_manager, temp_dir):
        """Test performance report writing."""
        report_data = {
            'total_return': 0.15,
            'sharpe_ratio': 1.8,
            'max_drawdown': -0.05,
            'total_trades': 150
        }
        
        filepath = output_manager.write_performance_report(report_data)
        
        assert filepath is not None
        written_path = Path(filepath)
        assert written_path.exists()
        
        # Verify JSON content
        with open(written_path, 'r') as f:
            written_data = json.load(f)
        assert written_data == report_data
    
    def test_write_backtest_results(self, output_manager, temp_dir):
        """Test backtest results writing."""
        results_data = {
            'run_id': 'test_run_123',
            'strategy': 'fuzzy_fajuto',
            'performance': {'return': 0.2},
            'timestamp': '2025-01-01T10:00:00'
        }
        run_id = 'test_run_123'
        
        filepath = output_manager.write_backtest_results(results_data, run_id)
        
        assert filepath is not None
        written_path = Path(filepath)
        assert written_path.exists()
        assert run_id in written_path.name
        
        # Verify content
        with open(written_path, 'r') as f:
            written_data = json.load(f)
        assert written_data == results_data
    
    def test_write_validation_log(self, output_manager, temp_dir):
        """Test validation log writing."""
        log_entries = [
            'Test entry 1',
            'Test entry 2',
            'Test entry 3'
        ]
        
        filepath = output_manager.write_validation_log(log_entries)
        
        assert filepath is not None
        written_path = Path(filepath)
        assert written_path.exists()
        
        # Verify content
        with open(written_path, 'r') as f:
            content = f.read()
        
        for entry in log_entries:
            assert entry in content
    
    def test_get_output_path(self, output_manager):
        """Test output path generation."""
        # Test valid categories
        results_path = output_manager.get_output_path('results', 'test.csv')
        assert 'results' in str(results_path)
        assert 'test.csv' in str(results_path)
        
        reports_path = output_manager.get_output_path('reports', 'report.json')
        assert 'reports' in str(reports_path)
        
        detailed_path = output_manager.get_output_path('detailed', 'detail.json')
        assert 'detailed' in str(detailed_path)
        
        # Test invalid category
        with pytest.raises(ValueError, match="Unknown output category"):
            output_manager.get_output_path('invalid_category', 'test.txt')
    
    def test_atomic_write_success(self, output_manager, temp_dir):
        """Test atomic write context manager success case."""
        test_file = temp_dir / 'test_atomic.txt'
        test_content = 'Test atomic write content'
        
        with output_manager._atomic_write(test_file) as f:
            f.write(test_content)
        
        # File should exist and have correct content
        assert test_file.exists()
        with open(test_file, 'r') as f:
            assert f.read() == test_content
        
        # Temp file should not exist
        temp_file = test_file.with_suffix(test_file.suffix + '.tmp')
        assert not temp_file.exists()
    
    def test_atomic_write_failure(self, output_manager, temp_dir):
        """Test atomic write context manager failure case."""
        test_file = temp_dir / 'test_atomic_fail.txt'
        
        with pytest.raises(ValueError):
            with output_manager._atomic_write(test_file) as f:
                f.write('Some content')
                raise ValueError("Test error")
        
        # File should not exist after failure
        assert not test_file.exists()
        
        # Temp file should also not exist (cleaned up)
        temp_file = test_file.with_suffix(test_file.suffix + '.tmp')
        assert not temp_file.exists()
    
    def test_cleanup_temp_files(self, output_manager, temp_dir):
        """Test temporary file cleanup."""
        # Create some temp files
        results_dir = temp_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        temp_file1 = results_dir / 'test1.tmp'
        temp_file2 = results_dir / 'test2.temp'
        normal_file = results_dir / 'normal.txt'
        
        temp_file1.write_text('temp content 1')
        temp_file2.write_text('temp content 2')
        normal_file.write_text('normal content')
        
        # Run cleanup
        output_manager.cleanup_temp_files()
        
        # Temp files should be removed, normal file should remain
        assert not temp_file1.exists()
        assert not temp_file2.exists()
        assert normal_file.exists()
    
    def test_get_stats(self, output_manager, sample_fills_df):
        """Test statistics retrieval."""
        # Write some files to populate stats
        output_manager.write_unified_fills(sample_fills_df)
        
        stats = output_manager.get_stats()
        
        assert 'audit_only' in stats
        assert 'files_written' in stats
        assert 'written_files' in stats
        assert 'config' in stats
        
        assert stats['files_written'] > 0
        assert len(stats['written_files']) > 0
    
    def test_error_handling_file_write_failure(self, output_manager, sample_fills_df):
        """Test error handling when file write fails."""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            artifacts = output_manager.write_unified_fills(sample_fills_df)
            
            # Should handle error gracefully and return empty artifacts
            assert artifacts == {}
    
    def test_global_instance_management(self, output_config):
        """Test global instance management functions."""
        # Reset to clean state
        reset_output_manager()
        
        # Get first instance
        om1 = get_output_manager(output_config)
        
        # Get second instance (should be same)
        om2 = get_output_manager()
        
        assert om1 is om2
        
        # Reset and get new instance
        reset_output_manager()
        om3 = get_output_manager(output_config)
        
        assert om3 is not om1
    
    def test_write_operations_in_audit_mode(self, output_config):
        """Test that all write operations respect audit mode."""
        with patch('os.getenv', return_value='1'):  # Audit mode
            om = OutputManager(output_config)
            
            # All write operations should return None/empty in audit mode
            assert om.write_unified_fills(pd.DataFrame()) == {}
            assert om.write_signal_data(pd.DataFrame()) is None
            assert om.write_performance_report({}) is None
            assert om.write_backtest_results({}, 'test') is None
            assert om.write_validation_log(['test']) is None
