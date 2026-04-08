"""Test suite for biodiversity monitoring system."""

import pytest
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.processing import BiodiversityDataGenerator, DataProcessor
from models.biodiversity_models import DeviceManager, BiodiversityNeuralNetwork
from eval.evaluation import BiodiversityEvaluator


class TestBiodiversityDataGenerator:
    """Test data generation functionality."""
    
    def test_data_generator_initialization(self):
        """Test data generator initialization."""
        config = OmegaConf.create({
            'synthetic': {
                'n_samples': 100,
                'n_species': 3,
                'features': ['temperature', 'humidity']
            },
            'species': {
                'bird': {'conditions': {'temperature': [15, 30]}},
                'monkey': {'conditions': {'temperature': [20, 35]}},
                'insect': {'conditions': {'temperature': [25, 40]}}
            },
            'model': {'random_seed': 42}
        })
        
        generator = BiodiversityDataGenerator(config)
        assert generator.n_samples == 100
        assert generator.n_species == 3
    
    def test_feature_generation(self):
        """Test feature generation."""
        config = OmegaConf.create({
            'synthetic': {
                'n_samples': 50,
                'n_species': 2,
                'features': ['temperature', 'humidity']
            },
            'species': {
                'bird': {'conditions': {'temperature': [15, 30]}},
                'monkey': {'conditions': {'temperature': [20, 35]}}
            },
            'model': {'random_seed': 42}
        })
        
        generator = BiodiversityDataGenerator(config)
        features_df = generator.generate_features()
        
        assert len(features_df) == 50
        assert 'temperature' in features_df.columns
        assert 'humidity' in features_df.columns
        assert 'latitude' in features_df.columns
        assert 'longitude' in features_df.columns


class TestDataProcessor:
    """Test data processing functionality."""
    
    def test_data_processor_initialization(self):
        """Test data processor initialization."""
        config = OmegaConf.create({
            'evaluation': {
                'test_size': 0.2,
                'validation_size': 0.2
            }
        })
        
        processor = DataProcessor(config)
        assert processor.config == config
    
    def test_feature_preparation(self):
        """Test feature preparation."""
        config = OmegaConf.create({
            'evaluation': {
                'test_size': 0.2,
                'validation_size': 0.2
            }
        })
        
        processor = DataProcessor(config)
        
        # Create test data
        features_df = pd.DataFrame({
            'temperature': [20, 25, 30],
            'humidity': [0.5, 0.6, 0.7],
            'latitude': [40.0, 40.1, 40.2],
            'longitude': [-74.0, -74.1, -74.2]
        })
        
        X = processor.prepare_features(features_df)
        
        assert X.shape[0] == 3
        assert X.shape[1] == 2  # Only temperature and humidity
        assert np.all(X >= 0) and np.all(X <= 1)  # Normalized


class TestDeviceManager:
    """Test device management functionality."""
    
    def test_device_selection(self):
        """Test device selection."""
        device = DeviceManager.get_device("cpu")
        assert str(device) == "cpu"
    
    def test_auto_device_selection(self):
        """Test automatic device selection."""
        device = DeviceManager.get_device("auto")
        assert device is not None


class TestBiodiversityNeuralNetwork:
    """Test neural network functionality."""
    
    def test_network_initialization(self):
        """Test network initialization."""
        network = BiodiversityNeuralNetwork(
            input_size=4,
            hidden_layers=[32, 16],
            output_size=3,
            dropout_rate=0.2
        )
        
        assert network is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        import torch
        
        network = BiodiversityNeuralNetwork(
            input_size=4,
            hidden_layers=[32, 16],
            output_size=3,
            dropout_rate=0.2
        )
        
        x = torch.randn(10, 4)
        output = network(x)
        
        assert output.shape == (10, 3)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output


class TestBiodiversityEvaluator:
    """Test evaluation functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        config = OmegaConf.create({})
        species_names = ['bird', 'monkey', 'insect']
        
        evaluator = BiodiversityEvaluator(config, species_names)
        
        assert evaluator.species_names == species_names
        assert len(evaluator.results) == 0
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        config = OmegaConf.create({})
        species_names = ['bird', 'monkey']
        
        evaluator = BiodiversityEvaluator(config, species_names)
        
        # Create test data
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 0]])
        
        metrics = evaluator._calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'bird_accuracy' in metrics
        assert 'monkey_accuracy' in metrics


if __name__ == "__main__":
    pytest.main([__file__])
