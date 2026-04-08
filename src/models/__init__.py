"""Model implementations for biodiversity monitoring system."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Tuple, Optional
from omegaconf import DictConfig
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manage device selection for PyTorch models."""
    
    @staticmethod
    def get_device(device_preference: str = "auto") -> torch.device:
        """Get the best available device.
        
        Args:
            device_preference: Device preference ("auto", "cuda", "mps", "cpu")
            
        Returns:
            PyTorch device
        """
        if device_preference == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Using CUDA device")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using MPS device")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(device_preference)
            logger.info(f"Using {device_preference} device")
        
        return device


class BiodiversityNeuralNetwork(nn.Module):
    """Neural network for multi-label species classification."""
    
    def __init__(self, input_size: int, hidden_layers: list, output_size: int, dropout_rate: float = 0.2):
        """Initialize neural network.
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            output_size: Number of output classes (species)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # Multi-label classification
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.network(x)


class ModelTrainer:
    """Trainer for neural network models."""
    
    def __init__(self, config: DictConfig):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = DeviceManager.get_device(config.model.device)
        
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> BiodiversityNeuralNetwork:
        """Train neural network model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained neural network model
        """
        logger.info("Training neural network model")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.config.model.neural_network.batch_size, shuffle=True)
        
        # Initialize model
        model = BiodiversityNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_layers=self.config.model.neural_network.hidden_layers,
            output_size=y_train.shape[1],
            dropout_rate=self.config.model.neural_network.dropout_rate
        ).to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=self.config.model.neural_network.learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.model.neural_network.epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.model.neural_network.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        logger.info("Neural network training completed")
        return model


class BaselineModels:
    """Collection of baseline machine learning models."""
    
    def __init__(self, config: DictConfig):
        """Initialize baseline models.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained Random Forest model
        """
        logger.info("Training Random Forest model")
        
        # For multi-label classification, we'll train separate models for each species
        models = {}
        rf_config = self.config.model.baseline.random_forest
        
        for i in range(y_train.shape[1]):
            model = RandomForestClassifier(
                n_estimators=rf_config.n_estimators,
                max_depth=rf_config.max_depth,
                random_state=rf_config.random_state
            )
            model.fit(X_train, y_train[:, i])
            models[f'species_{i}'] = model
        
        return models
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, xgb.XGBClassifier]:
        """Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained XGBoost models
        """
        logger.info("Training XGBoost model")
        
        models = {}
        xgb_config = self.config.model.baseline.xgboost
        
        for i in range(y_train.shape[1]):
            model = xgb.XGBClassifier(
                n_estimators=xgb_config.n_estimators,
                max_depth=xgb_config.max_depth,
                learning_rate=xgb_config.learning_rate,
                random_state=xgb_config.random_state,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train[:, i])
            models[f'species_{i}'] = model
        
        return models
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, lgb.LGBMClassifier]:
        """Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained LightGBM models
        """
        logger.info("Training LightGBM model")
        
        models = {}
        lgb_config = self.config.model.baseline.lightgbm
        
        for i in range(y_train.shape[1]):
            model = lgb.LGBMClassifier(
                n_estimators=lgb_config.n_estimators,
                max_depth=lgb_config.max_depth,
                learning_rate=lgb_config.learning_rate,
                random_state=lgb_config.random_state,
                verbose=-1
            )
            model.fit(X_train, y_train[:, i])
            models[f'species_{i}'] = model
        
        return models


class ModelEnsemble:
    """Ensemble of multiple models for better predictions."""
    
    def __init__(self, models: Dict[str, Any]):
        """Initialize ensemble.
        
        Args:
            models: Dictionary of trained models
        """
        self.models = models
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        for model_name, model in self.models.items():
            if isinstance(model, dict):
                # Multi-label models (one per species)
                species_preds = []
                for species_model in model.values():
                    pred = species_model.predict_proba(X)[:, 1]  # Probability of positive class
                    species_preds.append(pred)
                predictions.append(np.column_stack(species_preds))
            else:
                # Single model
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
