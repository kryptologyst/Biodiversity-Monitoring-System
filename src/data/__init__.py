"""Data processing module for biodiversity monitoring system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


class BiodiversityDataGenerator:
    """Generate synthetic biodiversity monitoring data."""
    
    def __init__(self, config: DictConfig):
        """Initialize data generator with configuration.
        
        Args:
            config: Configuration dictionary containing data generation parameters
        """
        self.config = config
        self.n_samples = config.synthetic.n_samples
        self.n_species = config.synthetic.n_species
        self.features = config.synthetic.features
        self.species_config = config.species
        
        # Set random seed for reproducibility
        np.random.seed(config.model.random_seed)
        
    def generate_features(self) -> pd.DataFrame:
        """Generate synthetic environmental features.
        
        Returns:
            DataFrame containing environmental features
        """
        logger.info(f"Generating {self.n_samples} samples with {len(self.features)} features")
        
        data = {}
        
        # Generate each feature with realistic distributions
        if 'sound_activity' in self.features:
            data['sound_activity'] = np.random.normal(50, 15, self.n_samples)
        
        if 'vegetation_index' in self.features:
            data['vegetation_index'] = np.random.normal(0.6, 0.1, self.n_samples)
        
        if 'time_of_day' in self.features:
            data['time_of_day'] = np.random.randint(0, 24, self.n_samples)
        
        if 'temperature' in self.features:
            data['temperature'] = np.random.normal(22, 5, self.n_samples)
        
        if 'humidity' in self.features:
            data['humidity'] = np.random.uniform(0.3, 0.9, self.n_samples)
        
        if 'light_level' in self.features:
            data['light_level'] = np.random.uniform(0.1, 1.0, self.n_samples)
        
        if 'wind_speed' in self.features:
            data['wind_speed'] = np.random.exponential(3, self.n_samples)
        
        if 'precipitation' in self.features:
            data['precipitation'] = np.random.exponential(0.5, self.n_samples)
        
        # Add spatial coordinates (simulated monitoring locations)
        data['latitude'] = np.random.uniform(40.5, 41.0, self.n_samples)
        data['longitude'] = np.random.uniform(-74.5, -73.5, self.n_samples)
        
        # Add temporal information
        data['date'] = pd.date_range('2023-01-01', periods=self.n_samples, freq='H')
        
        return pd.DataFrame(data)
    
    def generate_species_labels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate species presence labels based on environmental conditions.
        
        Args:
            features_df: DataFrame containing environmental features
            
        Returns:
            DataFrame containing species presence labels
        """
        logger.info("Generating species presence labels")
        
        labels = {}
        
        for species_name, species_info in self.species_config.items():
            conditions = species_info.conditions
            presence = np.ones(self.n_samples, dtype=bool)
            
            # Apply conditions for each species
            for feature, (min_val, max_val) in conditions.items():
                if feature in features_df.columns:
                    feature_values = features_df[feature].values
                    presence &= (feature_values >= min_val) & (feature_values <= max_val)
            
            # Add some randomness to make it more realistic
            noise = np.random.random(self.n_samples) < 0.1
            presence = presence.astype(int)
            presence[noise] = 1 - presence[noise]
            
            labels[f'{species_name}_present'] = presence
        
        return pd.DataFrame(labels)
    
    def generate_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset with features and labels.
        
        Returns:
            Tuple of (features_df, labels_df)
        """
        logger.info("Generating complete biodiversity dataset")
        
        features_df = self.generate_features()
        labels_df = self.generate_species_labels(features_df)
        
        logger.info(f"Generated dataset: {features_df.shape[0]} samples, "
                   f"{features_df.shape[1]} features, {labels_df.shape[1]} species")
        
        return features_df, labels_df


class DataProcessor:
    """Process and prepare biodiversity data for modeling."""
    
    def __init__(self, config: DictConfig):
        """Initialize data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def prepare_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare features for modeling.
        
        Args:
            features_df: DataFrame containing features
            
        Returns:
            Numpy array of prepared features
        """
        # Select only numeric features for modeling
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        numeric_features = [f for f in numeric_features if f not in ['latitude', 'longitude']]
        
        # Normalize features
        features_array = features_df[numeric_features].values
        
        # Simple min-max normalization
        features_normalized = (features_array - features_array.min(axis=0)) / (
            features_array.max(axis=0) - features_array.min(axis=0) + 1e-8
        )
        
        logger.info(f"Prepared features: {features_normalized.shape}")
        return features_normalized
    
    def prepare_labels(self, labels_df: pd.DataFrame) -> np.ndarray:
        """Prepare labels for modeling.
        
        Args:
            labels_df: DataFrame containing species presence labels
            
        Returns:
            Numpy array of prepared labels
        """
        labels_array = labels_df.values
        logger.info(f"Prepared labels: {labels_array.shape}")
        return labels_array
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets.
        
        Args:
            X: Feature array
            y: Label array
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        test_size = self.config.evaluation.test_size
        val_size = self.config.evaluation.validation_size
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.model.random_seed
        )
        
        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.config.model.random_seed
        )
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, "
                   f"Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
