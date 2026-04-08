#!/usr/bin/env python3
"""Main training script for biodiversity monitoring system."""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.processing import BiodiversityDataGenerator, DataProcessor
from models.biodiversity_models import ModelTrainer, BaselineModels, ModelEnsemble
from eval.evaluation import BiodiversityEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    logger.info("Starting Biodiversity Monitoring System Training")
    
    # Load configuration
    config = OmegaConf.load("configs/data.yaml")
    model_config = OmegaConf.load("configs/model.yaml")
    config = OmegaConf.merge(config, model_config)
    
    # Set random seeds for reproducibility
    np.random.seed(config.model.random_seed)
    torch.manual_seed(config.model.random_seed)
    
    # Create output directories
    os.makedirs("assets", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Generate synthetic data
    logger.info("Generating synthetic biodiversity data")
    data_generator = BiodiversityDataGenerator(config)
    features_df, labels_df = data_generator.generate_dataset()
    
    # Save raw data
    features_df.to_csv("data/processed/features.csv", index=False)
    labels_df.to_csv("data/processed/labels.csv", index=False)
    logger.info("Data saved to data/processed/")
    
    # Process data
    logger.info("Processing data for modeling")
    processor = DataProcessor(config)
    X = processor.prepare_features(features_df)
    y = processor.prepare_labels(labels_df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    
    # Get species names
    species_names = list(config.species.keys())
    
    # Initialize evaluator
    evaluator = BiodiversityEvaluator(config, species_names)
    
    # Train baseline models
    logger.info("Training baseline models")
    baseline_models = BaselineModels(config)
    
    # Random Forest
    rf_models = baseline_models.train_random_forest(X_train, y_train)
    evaluator.evaluate_model(rf_models, X_test, y_test, "Random Forest")
    
    # XGBoost
    xgb_models = baseline_models.train_xgboost(X_train, y_train)
    evaluator.evaluate_model(xgb_models, X_test, y_test, "XGBoost")
    
    # LightGBM
    lgb_models = baseline_models.train_lightgbm(X_train, y_train)
    evaluator.evaluate_model(lgb_models, X_test, y_test, "LightGBM")
    
    # Train neural network
    logger.info("Training neural network")
    trainer = ModelTrainer(config)
    nn_model = trainer.train_neural_network(X_train, y_train, X_val, y_val)
    evaluator.evaluate_model(nn_model, X_test, y_test, "Neural Network")
    
    # Create ensemble
    logger.info("Creating ensemble model")
    ensemble_models = {
        "random_forest": rf_models,
        "xgboost": xgb_models,
        "lightgbm": lgb_models,
        "neural_network": nn_model
    }
    ensemble = ModelEnsemble(ensemble_models)
    evaluator.evaluate_model(ensemble, X_test, y_test, "Ensemble")
    
    # Generate evaluation report
    logger.info("Generating evaluation report")
    report = evaluator.generate_report("assets/evaluation_report.txt")
    print(report)
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard()
    leaderboard.to_csv("assets/model_leaderboard.csv")
    print("\nModel Leaderboard:")
    print(leaderboard[['rank', 'accuracy', 'f1_macro', 'jaccard_score']].to_string())
    
    # Plot confusion matrices for best model
    best_model_name = leaderboard.index[0]
    best_model = ensemble_models[best_model_name.lower().replace(" ", "_")]
    
    evaluator.plot_confusion_matrices(
        best_model, X_test, y_test, best_model_name, 
        "assets/confusion_matrices.png"
    )
    
    # Plot species detection rates
    evaluator.plot_species_detection_rates("assets/species_detection_rates.png")
    
    # Save model predictions for demo
    logger.info("Saving predictions for demo")
    predictions_df = pd.DataFrame(
        ensemble.predict(X_test),
        columns=[f"{species}_predicted" for species in species_names]
    )
    
    # Combine with original features for demo
    demo_data = pd.concat([
        features_df.iloc[-len(X_test):].reset_index(drop=True),
        predictions_df
    ], axis=1)
    
    demo_data.to_csv("assets/demo_data.csv", index=False)
    
    logger.info("Training completed successfully!")
    logger.info("Results saved to assets/ directory")
    logger.info("Run 'streamlit run demo/app.py' to launch the interactive demo")


if __name__ == "__main__":
    main()
