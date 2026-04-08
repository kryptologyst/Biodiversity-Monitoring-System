"""Evaluation module for biodiversity monitoring system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    jaccard_score, hamming_loss, multilabel_confusion_matrix
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


class BiodiversityEvaluator:
    """Evaluator for biodiversity monitoring models."""
    
    def __init__(self, config: DictConfig, species_names: List[str]):
        """Initialize evaluator.
        
        Args:
            config: Configuration dictionary
            species_names: List of species names
        """
        self.config = config
        self.species_names = species_names
        self.results = {}
        
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, float]:
        """Evaluate a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}")
        
        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            # For neural networks
            import torch
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                y_pred_proba = model(X_tensor).numpy()
                y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Store results
        self.results[model_name] = metrics
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1 Score: {metrics['f1_macro']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Multi-label specific metrics
        metrics['jaccard_score'] = jaccard_score(y_true, y_pred, average='macro')
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        
        # Per-species metrics
        for i, species in enumerate(self.species_names):
            species_accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
            species_precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            species_recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            species_f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            
            metrics[f'{species}_accuracy'] = species_accuracy
            metrics[f'{species}_precision'] = species_precision
            metrics[f'{species}_recall'] = species_recall
            metrics[f'{species}_f1'] = species_f1
        
        return metrics
    
    def create_leaderboard(self) -> pd.DataFrame:
        """Create a leaderboard of model performance.
        
        Returns:
            DataFrame with model rankings
        """
        if not self.results:
            logger.warning("No results available for leaderboard")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results).T
        
        # Sort by accuracy
        df = df.sort_values('accuracy', ascending=False)
        
        # Add ranking
        df['rank'] = range(1, len(df) + 1)
        
        logger.info("Created model leaderboard")
        return df
    
    def plot_confusion_matrices(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                               model_name: str, save_path: str = None) -> None:
        """Plot confusion matrices for each species.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            save_path: Path to save the plot
        """
        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            # For neural networks
            import torch
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                y_pred_proba = model(X_tensor).numpy()
                y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Create subplots
        n_species = len(self.species_names)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, species in enumerate(self.species_names):
            cm = confusion_matrix(y_test[:, i], y_pred[:, i])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{species} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Remove empty subplot
        if n_species < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_species_detection_rates(self, save_path: str = None) -> None:
        """Plot species detection rates across models.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            logger.warning("No results available for plotting")
            return
        
        # Extract per-species metrics
        models = list(self.results.keys())
        species_metrics = {}
        
        for species in self.species_names:
            species_metrics[species] = {
                'precision': [self.results[model].get(f'{species}_precision', 0) for model in models],
                'recall': [self.results[model].get(f'{species}_recall', 0) for model in models],
                'f1': [self.results[model].get(f'{species}_f1', 0) for model in models]
            }
        
        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['precision', 'recall', 'f1']
        for i, metric in enumerate(metrics):
            x = np.arange(len(models))
            width = 0.2
            
            for j, species in enumerate(self.species_names):
                axes[i].bar(x + j * width, species_metrics[species][metric], 
                           width, label=species, alpha=0.8)
            
            axes[i].set_xlabel('Models')
            axes[i].set_ylabel(f'{metric.capitalize()} Score')
            axes[i].set_title(f'{metric.capitalize()} by Species and Model')
            axes[i].set_xticks(x + width)
            axes[i].set_xticklabels(models, rotation=45)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Species detection rates plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: str = None) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        if not self.results:
            return "No evaluation results available."
        
        report = []
        report.append("=" * 60)
        report.append("BIODIVERSITY MONITORING SYSTEM - EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall leaderboard
        leaderboard = self.create_leaderboard()
        report.append("MODEL LEADERBOARD:")
        report.append("-" * 30)
        report.append(leaderboard[['rank', 'accuracy', 'f1_macro', 'jaccard_score']].to_string())
        report.append("")
        
        # Detailed results per model
        for model_name, metrics in self.results.items():
            report.append(f"DETAILED RESULTS - {model_name.upper()}:")
            report.append("-" * 40)
            report.append(f"Overall Accuracy: {metrics['accuracy']:.4f}")
            report.append(f"Macro F1 Score: {metrics['f1_macro']:.4f}")
            report.append(f"Jaccard Score: {metrics['jaccard_score']:.4f}")
            report.append(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
            report.append("")
            
            # Per-species results
            report.append("Per-Species Performance:")
            for species in self.species_names:
                acc = metrics.get(f'{species}_accuracy', 0)
                prec = metrics.get(f'{species}_precision', 0)
                rec = metrics.get(f'{species}_recall', 0)
                f1 = metrics.get(f'{species}_f1', 0)
                
                report.append(f"  {species}:")
                report.append(f"    Accuracy: {acc:.4f}")
                report.append(f"    Precision: {prec:.4f}")
                report.append(f"    Recall: {rec:.4f}")
                report.append(f"    F1 Score: {f1:.4f}")
            
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report_text
