# Biodiversity Monitoring System

A comprehensive biodiversity monitoring system that detects species presence using environmental sensor data and machine learning models. This system demonstrates how environmental conditions can be used to predict the presence of different species in conservation areas.

## Features

- **Multi-species Detection**: Detects presence of birds, monkeys, insects, reptiles, and amphibians
- **Environmental Sensors**: Uses temperature, humidity, sound activity, vegetation index, and other environmental factors
- **Multiple ML Models**: Implements Random Forest, XGBoost, LightGBM, and Neural Networks
- **Interactive Demo**: Streamlit-based web application with maps and visualizations
- **Comprehensive Evaluation**: Detailed metrics and model comparison
- **Spatial Analysis**: Geographic visualization of species detections

## Quick Start

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Biodiversity-Monitoring-System.git
cd Biodiversity-Monitoring-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. **Train Models**:
```bash
python scripts/train.py
```

2. **Launch Interactive Demo**:
```bash
streamlit run demo/app.py
```

3. **View Results**:
   - Model performance: `assets/model_leaderboard.csv`
   - Evaluation report: `assets/evaluation_report.txt`
   - Visualizations: `assets/confusion_matrices.png`, `assets/species_detection_rates.png`

## Project Structure

```
biodiversity-monitoring-system/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model implementations
│   ├── eval/              # Evaluation modules
│   └── viz/               # Visualization modules
├── configs/               # Configuration files
├── data/                  # Data directories
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   └── external/         # External data sources
├── scripts/              # Training and utility scripts
├── demo/                 # Streamlit demo application
├── tests/                # Unit tests
├── assets/               # Generated outputs and visualizations
├── notebooks/            # Jupyter notebooks for exploration
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## Data Schema

### Environmental Features
- `sound_activity`: Sound level in decibels (dB)
- `vegetation_index`: NDVI-like vegetation index (0-1)
- `time_of_day`: Hour of day (0-23)
- `temperature`: Temperature in Celsius (°C)
- `humidity`: Relative humidity (0-1)
- `light_level`: Light intensity (0-1)
- `wind_speed`: Wind speed (m/s)
- `precipitation`: Precipitation amount (mm)

### Spatial Information
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `date`: Timestamp of observation

### Species Labels
- `bird_present`: Binary indicator for bird presence
- `monkey_present`: Binary indicator for monkey presence
- `insect_present`: Binary indicator for insect presence
- `reptile_present`: Binary indicator for reptile presence
- `amphibian_present`: Binary indicator for amphibian presence

## Models

### Baseline Models
- **Random Forest**: Ensemble of decision trees with bootstrap aggregation
- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Light gradient boosting machine for efficiency

### Advanced Models
- **Neural Network**: Multi-layer perceptron with dropout regularization
- **Ensemble**: Combination of all models for improved performance

## Evaluation Metrics

### Classification Metrics
- Accuracy: Overall correctness
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1 Score: Harmonic mean of precision and recall

### Multi-label Metrics
- Jaccard Score: Intersection over union for multi-label classification
- Hamming Loss: Fraction of incorrectly predicted labels

### Per-Species Metrics
- Individual accuracy, precision, recall, and F1 for each species

## Configuration

The system uses YAML configuration files:

- `configs/data.yaml`: Data generation and processing parameters
- `configs/model.yaml`: Model training parameters

Key configuration options:
- Number of samples and species
- Model hyperparameters
- Evaluation metrics
- Visualization settings

## Demo Application

The Streamlit demo provides:

### Interactive Map
- Geographic visualization of species detections
- Color-coded markers based on number of species detected
- Popup information for each location

### Analysis Charts
- Species distribution across locations
- Environmental condition distributions
- Temporal patterns of species activity

### Model Performance
- Leaderboard of model performance
- Accuracy comparisons
- Detailed evaluation metrics

### Filters
- Temperature range selection
- Time of day filtering
- Species selection

## Training Commands

```bash
# Train all models and generate evaluation
python scripts/train.py

# Run specific model training (example)
python -c "
from src.models.biodiversity_models import BaselineModels
from omegaconf import OmegaConf
config = OmegaConf.load('configs/model.yaml')
models = BaselineModels(config)
# ... training code
"
```

## Evaluation Commands

```bash
# Generate comprehensive evaluation report
python -c "
from src.eval.evaluation import BiodiversityEvaluator
# ... evaluation code
"

# Create visualizations
python -c "
import matplotlib.pyplot as plt
# ... plotting code
"
```

## Demo Instructions

1. **Launch the demo**:
   ```bash
   streamlit run demo/app.py
   ```

2. **Navigate the interface**:
   - Use sidebar filters to adjust data view
   - Explore the interactive map
   - View analysis charts in different tabs
   - Check model performance metrics

3. **Screenshots**:
   - Map view showing species detections
   - Charts showing environmental distributions
   - Model performance comparison

## Known Limitations

- **Synthetic Data**: Current implementation uses synthetic data for demonstration
- **Limited Species**: Only 5 species types are currently supported
- **Environmental Factors**: Simplified environmental model
- **Spatial Resolution**: Limited to point-based observations
- **Temporal Resolution**: Hourly resolution only

## Disclaimer

This is a research demonstration project using synthetic data. The system is designed for educational and research purposes. For operational use in real conservation scenarios:

- Validate with real field data
- Ensure compliance with local regulations
- Consider privacy implications for sensitive species
- Implement proper data governance protocols
- Consult with conservation experts

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Issues

For questions, bug reports, or feature requests, please visit:
https://github.com/kryptologyst

## Author

**kryptologyst** - [GitHub](https://github.com/kryptologyst)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# Biodiversity-Monitoring-System
