# Predictive Maintenance System 🔧

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

A machine learning system for predicting equipment failures in manufacturing environments. This project uses sensor data to predict specific failure types before they occur, enabling proactive maintenance and reducing downtime.

## 🎯 Overview

This system predicts **5 different failure types**:
- **TWF** (Tool Wear Failure) - Cutting tool degradation
- **HDF** (Heat Dissipation Failure) - Overheating issues
- **PWF** (Power Failure) - Power system problems
- **OSF** (Overstrain Failure) - Mechanical overload
- **No Failure** - Normal operation

## 🚀 Features

- **Multi-class Classification**: Predicts specific failure types, not just binary failure/no-failure
- **Comprehensive EDA**: Extensive exploratory data analysis with 15+ visualization types
- **Feature Engineering**: Domain-specific feature creation based on manufacturing knowledge
- **Multiple Modeling Approaches**: Standard features, selected features, and PCA components
- **Advanced Visualizations**: 15+ plot types for comprehensive analysis
- **Production Ready**: Configurable pipeline with proper error handling and logging
- **Well Documented**: Comprehensive documentation and clean code structure

## 📊 Dataset Attribution

This project uses the **AI4I 2020 Predictive Maintenance Dataset**:
- **Source**: [Stephan Matzka on Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- **License**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Citation**: 
  ```
  Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. 
  UCI Machine Learning Repository. DOI: 10.24432/C5NS5P
  ```

### ⚠️ **Usage Restrictions**
- This dataset and derived models are for **non-commercial use only**
- Commercial applications require separate licensing arrangements

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Kaggle account and API credentials

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/BBahtiri/Predictive_Maintenance
   cd Predictive_Maintenance
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Kaggle API**
   - Create account at [kaggle.com](https://kaggle.com)
   - Go to Account → API → Create New API Token
   - Place `kaggle.json` in `~/.kaggle/` directory
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## 🚀 Quick Start

### Basic Usage
```bash
python predictive_maintenance.py
```

### Advanced Configuration
```python
# Edit configuration in predictive_maintenance.py
class Config:
    ENABLE_FEATURE_ENGINEERING = True   # Enable advanced features
    USE_PCA_FEATURES = False            # Use PCA transformation
    RANDOM_STATE = 42                   # Reproducibility
```

## 📈 Model Performance

The system achieves strong performance across all failure types:

| Failure Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| No Failure  | 0.98      | 0.99   | 0.99     |
| TWF         | 0.85      | 0.82   | 0.83     |
| HDF         | 0.78      | 0.75   | 0.76     |
| PWF         | 0.80      | 0.77   | 0.78     |
| OSF         | 0.82      | 0.79   | 0.80     |

*Results may vary based on configuration and data splits*

## 📁 Project Structure

```
predictive-maintenance/
├── README.md                    # Project documentation
├── LICENSE                      # CC BY-NC-SA 4.0 license
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── predictive_maintenance.py    # Main pipeline script
├── plotting_utils.py           # Visualization utilities
├── data/                       # Data directory (auto-created)
├── models/                     # Saved models (auto-created)
├── plots/                      # Generated visualizations (auto-created)
│   ├── eda/                   # Exploratory data analysis
│   ├── feature_engineering/   # Feature engineering plots
│   ├── pca/                   # PCA analysis
│   └── model_eval/            # Model evaluation
└── docs/                       # Additional documentation
```

## 🔧 Configuration Options

### Feature Engineering
```python
Config.ENABLE_FEATURE_ENGINEERING = True
```
- Creates 20+ engineered features
- Tool wear efficiency metrics
- Thermal stress indicators
- Power stability measures
- Type-specific interactions

### PCA Analysis
```python
Config.USE_PCA_FEATURES = True
```
- Dimensionality reduction
- Captures 95% variance
- Reduces feature complexity
- Alternative modeling approach

## 📊 Generated Outputs

### Visualizations (15+ plot types)
- **EDA**: Data distributions, correlations, failure patterns
- **Feature Analysis**: Engineered feature distributions
- **PCA**: Component analysis and biplots
- **Model Evaluation**: Confusion matrices, feature importance

### Models
- **XGBoost Classifier**: Optimized for multi-class prediction
- **Feature Selection**: Random Forest-based importance
- **Cross-validation**: Stratified sampling for robust evaluation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) to maintain compatibility with the underlying dataset.

### ✅ Permitted Uses:
- Educational and research purposes
- Personal learning projects
- Academic publications
- Open source contributions

### ❌ Restricted Uses:
- Commercial applications
- Proprietary software development
- Revenue-generating services
- Closed-source derivatives

### 📄 Commercial Use
For commercial applications, please contact the original dataset author for licensing arrangements.

## 🙏 Acknowledgments

- **Dataset**: Stephan Matzka for the AI4I 2020 Predictive Maintenance Dataset
- **Libraries**: scikit-learn, XGBoost, pandas, matplotlib, seaborn communities
- **Inspiration**: Industrial IoT and Industry 4.0 research community

## 📞 Contact

- **GitHub**: [@bbahtiri](https://github.com/bbahtiri)
- **Email**: betimbahtiri@outlook.de
- **LinkedIn**: [LinkedIN(https://www.linkedin.com/in/dr-ing-betim-bahtiri-4852ab242/)

---

⭐ **Star this repository if you find it helpful!**