"""
Predictive Maintenance System
Machine learning system for predicting equipment failures in manufacturing.

Dataset: AI4I 2020 Predictive Maintenance Dataset
Source: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
License: CC BY-NC-SA 4.0
Project License: CC BY-NC-SA 4.0 (Non-commercial use only)

Author: [Your Name]
Date: [Current Date]
"""

import sys
import os
from pathlib import Path
import logging
import time

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import utility functions
try:
    import plotting_utils as pu
except ImportError as e:
    logger.error(f"Failed to import plotting_utils: {e}")
    sys.exit(1)

# --- Configuration ---
class Config:
    """Configuration settings for the predictive maintenance pipeline."""
    
    # Control Flags
    ENABLE_FEATURE_ENGINEERING = False  # Set to True to enable feature engineering
    USE_PCA_FEATURES = False  # Set to True to train model using PCA components
    
    # Paths
    BASE_DIR = Path(__file__).parent
    PLOTS_DIR = BASE_DIR / "plots"
    MODELS_DIR = BASE_DIR / "models"
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Dataset information
    DATASET_NAME = "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
    DATASET_FILE = "ai4i2020.csv"

def setup_directories():
    """Create necessary directories for outputs."""
    plot_subdirs = {
        "eda": Config.PLOTS_DIR / "eda",
        "feature_engineering": Config.PLOTS_DIR / "feature_engineering",
        "transformed_data": Config.PLOTS_DIR / "transformed_data",
        "pca": Config.PLOTS_DIR / "pca",
        "target_inspection": Config.PLOTS_DIR / "target_inspection",
        "feature_selection": Config.PLOTS_DIR / "feature_selection",
        "model_eval": Config.PLOTS_DIR / "model_eval"
    }
    
    # Create all directories
    Config.PLOTS_DIR.mkdir(exist_ok=True)
    Config.MODELS_DIR.mkdir(exist_ok=True)
    
    for subdir_path in plot_subdirs.values():
        subdir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Directory structure created in: {Config.PLOTS_DIR}")
    return plot_subdirs

def download_dataset():
    """Download the dataset from Kaggle."""
    try:
        import kagglehub
        logger.info("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download(Config.DATASET_NAME)
        logger.info(f"Dataset downloaded to: {path}")
        return path
    except ImportError:
        logger.error("kagglehub not installed. Please install: pip install kagglehub")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.error("Please ensure you have Kaggle API credentials configured")
        logger.error("Visit: https://www.kaggle.com/docs/api for setup instructions")
        sys.exit(1)

def load_and_prepare_data(dataset_path):
    """Load and perform initial data preparation."""
    try:
        df = pd.read_csv(Path(dataset_path) / Config.DATASET_FILE)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {dataset_path}/{Config.DATASET_FILE}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Remove unnecessary columns
    columns_to_drop = ['UDI', 'Product ID', 'RNF']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(existing_columns_to_drop, axis=1)
    
    logger.info(f"Dropped columns: {existing_columns_to_drop}")
    logger.info(f"Final dataset shape: {df.shape}")
    
    return df

def perform_eda(df, plot_subdirs):
    """Perform exploratory data analysis."""
    logger.info("Starting Exploratory Data Analysis...")
    
    # Display basic information
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast 5 rows:\n{df.tail()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nBasic statistics:\n{df.describe()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Failure distribution analysis
    print(f"\nMachine failure distribution:\n{df['Machine failure'].value_counts()}")
    
    failure_types = [col for col in df.columns if 'failure' in col.lower()]
    if len(failure_types) > 1:
        print("\nFailure types distribution:")
        for failure_type in failure_types:
            print(f"\n{failure_type}:\n{df[failure_type].value_counts()}")
    
    print(f"\nProduct type distribution:\n{df['Type'].value_counts()}")
    
    # Visualizations
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    pu.plot_numerical_distributions(df, numerical_cols, 
                                   filename=str(plot_subdirs["eda"] / 'numerical_distributions.png'))
    
    correlation_eda = df.select_dtypes(include=[np.number]).corr()
    pu.plot_correlation_heatmap(correlation_eda, title='Correlation Heatmap (Initial)', 
                               filename=str(plot_subdirs["eda"] / 'correlation_heatmap.png'))
    
    pu.plot_failure_rate_by_type(df, filename=str(plot_subdirs["eda"] / 'failure_by_type.png'))
    
    # Feature-specific analysis
    analyze_specific_features(df, plot_subdirs)
    
    logger.info("EDA completed successfully")
    return df

def analyze_specific_features(df, plot_subdirs):
    """Analyze specific features related to different failure types."""
    
    # Tool wear vs torque analysis
    pu.plot_scatter(df, x_col='Tool wear [min]', y_col='Torque [Nm]', hue_col='Machine failure',
                   title='Tool Wear vs Torque by Machine Failure', 
                   filename=str(plot_subdirs["eda"] / 'tool_wear_vs_torque.png'))
    
    # Temperature difference analysis
    df['Temp Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
    hdf_lines = [
        ('h', 8.6, 'r', '--', 'Critical Temp Diff (8.6 K)'),
        ('v', 1380, 'g', '--', 'Critical Speed (1380 rpm)')
    ]
    pu.plot_scatter(df, x_col='Rotational speed [rpm]', y_col='Temp Difference', 
                   hue_col='Machine failure', title='Temperature Difference vs Rotational Speed', 
                   filename=str(plot_subdirs["eda"] / 'temp_diff_vs_speed.png'), add_lines=hdf_lines)
    
    # Power analysis
    df['Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
    pwf_lines = [
        (3500, 'r', '--', 'Min Power Threshold (3500 W)'),
        (9000, 'g', '--', 'Max Power Threshold (9000 W)')
    ]
    pu.plot_histogram(df['Power [W]'], title='Power Distribution', 
                     filename=str(plot_subdirs["eda"] / 'power_distribution.png'), add_lines=pwf_lines)
    
    # Overstrain analysis
    df['Overstrain Indicator'] = df['Tool wear [min]'] * df['Torque [Nm]']
    pu.plot_overstrain_analysis(df, filename=str(plot_subdirs["eda"] / 'overstrain_analysis.png'))
    
    # Tool wear distribution
    twf_lines = [
        (200, 'r', '--', 'Min Tool Wear Failure (200 min)'),
        (240, 'g', '--', 'Max Tool Wear Failure (240 min)')
    ]
    pu.plot_histogram(df['Tool wear [min]'], title='Tool Wear Distribution', 
                     filename=str(plot_subdirs["eda"] / 'tool_wear_distribution.png'), add_lines=twf_lines)
    
    # Failure-specific analysis
    failures_df = df[df['Machine failure'] == 1].copy()
    if not failures_df.empty:
        pu.plot_failure_distributions(failures_df, filename=str(plot_subdirs["eda"] / 'failure_distributions.png'))
        pu.plot_failure_correlation(failures_df, filename=str(plot_subdirs["eda"] / 'failure_correlation.png'))
        pu.plot_detailed_failure_analysis(failures_df, filename=str(plot_subdirs["eda"] / 'failure_detailed_analysis.png'))
        
        failure_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                           'Torque [Nm]', 'Tool wear [min]']
        pu.plot_failure_relationships_pairplot(failures_df, failure_features, 
                                              filename=str(plot_subdirs["eda"] / 'failure_relationships.png'))

def engineer_features(df, plot_subdirs):
    """Engineer additional features if enabled."""
    if not Config.ENABLE_FEATURE_ENGINEERING:
        logger.info("Feature engineering disabled")
        return df
    
    logger.info("Starting feature engineering...")
    
    # Tool Wear features
    df['Tool_Wear_Rate'] = df['Tool wear [min]'] / df['Rotational speed [rpm]'] * 1000
    df['Tool_Wear_Efficiency'] = df['Power [W]'] / (df['Tool wear [min]'] + 1)
    
    # Heat-related features
    df['Heat_Load_Index'] = df['Temp Difference'] * df['Rotational speed [rpm]'] / 1000
    df['Thermal_Stress'] = df['Temp Difference'] * df['Torque [Nm]']
    
    # Power-related features
    df['Power_Density'] = df['Power [W]'] / df['Torque [Nm]']
    df['Power_Stability'] = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1)
    
    # Strain-related features
    df['Normalized_Strain'] = df.apply(
        lambda row: row['Overstrain Indicator'] / (11000 if row['Type'] == 'L' else 12000 if row['Type'] == 'M' else 13000), 
        axis=1
    )
    
    # Threshold indicators
    df['Near_Tool_Failure'] = ((df['Tool wear [min]'] > 180) & (df['Tool wear [min]'] < 240)).astype(int)
    df['Critical_Temp_Diff'] = (df['Temp Difference'] > 8.0).astype(int)
    df['High_Torque_Flag'] = (df['Torque [Nm]'] > df['Torque [Nm]'].quantile(0.9)).astype(int)
    df['Low_Speed_High_Torque'] = ((df['Rotational speed [rpm]'] < 1400) & (df['Torque [Nm]'] > 40)).astype(int)
    
    # Polynomial features
    df['Tool_Wear_Squared'] = df['Tool wear [min]'] ** 2
    df['Torque_Squared'] = df['Torque [Nm]'] ** 2
    df['Speed_Squared'] = df['Rotational speed [rpm]'] ** 2
    
    # Log transformations
    df['Log_Tool_Wear'] = np.log1p(df['Tool wear [min]'])
    
    # Interactions
    df['Wear_Torque_Interaction'] = df['Tool wear [min]'] * df['Torque [Nm]']
    df['Temp_Speed_Interaction'] = df['Temp Difference'] * df['Rotational speed [rpm]'] / 1000
    df['Power_Wear_Ratio'] = df['Power [W]'] / (df['Tool wear [min]'] + 1)
    
    # Type-specific features
    for product_type in ['L', 'M', 'H']:
        mask = df['Type'] == product_type
        df[f'Type_{product_type}_Wear'] = df['Tool wear [min]'] * mask
        df[f'Type_{product_type}_Torque'] = df['Torque [Nm]'] * mask
    
    # TWF-specific features
    df['TWF_Risk_Score'] = ((df['Tool wear [min]'] - 200) / 40).clip(0, 1)
    df['High_Wear_Low_Efficiency'] = ((df['Tool wear [min]'] > 180) & 
                                     (df['Power_Wear_Ratio'] < df['Power_Wear_Ratio'].quantile(0.3))).astype(int)
    df['TWF_Risk_Index'] = df['TWF_Risk_Score'] * df['Torque [Nm]'] / 40
    
    # Visualize engineered features
    engineered_features = [col for col in df.columns if any(keyword in col for keyword in ['TWF', 'Tool', 'Wear'])][:6]
    if 'TWF' in df.columns:
        pu.plot_engineered_feature_distributions(df, engineered_features, hue_col='TWF', 
                                                filename=str(plot_subdirs["feature_engineering"] / 'engineered_features.png'))
    
    logger.info(f"Feature engineering completed. New shape: {df.shape}")
    return df

def prepare_features_and_target(df):
    """Prepare features and target variables."""
    target_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF']
    numerical_cols = [col for col in df.columns if df[col].dtype != 'object' and col not in target_cols]
    categorical_cols = ['Type']
    
    X = df[numerical_cols + categorical_cols]
    
    # Create multi-class target
    y_multiclass = df[['TWF', 'HDF', 'PWF', 'OSF']].idxmax(axis=1)
    y_multiclass[df['Machine failure'] == 0] = 'No Failure'
    
    # Encode target
    label_encoder = LabelEncoder()
    y_multiclass_encoded = label_encoder.fit_transform(y_multiclass)
    y_classes = label_encoder.classes_
    
    logger.info("Class encoding mapping:")
    for i, class_name in enumerate(y_classes):
        logger.info(f"{class_name} -> {i}")
    
    return X, y_multiclass_encoded, label_encoder, y_classes, numerical_cols, categorical_cols

def preprocess_data(X, y, numerical_cols, categorical_cols):
    """Split and preprocess the data."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    
    # Define preprocessor
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Fit and transform
    logger.info("Preprocessing data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names
    num_feature_names = numerical_cols
    cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
    feature_names = num_feature_names + cat_feature_names
    
    # Create DataFrames with clean names
    X_train_clean = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_test_clean = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
    
    # Clean column names for XGBoost
    clean_columns = [col.replace('[', '_').replace(']', '_').replace(' ', '_').replace('<', '_') 
                    for col in X_train_clean.columns]
    X_train_clean.columns = clean_columns
    X_test_clean.columns = clean_columns
    
    return X_train_clean, X_test_clean, y_train, y_test, clean_columns

def perform_pca_analysis(X_train_clean, y_train, plot_subdirs):
    """Perform PCA analysis on the training data."""
    logger.info("Starting PCA analysis...")
    
    pca = PCA()
    principal_components = pca.fit_transform(X_train_clean)
    explained_variance = pca.explained_variance_ratio_
    
    cumulative_variance = np.cumsum(explained_variance)
    n_components_pca = np.argmax(cumulative_variance >= 0.95) + 1
    logger.info(f"Number of PCA components needed for >= 95% variance: {n_components_pca}")
    
    # Create PCA transformer for later use
    pca_transformer = PCA(n_components=n_components_pca)
    pca_transformer.fit(X_train_clean)
    
    # Visualizations
    pu.plot_pca_explained_variance(explained_variance, 
                                  filename=str(plot_subdirs["pca"] / 'pca_explained_variance.png'))
    pu.plot_pca_cumulative_variance(explained_variance, 
                                   filename=str(plot_subdirs["pca"] / 'pca_cumulative_variance.png'))
    pu.plot_pca_scatter(principal_components, y_train, explained_variance, dim=2, 
                       filename=str(plot_subdirs["pca"] / 'pca_scatter_2d.png'))
    
    if principal_components.shape[1] >= 3:
        pu.plot_pca_scatter(principal_components, y_train, explained_variance, dim=3, 
                           filename=str(plot_subdirs["pca"] / 'pca_scatter_3d.png'))
    
    pu.plot_pca_biplot(principal_components, pca, list(X_train_clean.columns), y_train, explained_variance, 
                      filename=str(plot_subdirs["pca"] / 'pca_biplot.png'))
    
    return pca_transformer, n_components_pca

def select_features(X_train_clean, y_train, plot_subdirs):
    """Perform feature selection if feature engineering is enabled."""
    if not Config.ENABLE_FEATURE_ENGINEERING:
        logger.info("Feature selection skipped (feature engineering disabled)")
        return list(X_train_clean.columns)
    
    logger.info("Starting feature selection...")
    
    # Random Forest feature importance
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE, class_weight='balanced')
    rf_selector.fit(X_train_clean, y_train)
    
    feature_importance = pd.DataFrame({
        'Feature': X_train_clean.columns,
        'Importance': rf_selector.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    logger.info("Top 15 features by Random Forest importance:")
    logger.info(f"\n{feature_importance.head(15)}")
    
    # Plot feature importance
    pu.plot_feature_importance(feature_importance, top_n=15, title='Top 15 Features by Random Forest Importance',
                              filename=str(plot_subdirs["feature_selection"] / 'rf_feature_importance.png'))
    
    # Select features above median importance
    selector = SelectFromModel(rf_selector, threshold='median', prefit=True)
    selected_features_mask = selector.get_support()
    selected_features = X_train_clean.columns[selected_features_mask].tolist()
    
    # Add TWF-specific features
    twf_features = [col for col in X_train_clean.columns if any(keyword in col for keyword in ['TWF', 'Tool', 'Wear'])]
    additional_features = [f for f in twf_features if f not in selected_features]
    
    final_features = selected_features + additional_features
    
    logger.info(f"Selected {len(final_features)} features from {X_train_clean.shape[1]} total features")
    
    return final_features

def train_and_evaluate_models(X_train_clean, X_test_clean, y_train, y_test, y_classes, 
                             selected_features, plot_subdirs, pca_transformer=None, n_components_pca=None):
    """Train and evaluate XGBoost models."""
    
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Please install: pip install xgboost")
        sys.exit(1)
    
    models_trained = {}
    
    # Calculate sample weights for class imbalance
    class_weights = len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
    sample_weights = np.array([class_weights[i] for i in y_train])
    
    if not Config.USE_PCA_FEATURES:
        logger.info("Training models with standard features...")
        
        # Model with all features
        logger.info("Training XGBoost with all features...")
        xgb_all = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(y_classes),
            eval_metric='mlogloss',
            random_state=Config.RANDOM_STATE
        )
        xgb_all.fit(X_train_clean, y_train, sample_weight=sample_weights)
        
        # Evaluate all features model
        evaluate_model(xgb_all, X_test_clean, y_test, y_classes, "All Features", plot_subdirs)
        models_trained['all_features'] = xgb_all
        
        # Model with selected features (if feature engineering enabled)
        if Config.ENABLE_FEATURE_ENGINEERING and selected_features:
            logger.info("Training XGBoost with selected features...")
            X_train_selected = X_train_clean[selected_features]
            X_test_selected = X_test_clean[selected_features]
            
            xgb_selected = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=len(y_classes),
                eval_metric='mlogloss',
                random_state=Config.RANDOM_STATE
            )
            xgb_selected.fit(X_train_selected, y_train, sample_weight=sample_weights)
            
            # Evaluate selected features model
            evaluate_model(xgb_selected, X_test_selected, y_test, y_classes, "Selected Features", plot_subdirs)
            models_trained['selected_features'] = xgb_selected
    
    else:
        # PCA features model
        if pca_transformer is not None:
            logger.info(f"Training XGBoost with {n_components_pca} PCA features...")
            X_train_pca = pca_transformer.transform(X_train_clean)
            X_test_pca = pca_transformer.transform(X_test_clean)
            
            xgb_pca = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=len(y_classes),
                eval_metric='mlogloss',
                random_state=Config.RANDOM_STATE
            )
            xgb_pca.fit(X_train_pca, y_train, sample_weight=sample_weights)
            
            # Evaluate PCA model
            evaluate_model(xgb_pca, X_test_pca, y_test, y_classes, f"PCA ({n_components_pca} components)", plot_subdirs)
            models_trained['pca'] = xgb_pca
    
    return models_trained

def evaluate_model(model, X_test, y_test, y_classes, model_name, plot_subdirs):
    """Evaluate a trained model."""
    logger.info(f"Evaluating {model_name} model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = pu.get_metrics(y_test, y_pred, y_classes)
    
    logger.info(f"\n{model_name} Model Metrics:")
    for metric_name, metric_value in metrics.items():
        if metric_name == 'F1 Scores per Class':
            logger.info(f"{metric_name}:")
            for i, class_name in enumerate(y_classes):
                score = metric_value[i] if i < len(metric_value) else 0.0
                logger.info(f"  {class_name}: {score:.4f}")
        else:
            logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Classification report
    y_test_decoded = y_classes[y_test] if hasattr(y_test, '__iter__') else [y_classes[y_test]]
    y_pred_decoded = y_classes[y_pred] if hasattr(y_pred, '__iter__') else [y_classes[y_pred]]
    
    report = classification_report(y_test_decoded, y_pred_decoded, target_names=y_classes, zero_division=0)
    logger.info(f"\n{model_name} Classification Report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=y_classes)
    
    # Plot confusion matrix
    safe_model_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    pu.plot_confusion_matrix(cm, y_classes, title=f'{model_name} Confusion Matrix',
                            filename=str(plot_subdirs["model_eval"] / f'confusion_matrix_{safe_model_name}.png'))
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_names = [f'Feature_{i}' for i in range(len(model.feature_importances_))]
        if hasattr(X_test, 'columns'):
            feature_names = list(X_test.columns)
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        pu.plot_feature_importance(importance_df, top_n=15, title=f'{model_name} Feature Importance',
                                  filename=str(plot_subdirs["model_eval"] / f'feature_importance_{safe_model_name}.png'))
    
    return metrics

def save_best_model(models_trained):
    """Save the best performing model."""
    if not models_trained:
        logger.warning("No models to save")
        return
    
    # For simplicity, save the first model (or implement comparison logic)
    model_name, model = next(iter(models_trained.items()))
    
    model_filename = Config.MODELS_DIR / f'xgboost_predictive_maintenance_{model_name}.pkl'
    joblib.dump(model, model_filename)
    logger.info(f"Model saved as: {model_filename}")

def main():
    """Main pipeline for predictive maintenance model training and evaluation."""
    logger.info("Starting Predictive Maintenance Pipeline")
    logger.info(f"Configuration: Feature Engineering={Config.ENABLE_FEATURE_ENGINEERING}, PCA={Config.USE_PCA_FEATURES}")
    
    # Setup
    plot_subdirs = setup_directories()
    
    # Data loading
    dataset_path = download_dataset()
    df = load_and_prepare_data(dataset_path)
    
    # EDA
    df = perform_eda(df, plot_subdirs)
    
    # Feature engineering
    df = engineer_features(df, plot_subdirs)
    
    # Prepare features and target
    X, y, label_encoder, y_classes, numerical_cols, categorical_cols = prepare_features_and_target(df)
    
    # Preprocess data
    X_train_clean, X_test_clean, y_train, y_test, feature_names = preprocess_data(
        X, y, numerical_cols, categorical_cols
    )
    
    # Target distribution visualization
    pu.plot_target_distribution(pd.Series(y_train), title='Target Distribution (Training Set)',
                               filename=str(plot_subdirs["target_inspection"] / 'target_distribution.png'),
                               labels=[y_classes[i] for i in sorted(pd.Series(y_train).unique())])
    
    # PCA analysis
    pca_transformer, n_components_pca = perform_pca_analysis(X_train_clean, y_train, plot_subdirs)
    
    # Feature selection
    selected_features = select_features(X_train_clean, y_train, plot_subdirs)
    
    # Model training and evaluation
    models_trained = train_and_evaluate_models(
        X_train_clean, X_test_clean, y_train, y_test, y_classes,
        selected_features, plot_subdirs, pca_transformer, n_components_pca
    )
    
    # Save best model
    save_best_model(models_trained)
    
    logger.info("Predictive Maintenance Pipeline completed successfully!")
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Models saved in: {Config.MODELS_DIR}")
    print(f"Plots saved in: {Config.PLOTS_DIR}")

if __name__ == "__main__":
    main()