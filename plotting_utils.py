"""
Plotting Utilities for Predictive Maintenance System

This module provides visualization functions for exploratory data analysis,
model evaluation, and feature analysis in the predictive maintenance pipeline.

License: CC BY-NC-SA 4.0 (Non-commercial use only)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    recall_score, precision_score, confusion_matrix
)
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Configure matplotlib and seaborn defaults
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=UserWarning)

def get_metrics(y_true, y_pred, y_classes):
    """
    Calculate various classification metrics for multi-class problems.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_classes : array-like
        All possible class labels
        
    Returns:
    --------
    dict : Dictionary containing various metrics
    """
    try:
        all_labels = np.arange(len(y_classes))
        f1_scores_per_class = f1_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)

        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'Macro Recall': recall_score(y_true, y_pred, average='macro', labels=all_labels, zero_division=0),
            'Macro Precision': precision_score(y_true, y_pred, average='macro', labels=all_labels, zero_division=0),
            'Macro F1': f1_score(y_true, y_pred, average='macro', labels=all_labels, zero_division=0),
            'F1 Scores per Class': f1_scores_per_class
        }
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {}

def _save_and_close_plot(filename):
    """Helper function to save plot and close figure."""
    try:
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error saving plot to {filename}: {e}")
        plt.close()

def plot_numerical_distributions(df, numerical_cols, filename='numerical_distributions.png'):
    """Plot histograms for numerical columns."""
    try:
        plt.figure(figsize=(15, 10))
        valid_cols = [col for col in numerical_cols if col in df.columns]
        num_plots = min(len(valid_cols), 9)
        
        for i, col in enumerate(valid_cols[:num_plots], 1):
            plt.subplot(3, 3, i)
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
        
        _save_and_close_plot(filename)
    except Exception as e:
        print(f"Error in plot_numerical_distributions: {e}")
        plt.close()

def plot_correlation_heatmap(correlation_matrix, title='Correlation Heatmap', filename='correlation_heatmap.png', annot=True):
    """Plot a correlation heatmap."""
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=annot, cmap='coolwarm', linewidths=0.5, fmt=".2f")
        plt.title(title)
        _save_and_close_plot(filename)
    except Exception as e:
        print(f"Error in plot_correlation_heatmap: {e}")
        plt.close()

def plot_failure_rate_by_type(df, filename='failure_by_type.png'):
    """Plot machine failure rate by product type."""
    try:
        if 'Type' in df.columns and 'Machine failure' in df.columns:
            plt.figure(figsize=(10, 6))
            pd.crosstab(df['Type'], df['Machine failure'], normalize='index').plot(
                kind='bar', stacked=True, ax=plt.gca()
            )
            plt.title('Machine Failure Rate by Product Type')
            plt.ylabel('Proportion')
            plt.legend(['No Failure', 'Failure'])
            _save_and_close_plot(filename)
        else:
            print("Warning: 'Type' or 'Machine failure' column not found for failure rate plot.")
    except Exception as e:
        print(f"Error in plot_failure_rate_by_type: {e}")
        plt.close()

def plot_scatter(df, x_col, y_col, hue_col, title, filename, add_lines=None):
    """Plot scatter plot with optional reference lines."""
    try:
        plt.figure(figsize=(10, 6))
        if all(col in df.columns for col in [x_col, y_col, hue_col]):
            sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df, alpha=0.6)
            
            if add_lines:
                for line_type, val, col, style, lbl in add_lines:
                    if line_type == 'h':
                        plt.axhline(y=val, color=col, linestyle=style, label=lbl)
                    elif line_type == 'v':
                        plt.axvline(x=val, color=col, linestyle=style, label=lbl)
                plt.legend()
            
            plt.title(title)
            _save_and_close_plot(filename)
        else:
            print(f"Warning: Required columns not found for scatter plot")
            plt.close()
    except Exception as e:
        print(f"Error in plot_scatter: {e}")
        plt.close()

def plot_histogram(series, title, filename, bins=50, add_lines=None):
    """Plot histogram with optional reference lines."""
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(series, bins=bins, kde=True)
        
        if add_lines:
            for val, col, style, lbl in add_lines:
                plt.axvline(x=val, color=col, linestyle=style, label=lbl)
            plt.legend()
        
        plt.title(title)
        _save_and_close_plot(filename)
    except Exception as e:
        print(f"Error in plot_histogram: {e}")
        plt.close()

def plot_overstrain_analysis(df, filename='overstrain_analysis.png'):
    """Plot tool wear vs torque with type-specific thresholds."""
    try:
        plt.figure(figsize=(10, 6))
        required_cols = ['Type', 'Tool wear [min]', 'Torque [Nm]']
        
        if all(col in df.columns for col in required_cols):
            for product_type, threshold in zip(['L', 'M', 'H'], [11000, 12000, 13000]):
                subset = df[df['Type'] == product_type]
                if not subset.empty:
                    plt.scatter(
                        subset['Tool wear [min]'], subset['Torque [Nm]'],
                        alpha=0.5, label=f'Type {product_type}'
                    )
                    
                    min_wear = max(1, df['Tool wear [min]'].min())
                    max_wear = df['Tool wear [min]'].max()
                    x = np.linspace(min_wear, max_wear, 100)
                    y = threshold / x
                    plt.plot(x, y, '--', label=f'Threshold for Type {product_type}')
            
            plt.xlabel('Tool wear [min]')
            plt.ylabel('Torque [Nm]')
            plt.title('Tool Wear vs Torque with Overstrain Thresholds')
            plt.legend()
            _save_and_close_plot(filename)
        else:
            print("Warning: Required columns not found for overstrain analysis")
            plt.close()
    except Exception as e:
        print(f"Error in plot_overstrain_analysis: {e}")
        plt.close()

def plot_failure_distributions(failures_df, filename='failure_distributions.png'):
    """Plot various distributions related to failures."""
    try:
        if failures_df.empty:
            print("Warning: No failure data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Count of each failure type
        failure_types = ['TWF', 'HDF', 'PWF', 'OSF']
        existing_types = [ft for ft in failure_types if ft in failures_df.columns]
        
        if existing_types:
            failure_counts = {ft: failures_df[ft].sum() for ft in existing_types}
            axes[0, 0].bar(failure_counts.keys(), failure_counts.values(), color='crimson')
            axes[0, 0].set_title('Count of Each Failure Type')
            axes[0, 0].set_ylabel('Number of Failures')
        
        # Failures by product type
        if 'Type' in failures_df.columns:
            type_counts = failures_df['Type'].value_counts()
            if not type_counts.empty:
                axes[0, 1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Failures by Product Type')
        
        # Tool wear distribution
        if 'Tool wear [min]' in failures_df.columns:
            sns.histplot(failures_df['Tool wear [min]'], bins=30, kde=True, color='red', ax=axes[1, 0])
            axes[1, 0].set_title('Tool Wear Distribution (Failures Only)')
        
        # Power distribution
        if 'Power [W]' in failures_df.columns:
            sns.histplot(failures_df['Power [W]'], bins=30, kde=True, color='orange', ax=axes[1, 1])
            axes[1, 1].set_title('Power Distribution (Failures Only)')
        
        _save_and_close_plot(filename)
    except Exception as e:
        print(f"Error in plot_failure_distributions: {e}")
        plt.close()

def plot_failure_correlation(failures_df, filename='failure_correlation.png'):
    """Plot correlation heatmap between different failure types."""
    try:
        plt.figure(figsize=(10, 8))
        failure_cols = [col for col in ['TWF', 'HDF', 'PWF', 'OSF'] if col in failures_df.columns]
        
        if failure_cols and len(failure_cols) > 1:
            failure_correlation = failures_df[failure_cols].corr()
            sns.heatmap(failure_correlation, annot=True, cmap='Reds', linewidths=0.5)
            plt.title('Correlation Between Failure Types')
            _save_and_close_plot(filename)
        else:
            print("Warning: Insufficient failure columns for correlation analysis")
            plt.close()
    except Exception as e:
        print(f"Error in plot_failure_correlation: {e}")
        plt.close()

def plot_detailed_failure_analysis(failures_df, filename='failure_detailed_analysis.png'):
    """Plot detailed scatter, box, and KDE plots for failure analysis."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Temperature difference vs speed
        if all(col in failures_df.columns for col in ['Rotational speed [rpm]', 'Temp Difference', 'HDF']):
            sns.scatterplot(
                x='Rotational speed [rpm]', y='Temp Difference', hue='HDF',
                palette={0: 'blue', 1: 'red'}, data=failures_df, ax=axes[0, 0]
            )
            axes[0, 0].axhline(y=8.6, color='black', linestyle='--')
            axes[0, 0].axvline(x=1380, color='black', linestyle='--')
            axes[0, 0].set_title('Temperature Difference vs Speed')
            axes[0, 0].legend(title='HDF Failure')
        
        # Tool wear vs torque
        if all(col in failures_df.columns for col in ['Tool wear [min]', 'Torque [Nm]', 'OSF']):
            sns.scatterplot(
                x='Tool wear [min]', y='Torque [Nm]', hue='OSF',
                palette={0: 'blue', 1: 'red'}, data=failures_df, ax=axes[0, 1]
            )
            axes[0, 1].set_title('Tool Wear vs Torque')
            axes[0, 1].legend(title='OSF Failure')
        
        # Torque by failure type
        failure_types = [col for col in ['TWF', 'HDF', 'PWF', 'OSF'] if col in failures_df.columns]
        if failure_types and 'Torque [Nm]' in failures_df.columns:
            melted_data = pd.melt(
                failures_df[['Torque [Nm]'] + failure_types],
                id_vars=['Torque [Nm]'], value_vars=failure_types,
                var_name='Failure Type', value_name='Failed'
            )
            melted_data = melted_data[melted_data['Failed'] == 1]
            if not melted_data.empty:
                sns.boxplot(x='Failure Type', y='Torque [Nm]', data=melted_data, ax=axes[1, 0])
            axes[1, 0].set_title('Torque Distribution by Failure Type')
        
        # Air temperature distribution by failure type
        if 'Air temperature [K]' in failures_df.columns and failure_types:
            for failure_type, color in zip(failure_types, ['red', 'blue', 'green', 'orange']):
                subset = failures_df[failures_df[failure_type] == 1]
                if len(subset) > 0:
                    sns.kdeplot(subset['Air temperature [K]'], label=failure_type, color=color, ax=axes[1, 1])
            axes[1, 1].legend()
            axes[1, 1].set_title('Air Temperature Distribution by Failure Type')
        
        _save_and_close_plot(filename)
    except Exception as e:
        print(f"Error in plot_detailed_failure_analysis: {e}")
        plt.close()

def plot_failure_relationships_pairplot(failures_df, failure_features, filename='failure_relationships.png'):
    """Plot pairplot showing relationships between features for failures."""
    try:
        valid_features = [f for f in failure_features if f in failures_df.columns]
        
        if valid_features and 'Type' in failures_df.columns and len(failures_df) > 1:
            # Create the pairplot
            g = sns.pairplot(
                failures_df, vars=valid_features, hue='Type',
                diag_kind='kde', plot_kws={'alpha': 0.6}
            )
            g.fig.suptitle('Relationships Between Features for Failed Machines', y=1.02)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("Warning: Insufficient data/columns for pairplot")
    except Exception as e:
        print(f"Error in plot_failure_relationships_pairplot: {e}")
        plt.close()

def plot_engineered_feature_distributions(df, features_to_plot, hue_col, filename='engineered_features.png'):
    """Plot distributions of engineered features, colored by hue column."""
    try:
        valid_features = [f for f in features_to_plot if f in df.columns]
        
        if not valid_features or hue_col not in df.columns:
            print("Warning: Cannot plot engineered features. Features or hue column missing.")
            return
        
        num_plots = min(len(valid_features), 6)
        plt.figure(figsize=(15, 10))
        
        for i, col in enumerate(valid_features[:num_plots], 1):
            plt.subplot(2, 3, i)
            sns.histplot(data=df, x=col, kde=True, hue=hue_col, palette={0: 'blue', 1: 'red'})
            plt.title(f'Distribution of {col}')
        
        _save_and_close_plot(filename)
    except Exception as e:
        print(f"Error in plot_engineered_feature_distributions: {e}")
        plt.close()

def plot_scaled_feature_distributions(df, features_to_plot, filename='scaled_features_distributions.png'):
    """Plot distributions (KDE) of scaled features."""
    try:
        valid_features = [f for f in features_to_plot if f in df.columns]
        
        if not valid_features:
            print("Warning: No valid features to plot scaled distributions.")
            return
        
        num_plots = min(len(valid_features), 9)
        plt.figure(figsize=(15, 10))
        
        for i, col in enumerate(valid_features[:num_plots], 1):
            plt.subplot(3, 3, i)
            sns.kdeplot(df[col], fill=True)
            plt.title(f'Distribution of {col}')
        
        _save_and_close_plot(filename)
    except Exception as e:
        print(f"Error in plot_scaled_feature_distributions: {e}")
        plt.close()

def plot_pca_explained_variance(explained_variance, filename='pca_explained_variance.png'):
    """Plot explained variance ratio for PCA components."""
    try:
        plt.figure(figsize=(10, 6))
        n_components = len(explained_variance)
        plt.bar(range(1, n_components + 1), explained_variance)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Principal Components')
        plt.xticks(range(1, min(n_components + 1, 21)))  # Limit x-axis labels
        _save_and_close_plot(filename)
    except Exception as e:
        print(f"Error in plot_pca_explained_variance: {e}")
        plt.close()

def plot_pca_cumulative_variance(explained_variance, filename='pca_cumulative_variance.png'):
    """Plot cumulative explained variance for PCA components."""
    try:
        plt.figure(figsize=(10, 6))
        cumulative_variance = np.cumsum(explained_variance)
        plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
        plt.axhline(y=0.99, color='g', linestyle='--', label='99% Explained Variance')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        _save_and_close_plot(filename)
    except Exception as e:
        print(f"Error in plot_pca_cumulative_variance: {e}")
        plt.close()

def plot_pca_scatter(principal_components, y_target, explained_variance, dim=2, filename='pca_scatter.png'):
    """Plot first 2 or 3 principal components."""
    try:
        if principal_components.shape[1] < dim:
            print(f"Warning: Not enough components ({principal_components.shape[1]}) for {dim}D PCA plot.")
            return
        
        if dim == 2:
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                principal_components[:, 0], principal_components[:, 1], 
                c=y_target, cmap='viridis', alpha=0.6, edgecolors='w', s=50
            )
            plt.colorbar(scatter, label='Target Class')
            plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
            plt.title('PCA: First Two Principal Components')
            plt.grid(True, alpha=0.3)
            _save_and_close_plot(filename)
            
        elif dim == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                principal_components[:, 0], principal_components[:, 1], principal_components[:, 2],
                c=y_target, cmap='viridis', alpha=0.6, s=50
            )
            ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
            ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
            ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
            ax.set_title('PCA: First Three Principal Components')
            plt.colorbar(scatter, ax=ax, label='Target Class')
            _save_and_close_plot(filename)
    except Exception as e:
        print(f"Error in plot_pca_scatter: {e}")
        plt.close()

def plot_pca_biplot(principal_components, pca_model, feature_names, y_target, explained_variance, filename='pca_biplot.png'):
    """Create a PCA biplot."""
    try:
        if principal_components.shape[1] < 2:
            print("Warning: Not enough components for biplot.")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Plot data points
        plt.scatter(
            principal_components[:, 0], principal_components[:, 1], 
            c=y_target, cmap='viridis', alpha=0.4, s=40
        )
        
        # Plot feature vectors
        feature_vectors = pca_model.components_.T
        max_comp_abs = np.abs(principal_components[:, :2]).max()
        max_vec_abs = np.abs(feature_vectors[:, :2]).max()
        
        if max_vec_abs > 0:
            arrow_scale = max_comp_abs / max_vec_abs * 0.5
            
            for i, (feature, vec) in enumerate(zip(feature_names[:len(feature_vectors)], feature_vectors)):
                plt.arrow(
                    0, 0, arrow_scale * vec[0], arrow_scale * vec[1],
                    head_width=0.05 * arrow_scale, head_length=0.05 * arrow_scale,
                    fc='red', ec='red', alpha=0.7
                )
                plt.text(
                    arrow_scale * vec[0] * 1.15, arrow_scale * vec[1] * 1.15, 
                    feature[:10], color='red', ha='center', va='center', fontsize=8
                )
        
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
        plt.title('PCA Biplot')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.colorbar(label='Target Class')
        _save_and_close_plot(filename)
    except Exception as e:
        print(f"Error in plot_pca_biplot: {e}")
        plt.close()

def plot_target_distribution(target_series, title, filename, labels=None):
    """Plot target variable distribution."""
    try:
        plt.figure(figsize=(10, 6))
        counts = target_series.value_counts()
        
        if labels is not None:
            counts = counts.sort_index()
            x_labels = labels
        else:
            x_labels = counts.index
        
        if len(x_labels) == len(counts):
            plt.bar(range(len(counts)), counts.values)
            plt.xticks(range(len(counts)), x_labels, rotation=45, ha='right')
            plt.title(title)
            plt.xlabel('Target Class')
            plt.ylabel('Count')
            _save_and_close_plot(filename)
        else:
            print(f"Warning: Mismatch between labels and counts for {filename}")
            plt.close()
    except Exception as e:
        print(f"Error in plot_target_distribution: {e}")
        plt.close()

def plot_feature_importance(importance_df, top_n, title, filename):
    """Plot feature importance as horizontal bar chart."""
    try:
        plt.figure(figsize=(10, 8))
        data_to_plot = importance_df.head(min(top_n, len(importance_df)))
        
        if not data_to_plot.empty:
            sns.barplot(x='Importance', y='Feature', data=data_to_plot, orient='h')
            plt.title(title)
            plt.xlabel('Importance')
            _save_and_close_plot(filename)
        else:
            print("Warning: No data to plot for feature importance")
            plt.close()
    except Exception as e:
        print(f"Error in plot_feature_importance: {e}")
        plt.close()

def plot_confusion_matrix(cm, class_names, title, filename):
    """Plot confusion matrix heatmap."""
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        _save_and_close_plot(filename)
    except Exception as e:
        print(f"Error in plot_confusion_matrix: {e}")
        plt.close()