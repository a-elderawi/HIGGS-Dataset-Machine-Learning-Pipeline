import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
import json

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """Load and sample HIGGS dataset"""
        print("Loading HIGGS dataset...")
        
        try:
            data_path = "data/raw/HIGGS.csv"
            if os.path.exists(data_path):
                feature_names = ['target'] + [f'feature_{i}' for i in range(28)]
                data = pd.read_csv(data_path, names=feature_names, 
                                 nrows=self.config['data']['sample_size'])
                print(f"Loaded real HIGGS dataset: {data.shape}")
            else:
                raise FileNotFoundError("HIGGS.csv not found")
                
        except:
            print("HIGGS.csv not found. Generating synthetic data for demonstration...")
            np.random.seed(self.config['data']['random_state'])
            n_samples = self.config['data']['sample_size']
            
            # Create correlated features (similar to physics data)
            X = np.random.randn(n_samples, 28)
            for i in range(1, 28):
                X[:, i] += 0.2 * X[:, 0] + 0.1 * np.random.randn(n_samples)
            
            # Create target with some relationship to features
            y = (X[:, 0] + X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.5) > 0
            y = y.astype(int)
            
            feature_names = [f'feature_{i}' for i in range(28)]
            data = pd.DataFrame(X, columns=feature_names)
            data['target'] = y
            
        # Save processed sample
        os.makedirs("data/processed", exist_ok=True)
        data.to_csv("data/processed/higgs_sample_100k.csv", index=False)
        
        return data
    
    def analyze_outliers(self, data):
        """Analyze and handle outliers using IQR method"""
        print("Analyzing outliers...")
        
        X = data.drop('target', axis=1)
        threshold = self.config['preprocessing']['outlier_threshold']
        outlier_info = {}
        
        for column in X.columns:
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = X[(X[column] < lower_bound) | (X[column] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(X)) * 100
            
            outlier_info[column] = {
                'count': outlier_count,
                'percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        # Create outlier visualization
        self._plot_outlier_analysis(X, outlier_info)
        
        # Handle outliers by capping
        X_cleaned = X.copy()
        for column in X.columns:
            info = outlier_info[column]
            X_cleaned[column] = X_cleaned[column].clip(
                lower=info['lower_bound'], 
                upper=info['upper_bound']
            )
        
        data_cleaned = X_cleaned.copy()
        data_cleaned['target'] = data['target']
        
        return data_cleaned, outlier_info
    
    def _plot_outlier_analysis(self, X, outlier_info):
        """Create outlier analysis visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Outlier Analysis - Box Plots (First 6 Features)', fontsize=16)
        
        for i, column in enumerate(X.columns[:6]):
            row, col = i // 3, i % 3
            axes[row, col].boxplot(X[column])
            axes[row, col].set_title(f'{column}\n({outlier_info[column]["percentage"]:.1f}% outliers)')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs("outputs/figures", exist_ok=True)
        plt.savefig("outputs/figures/outlier_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save outlier statistics
        outlier_df = pd.DataFrame(outlier_info).T
        os.makedirs("outputs/results", exist_ok=True)
        outlier_df.to_csv("outputs/results/outlier_statistics.csv")
    
    def scale_features(self, data):
        """Scale features to [0,1] range"""
        print("Scaling features...")
        
        X = data.drop('target', axis=1)
        y = data['target']
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        print(f"Features scaled to range: [{X_scaled_df.min().min():.3f}, {X_scaled_df.max().max():.3f}]")
        
        return X_scaled_df, y
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        data = self.load_data()
        data_cleaned, outlier_info = self.analyze_outliers(data)
        X_scaled, y = self.scale_features(data_cleaned)
        return X_scaled, y, outlier_info