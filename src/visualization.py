
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import os

class Visualizer:
    def __init__(self, config):
        self.config = config
        
    def plot_roc_curves(self, X, y, final_models, final_results, feature_sets):
        """Plot ROC curves for all models"""
        print("Generating ROC curves...")
        
        plt.figure(figsize=(12, 9))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, model) in enumerate(final_models.items()):
            # Get the best feature set for this model
            best_feature_set = final_results[model_name]['best_features']
            X_features = X[feature_sets[best_feature_set]]
            
            # Get prediction probabilities
            y_pred_proba = model.predict_proba(X_features)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            auc_score = final_results[model_name]['roc_auc']
            
            # Plot ROC curve
            plt.plot(fpr, tpr, linewidth=3, color=colors[i % len(colors)],
                    label=f'{model_name} (AUC = {auc_score:.4f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig("outputs/figures/roc_curves_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_performance_heatmap(self, final_results):
        """Create performance heatmap for all models"""
        print("Generating performance heatmap...")
        
        # Prepare data for heatmap
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        models = list(final_results.keys())
        
        heatmap_data = []
        for model in models:
            row = [final_results[model][metric] for metric in metrics]
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                                index=models)
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_df, annot=True, cmap='YlOrRd', fmt='.4f', 
                   cbar_kws={'label': 'Score'}, square=True)
        plt.title('Model Performance Comparison Heatmap', fontsize=16, fontweight='bold')
        plt.ylabel('Models', fontsize=12)
        plt.xlabel('Metrics', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        plt.savefig("outputs/figures/model_performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_usage_analysis(self, final_results, feature_sets):
        """Analyze which feature sets were selected by each model"""
        print("Generating feature usage analysis...")
        
        # Count feature set usage
        feature_usage = {}
        for model, results in final_results.items():
            best_features = results['best_features']
            if best_features not in feature_usage:
                feature_usage[best_features] = []
            feature_usage[best_features].append(model)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature set usage count
        feature_sets_names = list(feature_usage.keys())
        usage_counts = [len(models) for models in feature_usage.values()]
        
        ax1.bar(feature_sets_names, usage_counts, color=['skyblue', 'lightgreen', 'salmon'])
        ax1.set_xlabel('Feature Selection Method')
        ax1.set_ylabel('Number of Models Using This Method')
        ax1.set_title('Feature Selection Method Usage by Models')
        ax1.tick_params(axis='x', rotation=45)
        
        # Number of features used
        models = list(final_results.keys())
        n_features = [final_results[model]['n_features'] for model in models]
        
        ax2.bar(models, n_features, color=['blue', 'red', 'green', 'orange'])
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Number of Features Used')
        ax2.set_title('Number of Features Used by Each Model')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("outputs/figures/feature_usage_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report_plot(self, final_results):
        """Create a comprehensive summary plot"""
        print("Generating summary report plot...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('HIGGS Dataset ML Pipeline - Summary Report', fontsize=16, fontweight='bold')
        
        models = list(final_results.keys())
        
        # 1. ROC-AUC Scores
        roc_scores = [final_results[model]['roc_auc'] for model in models]
        colors = ['gold' if score == max(roc_scores) else 'lightblue' for score in roc_scores]
        
        bars1 = ax1.bar(models, roc_scores, color=colors)
        ax1.set_ylabel('ROC-AUC Score')
        ax1.set_title('ROC-AUC Performance by Model')
        ax1.set_ylim([min(roc_scores) - 0.02, max(roc_scores) + 0.02])
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars1, roc_scores):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. All Metrics Comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [final_results[model][metric] for model in models]
            ax2.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Score')
        ax2.set_title('All Metrics Comparison')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Feature Usage
        feature_methods = [final_results[model]['best_features'] for model in models]
        n_features = [final_results[model]['n_features'] for model in models]
        
        colors_features = {'all_features': 'red', 'anova_f_features': 'blue', 'mutual_info_features': 'green'}
        colors = [colors_features.get(method, 'gray') for method in feature_methods]
        
        bars3 = ax3.bar(models, n_features, color=colors)
        ax3.set_ylabel('Number of Features')
        ax3.set_title('Feature Count by Model')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add legend for feature methods
        unique_methods = list(set(feature_methods))
        legend_elements = [plt.Rectangle((0,0),1,1, color=colors_features.get(method, 'gray'), 
                                       label=method.replace('_', ' ').title()) for method in unique_methods]
        ax3.legend(handles=legend_elements, loc='upper right')
        
        # 4. Best Model Highlight
        best_model = max(final_results.keys(), key=lambda x: final_results[x]['roc_auc'])
        best_metrics = final_results[best_model]
        
        ax4.axis('off')
        ax4.text(0.5, 0.8, '🏆 BEST MODEL', fontsize=20, fontweight='bold', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.text(0.5, 0.6, f'{best_model}', fontsize=18, fontweight='bold', 
                ha='center', va='center', transform=ax4.transAxes, color='red')
        
        summary_text = f"""ROC-AUC: {best_metrics['roc_auc']:.4f}
Accuracy: {best_metrics['accuracy']:.4f}
Precision: {best_metrics['precision']:.4f}
Recall: {best_metrics['recall']:.4f}
F1-Score: {best_metrics['f1']:.4f}

Features: {best_metrics['best_features'].replace('_', ' ').title()}
Count: {best_metrics['n_features']} features"""
        
        ax4.text(0.5, 0.3, summary_text, fontsize=12, ha='center', va='center', 
                transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig("outputs/figures/summary_report.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_all_plots(self, outlier_info, feature_sets, nested_cv_results, 
                        final_results, X, y, final_models):
        """Create all visualizations"""
        print("Creating all visualizations...")
        
        # ROC Curves
        self.plot_roc_curves(X, y, final_models, final_results, feature_sets)
        
        # Performance Heatmap
        self.plot_model_performance_heatmap(final_results)
        
        # Feature Usage Analysis
        self.plot_feature_usage_analysis(final_results, feature_sets)
        
        # Summary Report
        self.create_summary_report_plot(final_results)
        
        print("All visualizations saved to outputs/figures/")
