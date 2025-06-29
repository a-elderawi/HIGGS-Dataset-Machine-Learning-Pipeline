
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json

class FeatureSelector:
    def __init__(self, config):
        self.config = config
        
    def apply_feature_selection(self, X, y):
        """Apply both ANOVA F-score and Mutual Information feature selection"""
        print("Applying feature selection...")
        
        k = self.config['feature_selection']['n_features']
        
        try:
            # Ensure inputs are properly formatted
            if isinstance(X, pd.DataFrame):
                X_array = np.ascontiguousarray(X.values)
                feature_names = X.columns.tolist()
            else:
                X_array = np.ascontiguousarray(X)
                feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
            
            if isinstance(y, pd.Series):
                y_array = np.ascontiguousarray(y.values)
            else:
                y_array = np.ascontiguousarray(y)
            
            # ANOVA F-score
            print("  Computing ANOVA F-scores...")
            f_selector = SelectKBest(score_func=f_classif, k=k)
            X_f_selected = f_selector.fit_transform(X_array, y_array)
            f_scores = f_selector.scores_
            f_selected_indices = f_selector.get_support(indices=True)
            f_selected_features = [feature_names[i] for i in f_selected_indices]
            
            # Mutual Information
            print("  Computing Mutual Information scores...")
            mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_mi_selected = mi_selector.fit_transform(X_array, y_array)
            mi_scores = mi_selector.scores_
            mi_selected_indices = mi_selector.get_support(indices=True)
            mi_selected_features = [feature_names[i] for i in mi_selected_indices]
            
            # Store selectors
            self.f_selector = f_selector
            self.mi_selector = mi_selector
            
            # Create feature importance DataFrames
            feature_importance_f = pd.DataFrame({
                'feature': feature_names,
                'f_score': f_scores,
                'selected': [name in f_selected_features for name in feature_names]
            }).sort_values('f_score', ascending=False)
            
            feature_importance_mi = pd.DataFrame({
                'feature': feature_names,
                'mi_score': mi_scores,
                'selected': [name in mi_selected_features for name in feature_names]
            }).sort_values('mi_score', ascending=False)
            
            # Create visualizations
            self._plot_feature_importance(feature_importance_f, feature_importance_mi, k)
            
            # Define feature sets
            feature_sets = {
                'all_features': feature_names,
                'anova_f_features': f_selected_features,
                'mutual_info_features': mi_selected_features
            }
            
            # Save feature selection results
            self._save_feature_selection_results(feature_importance_f, feature_importance_mi, feature_sets)
            
            print(f"  ANOVA F-score selected: {len(f_selected_features)} features")
            print(f"  Mutual Info selected: {len(mi_selected_features)} features")
            print(f"  Feature overlap: {len(set(f_selected_features) & set(mi_selected_features))} features")
            
            return feature_sets
            
        except Exception as e:
            print(f"Error in feature selection: {str(e)}")
            # Return default feature sets if feature selection fails
            feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]
            return {
                'all_features': feature_names,
                'anova_f_features': feature_names[:k],
                'mutual_info_features': feature_names[:k]
            }
    
    def _plot_feature_importance(self, feature_importance_f, feature_importance_mi, k):
        """Plot feature importance for both methods"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # ANOVA F-score
            top_f_features = feature_importance_f.head(k)
            colors_f = ['red' if selected else 'lightblue' for selected in top_f_features['selected']]
            y_pos = np.arange(len(top_f_features))
            
            axes[0].barh(y_pos, top_f_features['f_score'], color=colors_f)
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(top_f_features['feature'])
            axes[0].set_xlabel('ANOVA F-score')
            axes[0].set_title(f'Top {k} Features - ANOVA F-score')
            axes[0].invert_yaxis()
            axes[0].grid(True, alpha=0.3)
            
            # Mutual Information
            top_mi_features = feature_importance_mi.head(k)
            colors_mi = ['red' if selected else 'lightblue' for selected in top_mi_features['selected']]
            y_pos = np.arange(len(top_mi_features))
            
            axes[1].barh(y_pos, top_mi_features['mi_score'], color=colors_mi)
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(top_mi_features['feature'])
            axes[1].set_xlabel('Mutual Information Score')
            axes[1].set_title(f'Top {k} Features - Mutual Information')
            axes[1].invert_yaxis()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figures
            os.makedirs("outputs/figures", exist_ok=True)
            plt.savefig("outputs/figures/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            # Save individual plots
            self._save_individual_plots(top_f_features, top_mi_features, k)
            
        except Exception as e:
            print(f"Warning: Could not create feature importance plots: {str(e)}")
    
    def _save_individual_plots(self, top_f_features, top_mi_features, k):
        """Save individual feature importance plots"""
        try:
            # ANOVA F-score plot
            fig, ax = plt.subplots(figsize=(10, 8))
            colors_f = ['red' if selected else 'lightblue' for selected in top_f_features['selected']]
            y_pos = np.arange(len(top_f_features))
            
            ax.barh(y_pos, top_f_features['f_score'], color=colors_f)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_f_features['feature'])
            ax.set_xlabel('ANOVA F-score')
            ax.set_title(f'Top {k} Features - ANOVA F-score')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("outputs/figures/feature_importance_anova.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Mutual Information plot
            fig, ax = plt.subplots(figsize=(10, 8))
            colors_mi = ['red' if selected else 'lightblue' for selected in top_mi_features['selected']]
            y_pos = np.arange(len(top_mi_features))
            
            ax.barh(y_pos, top_mi_features['mi_score'], color=colors_mi)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_mi_features['feature'])
            ax.set_xlabel('Mutual Information Score')
            ax.set_title(f'Top {k} Features - Mutual Information')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("outputs/figures/feature_importance_mi.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not save individual plots: {str(e)}")
    
    def _save_feature_selection_results(self, feature_importance_f, feature_importance_mi, feature_sets):
        """Save feature selection results"""
        try:
            os.makedirs("outputs/results", exist_ok=True)
            
            # Save feature importance scores
            feature_importance_f.to_csv("outputs/results/anova_f_scores.csv", index=False)
            feature_importance_mi.to_csv("outputs/results/mutual_info_scores.csv", index=False)
            
            # Save feature sets
            with open("outputs/results/feature_sets.json", 'w') as f:
                json.dump(feature_sets, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save feature selection results: {str(e)}")
    
    def run_feature_selection(self, X, y):
        """Run complete feature selection pipeline"""
        return self.apply_feature_selection(X, y)

# ===========================
# FILE 5: src/visualization.py (FIXED)
# ===========================

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
        
        try:
            plt.figure(figsize=(12, 9))
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, (model_name, model) in enumerate(final_models.items()):
                try:
                    # Get the best feature set for this model
                    best_feature_set = final_results[model_name]['best_features']
                    
                    if isinstance(X, pd.DataFrame):
                        X_features = X[feature_sets[best_feature_set]].copy()
                        X_features = np.ascontiguousarray(X_features.values)
                    else:
                        X_features = X[:, feature_sets[best_feature_set]]
                        X_features = np.ascontiguousarray(X_features)
                    
                    # Get prediction probabilities
                    y_pred_proba = model.predict_proba(X_features)[:, 1]
                    
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(y, y_pred_proba)
                    auc_score = final_results[model_name]['roc_auc']
                    
                    # Plot ROC curve
                    plt.plot(fpr, tpr, linewidth=3, color=colors[i % len(colors)],
                            label=f'{model_name} (AUC = {auc_score:.4f})')
                            
                except Exception as e:
                    print(f"Warning: Could not plot ROC curve for {model_name}: {str(e)}")
                    continue
            
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
            os.makedirs("outputs/figures", exist_ok=True)
            plt.savefig("outputs/figures/roc_curves_comparison.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Warning: Could not create ROC curves: {str(e)}")
    
    def plot_model_performance_heatmap(self, final_results):
        """Create performance heatmap for all models"""
        print("Generating performance heatmap...")
        
        try:
            if not final_results:
                print("No results to plot heatmap")
                return
                
            # Prepare data for heatmap
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            models = list(final_results.keys())
            
            heatmap_data = []
            for model in models:
                row = [final_results[model].get(metric, 0.0) for metric in metrics]
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
            
        except Exception as e:
            print(f"Warning: Could not create performance heatmap: {str(e)}")
    
    def plot_feature_usage_analysis(self, final_results, feature_sets):
        """Analyze which feature sets were selected by each model"""
        print("Generating feature usage analysis...")
        
        try:
            if not final_results:
                print("No results for feature usage analysis")
                return
                
            # Count feature set usage
            feature_usage = {}
            for model, results in final_results.items():
                best_features = results.get('best_features', 'all_features')
                if best_features not in feature_usage:
                    feature_usage[best_features] = []
                feature_usage[best_features].append(model)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Feature set usage count
            feature_sets_names = list(feature_usage.keys())
            usage_counts = [len(models) for models in feature_usage.values()]
            
            colors = ['skyblue', 'lightgreen', 'salmon'][:len(feature_sets_names)]
            ax1.bar(feature_sets_names, usage_counts, color=colors)
            ax1.set_xlabel('Feature Selection Method')
            ax1.set_ylabel('Number of Models Using This Method')
            ax1.set_title('Feature Selection Method Usage by Models')
            ax1.tick_params(axis='x', rotation=45)
            
            # Number of features used
            models = list(final_results.keys())
            n_features = [final_results[model].get('n_features', 0) for model in models]
            
            colors2 = ['blue', 'red', 'green', 'orange'][:len(models)]
            ax2.bar(models, n_features, color=colors2)
            ax2.set_xlabel('Models')
            ax2.set_ylabel('Number of Features Used')
            ax2.set_title('Number of Features Used by Each Model')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig("outputs/figures/feature_usage_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Warning: Could not create feature usage analysis: {str(e)}")
    
    def create_summary_report_plot(self, final_results):
        """Create a comprehensive summary plot"""
        print("Generating summary report plot...")
        
        try:
            if not final_results:
                print("No results for summary report")
                return
                
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('HIGGS Dataset ML Pipeline - Summary Report', fontsize=16, fontweight='bold')
            
            models = list(final_results.keys())
            
            # 1. ROC-AUC Scores
            roc_scores = [final_results[model].get('roc_auc', 0.0) for model in models]
            colors = ['gold' if score == max(roc_scores) else 'lightblue' for score in roc_scores]
            
            bars1 = ax1.bar(models, roc_scores, color=colors)
            ax1.set_ylabel('ROC-AUC Score')
            ax1.set_title('ROC-AUC Performance by Model')
            ax1.set_ylim([max(0, min(roc_scores) - 0.02), max(roc_scores) + 0.02])
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
                values = [final_results[model].get(metric, 0.0) for model in models]
                ax2.bar(x + i*width, values, width, label=metric.capitalize())
            
            ax2.set_xlabel('Models')
            ax2.set_ylabel('Score')
            ax2.set_title('All Metrics Comparison')
            ax2.set_xticks(x + width * 1.5)
            ax2.set_xticklabels(models)
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Feature Usage
            feature_methods = [final_results[model].get('best_features', 'all_features') for model in models]
            n_features = [final_results[model].get('n_features', 0) for model in models]
            
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
            if models:
                best_model = max(final_results.keys(), key=lambda x: final_results[x].get('roc_auc', 0))
                best_metrics = final_results[best_model]
                
                ax4.axis('off')
                ax4.text(0.5, 0.8, '🏆 BEST MODEL', fontsize=20, fontweight='bold', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.text(0.5, 0.6, f'{best_model}', fontsize=18, fontweight='bold', 
                        ha='center', va='center', transform=ax4.transAxes, color='red')
                
                summary_text = f"""ROC-AUC: {best_metrics.get('roc_auc', 0):.4f}
Accuracy: {best_metrics.get('accuracy', 0):.4f}
Precision: {best_metrics.get('precision', 0):.4f}
Recall: {best_metrics.get('recall', 0):.4f}
F1-Score: {best_metrics.get('f1', 0):.4f}

Features: {best_metrics.get('best_features', 'N/A').replace('_', ' ').title()}
Count: {best_metrics.get('n_features', 0)} features"""
                
                ax4.text(0.5, 0.3, summary_text, fontsize=12, ha='center', va='center', 
                        transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            plt.tight_layout()
            plt.savefig("outputs/figures/summary_report.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Warning: Could not create summary report: {str(e)}")
    
    def create_all_plots(self, outlier_info, feature_sets, nested_cv_results, 
                        final_results, X, y, final_models):
        """Create all visualizations"""
        print("Creating all visualizations...")
        
        try:
            # ROC Curves
            if final_models and final_results:
                self.plot_roc_curves(X, y, final_models, final_results, feature_sets)
            
            # Performance Heatmap
            if final_results:
                self.plot_model_performance_heatmap(final_results)
            
            # Feature Usage Analysis
            if final_results and feature_sets:
                self.plot_feature_usage_analysis(final_results, feature_sets)
            
            # Summary Report
            if final_results:
                self.create_summary_report_plot(final_results)
            
            print("All visualizations completed!")
            
        except Exception as e:
            print(f"Error in visualization creation: {str(e)}")
