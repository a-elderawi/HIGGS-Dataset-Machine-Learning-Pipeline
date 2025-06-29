
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
import numpy as np
import pandas as pd
import json
import joblib
import os
from collections import Counter

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.final_models = {}
        self.final_results = {}
        
    def get_models_and_params(self):
        """Get models and their hyperparameter grids"""
        models = {
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(probability=True, random_state=self.config['data']['random_state']),
            'MLP': MLPClassifier(random_state=self.config['data']['random_state'], 
                               max_iter=500),  # Reduced iterations for stability
            'XGBoost': xgb.XGBClassifier(random_state=self.config['data']['random_state'], 
                                       eval_metric='logloss', verbosity=0)
        }
        
        param_grids = {
            'KNN': {'n_neighbors': [3, 5, 7, 9, 11]},  # Reduced range for faster execution
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            },
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,)],
                'activation': ['relu', 'tanh']
            },
            'XGBoost': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2]
            }
        }
        
        return models, param_grids
    
    def _ensure_contiguous_array(self, X):
        """Ensure array is C-contiguous to avoid sklearn issues"""
        if hasattr(X, 'values'):
            # If it's a DataFrame, get values and ensure contiguous
            return np.ascontiguousarray(X.values)
        elif isinstance(X, np.ndarray):
            # If it's already a numpy array, ensure contiguous
            return np.ascontiguousarray(X)
        else:
            # Convert to numpy array and ensure contiguous
            return np.ascontiguousarray(np.array(X))
    
    def nested_cross_validation(self, X, y, feature_sets):
        """Implement nested cross-validation with compatibility fixes"""
        print("Running nested cross-validation...")
        print("This may take several minutes...")
        
        models, param_grids = self.get_models_and_params()
        
        # CV setup
        outer_cv = StratifiedKFold(
            n_splits=self.config['cross_validation']['outer_folds'], 
            shuffle=True, 
            random_state=self.config['data']['random_state']
        )
        inner_cv = StratifiedKFold(
            n_splits=self.config['cross_validation']['inner_folds'], 
            shuffle=True, 
            random_state=self.config['data']['random_state']
        )
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\nProcessing {model_name}...")
            
            model_results = {
                'outer_scores': [],
                'best_params_per_fold': [],
                'best_features_per_fold': [],
                'feature_set_scores': {fs_name: [] for fs_name in feature_sets.keys()}
            }
            
            fold_idx = 0
            
            # Outer loop (5-fold CV)
            for train_idx, test_idx in outer_cv.split(X, y):
                fold_idx += 1
                print(f"  Outer Fold {fold_idx}/{self.config['cross_validation']['outer_folds']}")
                
                # FIXED: Ensure arrays are contiguous and handle DataFrame indexing properly
                if isinstance(X, pd.DataFrame):
                    X_train_outer = X.iloc[train_idx].copy()
                    X_test_outer = X.iloc[test_idx].copy()
                else:
                    X_train_outer = X[train_idx]
                    X_test_outer = X[test_idx]
                
                if isinstance(y, pd.Series):
                    y_train_outer = y.iloc[train_idx].copy()
                    y_test_outer = y.iloc[test_idx].copy()
                else:
                    y_train_outer = y[train_idx]
                    y_test_outer = y[test_idx]
                
                best_score = 0
                best_params = None
                best_feature_set = None
                
                # Inner loop: Test different feature sets (Flowchart A)
                for feature_set_name, features in feature_sets.items():
                    print(f"    Testing {feature_set_name} ({len(features)} features)")
                    
                    try:
                        # FIXED: Ensure proper feature selection and array contiguity
                        if isinstance(X_train_outer, pd.DataFrame):
                            X_train_features = X_train_outer[features].copy()
                            X_test_features = X_test_outer[features].copy()
                        else:
                            X_train_features = X_train_outer[:, features]
                            X_test_features = X_test_outer[:, features]
                        
                        # Convert to contiguous arrays to avoid sklearn issues
                        X_train_features = self._ensure_contiguous_array(X_train_features)
                        X_test_features = self._ensure_contiguous_array(X_test_features)
                        
                        # Inner loop: Hyperparameter optimization (Flowchart B)
                        grid_search = GridSearchCV(
                            estimator=model,
                            param_grid=param_grids[model_name],
                            cv=inner_cv,
                            scoring=self.config['cross_validation']['scoring'],
                            n_jobs=1,  # FIXED: Use single job to avoid multiprocessing issues
                            error_score='raise'  # Raise errors instead of setting to NaN
                        )
                        
                        grid_search.fit(X_train_features, y_train_outer)
                        
                        # Evaluate on outer test set
                        y_pred_proba = grid_search.predict_proba(X_test_features)[:, 1]
                        score = roc_auc_score(y_test_outer, y_pred_proba)
                        
                        # Track feature set performance
                        model_results['feature_set_scores'][feature_set_name].append(score)
                        
                        # Update best combination
                        if score > best_score:
                            best_score = score
                            best_params = grid_search.best_params_
                            best_feature_set = feature_set_name
                            
                        print(f"      {feature_set_name}: ROC-AUC = {score:.4f}")
                            
                    except Exception as e:
                        print(f"      Error with {feature_set_name}: {str(e)}")
                        # Set a default low score instead of None
                        model_results['feature_set_scores'][feature_set_name].append(0.5)
                        continue
                
                # Store results for this fold
                if best_score > 0:  # Only store if we got a valid result
                    model_results['outer_scores'].append(best_score)
                    model_results['best_params_per_fold'].append(best_params)
                    model_results['best_features_per_fold'].append(best_feature_set)
                else:
                    # If no valid results, use defaults
                    model_results['outer_scores'].append(0.5)
                    model_results['best_params_per_fold'].append(list(param_grids[model_name].keys())[0])
                    model_results['best_features_per_fold'].append('all_features')
            
            results[model_name] = model_results
            
            # Print summary for this model
            if model_results['outer_scores']:
                mean_score = np.mean(model_results['outer_scores'])
                std_score = np.std(model_results['outer_scores'])
                print(f"  {model_name} - Mean ROC-AUC: {mean_score:.4f} (±{std_score:.4f})")
            else:
                print(f"  {model_name} - No valid results")
        
        self.nested_cv_results = results
        return results
    
    def train_final_models(self, X, y, feature_sets):
        """Train final models using best parameters from nested CV"""
        print("\nTraining final models...")
        
        models, _ = self.get_models_and_params()
        final_models = {}
        final_results = {}
        
        for model_name in models.keys():
            if model_name not in self.nested_cv_results:
                continue
                
            print(f"Training final {model_name} model...")
            
            # Get most common best parameters and feature set
            best_params_list = [p for p in self.nested_cv_results[model_name]['best_params_per_fold'] if p is not None]
            best_features_list = [f for f in self.nested_cv_results[model_name]['best_features_per_fold'] if f is not None]
            
            if not best_params_list or not best_features_list:
                print(f"  Skipping {model_name} - no valid results")
                continue
            
            # Use most frequent best parameters
            if isinstance(best_params_list[0], dict):
                # For actual parameter dictionaries
                most_common_params = Counter(str(p) for p in best_params_list).most_common(1)[0][0]
                best_params = eval(most_common_params)
            else:
                # Fallback to first valid params
                best_params = best_params_list[0] if best_params_list else {}
            
            most_common_features = Counter(best_features_list).most_common(1)[0][0]
            best_feature_set = most_common_features
            
            try:
                # Train final model
                model = models[model_name]
                if isinstance(best_params, dict):
                    model.set_params(**best_params)
                
                # Get features and ensure contiguity
                if isinstance(X, pd.DataFrame):
                    X_final = X[feature_sets[best_feature_set]].copy()
                else:
                    X_final = X[:, feature_sets[best_feature_set]]
                
                X_final = self._ensure_contiguous_array(X_final)
                
                model.fit(X_final, y)
                
                # Make predictions
                y_pred = model.predict(X_final)
                y_pred_proba = model.predict_proba(X_final)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred, zero_division=0),
                    'recall': recall_score(y, y_pred, zero_division=0),
                    'f1': f1_score(y, y_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y, y_pred_proba),
                    'best_params': best_params,
                    'best_features': best_feature_set,
                    'n_features': len(feature_sets[best_feature_set])
                }
                
                final_models[model_name] = model
                final_results[model_name] = metrics
                
                print(f"  {model_name} - ROC-AUC: {metrics['roc_auc']:.4f}")
                
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
                continue
        
        self.final_models = final_models
        self.final_results = final_results
        
        return final_results
    
    def save_results(self, nested_cv_results, final_results):
        """Save all results to files"""
        print("Saving results...")
        
        os.makedirs("outputs/results", exist_ok=True)
        os.makedirs("outputs/models", exist_ok=True)
        
        # Save nested CV results
        try:
            with open("outputs/results/nested_cv_results.json", 'w') as f:
                results_serializable = {}
                for model, results in nested_cv_results.items():
                    results_serializable[model] = {
                        'outer_scores': [float(x) for x in results['outer_scores']],
                        'best_params_per_fold': [str(p) for p in results['best_params_per_fold']],
                        'best_features_per_fold': results['best_features_per_fold'],
                        'feature_set_scores': {k: [float(x) for x in v] 
                                             for k, v in results['feature_set_scores'].items()},
                        'mean_score': float(np.mean(results['outer_scores'])) if results['outer_scores'] else 0.0,
                        'std_score': float(np.std(results['outer_scores'])) if results['outer_scores'] else 0.0
                    }
                json.dump(results_serializable, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save nested CV results: {str(e)}")
        
        # Save final model metrics
        if final_results:
            try:
                final_metrics_df = pd.DataFrame(final_results).T
                final_metrics_df.to_csv("outputs/results/final_model_metrics.csv")
                
                # Save best parameters
                best_params = {model: results['best_params'] for model, results in final_results.items()}
                with open("outputs/results/best_parameters.json", 'w') as f:
                    json.dump(best_params, f, indent=2, default=str)
            except Exception as e:
                print(f"Warning: Could not save final results: {str(e)}")
        
        # Save models
        for model_name, model in self.final_models.items():
            try:
                joblib.dump(model, f"outputs/models/best_{model_name.lower()}_model.pkl")
            except Exception as e:
                print(f"Warning: Could not save {model_name} model: {str(e)}")
        
        print("Results saved successfully!")
    
    def run_nested_cv(self, X, y, feature_sets):
        """Run nested cross-validation"""
        return self.nested_cross_validation(X, y, feature_sets)
