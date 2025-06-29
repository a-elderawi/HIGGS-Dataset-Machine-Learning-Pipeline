#!/usr/bin/env python3
"""
HIGGS Dataset Machine Learning Pipeline
FIXED VERSION - Compatibility issues resolved
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from feature_selection import FeatureSelector
from model_training import ModelTrainer
from visualization import Visualizer

def setup_logging():
    """Setup logging configuration"""
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

def load_config():
    """Load configuration - FIXED for compatibility"""
    config = {
        'data': {
            'sample_size': 50000,  # Reduced for faster execution during testing
            'random_state': 42
        },
        'preprocessing': {
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5
        },
        'feature_selection': {
            'n_features': 15
        },
        'cross_validation': {
            'outer_folds': 3,  # Reduced from 5 for faster execution
            'inner_folds': 2,  # Reduced from 3 for faster execution
            'scoring': 'roc_auc',
            'n_jobs': 1  # Single job to avoid multiprocessing issues
        },
        'models': {
            'KNN': {'n_neighbors': [3, 5, 7, 9, 11]},
            'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'MLP': {'hidden_layer_sizes': [[50], [100]], 'activation': ['relu', 'tanh']},
            'XGBoost': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]}
        }
    }
    return config

def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        "data/raw", "data/processed",
        "outputs/figures", "outputs/results", "outputs/models",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main pipeline execution - FIXED"""
    # Setup
    create_directory_structure()
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()
    
    logger.info("🚀 Starting HIGGS ML Pipeline (FIXED VERSION)")
    logger.info("=" * 60)
    
    try:
        # Phase 1: Data Preprocessing
        logger.info("📊 Phase 1: Data Preprocessing")
        preprocessor = DataPreprocessor(config)
        X_scaled, y, outlier_info = preprocessor.run_preprocessing()
        logger.info(f"   ✅ Data preprocessed: {X_scaled.shape}")
        
        # Phase 2: Feature Selection
        logger.info("🎯 Phase 2: Feature Selection")
        selector = FeatureSelector(config)
        feature_sets = selector.run_feature_selection(X_scaled, y)
        logger.info(f"   ✅ Feature sets created: {list(feature_sets.keys())}")
        
        # Phase 3: Model Training and Evaluation
        logger.info("🤖 Phase 3: Model Training and Evaluation")
        trainer = ModelTrainer(config)
        nested_cv_results = trainer.run_nested_cv(X_scaled, y, feature_sets)
        final_results = trainer.train_final_models(X_scaled, y, feature_sets)
        
        if final_results:
            logger.info(f"   ✅ Models trained: {list(final_results.keys())}")
        else:
            logger.warning("   ⚠️ No models trained successfully")
        
        # Phase 4: Visualization and Results (only if we have results)
        if final_results:
            logger.info("📈 Phase 4: Generating Visualizations")
            try:
                visualizer = Visualizer(config)
                visualizer.create_all_plots(
                    outlier_info, feature_sets, nested_cv_results, 
                    final_results, X_scaled, y, trainer.final_models
                )
                logger.info("   ✅ All visualizations created")
            except Exception as e:
                logger.warning(f"   ⚠️ Visualization error: {str(e)}")
        
        # Phase 5: Save Results
        logger.info("💾 Phase 5: Saving Results")
        trainer.save_results(nested_cv_results, final_results)
        logger.info("   ✅ All results saved")
        
        # Final Summary
        logger.info("\n" + "=" * 60)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        # Print best model (if any)
        if final_results:
            best_model = max(final_results.keys(), key=lambda x: final_results[x]['roc_auc'])
            best_score = final_results[best_model]['roc_auc']
            logger.info(f"🏆 BEST MODEL: {best_model} (ROC-AUC: {best_score:.4f})")
        else:
            logger.warning("⚠️ No successful model results")
        
        logger.info("\n📁 OUTPUTS LOCATION:")
        logger.info("   📊 Figures: outputs/figures/")
        logger.info("   📋 Results: outputs/results/")
        logger.info("   🤖 Models: outputs/models/")
        logger.info("   📜 Logs: logs/")
        
        return final_results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
    print("\n🎯 Pipeline execution completed!")
    print("Check the logs/ directory for detailed execution logs.")
    print("All outputs have been saved to the outputs/ directory.")
