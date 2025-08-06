import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.logger import setup_logger
from src.models.predictive import F1PredictiveModels

logger = setup_logger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           model_name: str) -> Dict[str, float]:
        """Evaluate model predictions with comprehensive metrics"""
        try:
            metrics = {
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
                'max_error': np.max(np.abs(y_true - y_pred)),
                'median_error': np.median(np.abs(y_true - y_pred))
            }
            
            self.logger.info(f"Evaluation complete for {model_name}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating predictions: {e}")
            return {}
    
    def generate_prediction_report(self, session_id: str, 
                                 predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed prediction performance report"""
        try:
            # Aggregate prediction statistics
            accuracy_scores = []
            confidence_scores = []
            
            for pred in predictions:
                if 'actual' in pred and 'predicted' in pred:
                    error = abs(pred['actual'] - pred['predicted'])
                    accuracy = 1 - (error / pred['actual'])
                    accuracy_scores.append(accuracy)
                
                if 'confidence_interval' in pred:
                    ci_width = pred['confidence_interval'][1] - pred['confidence_interval'][0]
                    confidence_scores.append(ci_width)
            
            return {
                'session_id': session_id,
                'total_predictions': len(predictions),
                'avg_accuracy': np.mean(accuracy_scores) if accuracy_scores else 0,
                'avg_confidence_width': np.mean(confidence_scores) if confidence_scores else 0,
                'prediction_details': predictions[:10]  # First 10 for review
            }
            
        except Exception as e:
            self.logger.error(f"Error generating prediction report: {e}")
            return {}
        

# Example usage and testing for ModelEvaluator
if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    
    # Add src to path for imports (adjust as needed for your project structure)
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from src.models.predictive import F1PredictiveModels
        from src.database.connection import db_pool
        
        print("=== F1 Model Evaluation Demo ===\n")
        
        # Initialize evaluator and ML models
        evaluator = ModelEvaluator()
        ml_models = F1PredictiveModels()
        
        # Use actual session IDs from your database
        session_ids = ["2024_abu_dhabi_grand_prix_r", "2022_abu_dhabi_grand_prix_r"]
        
        print("1. Training models for evaluation...")
        print("-" * 40)
        
        # Train lap time models
        performances = ml_models.train_lap_time_model(session_ids)
        
        if performances:
            print("Lap Time Model Performance:")
            for model_name, performance in performances.items():
                if performance.mae != float('inf'):
                    print(f"  {model_name}:")
                    print(f"    MAE: {performance.mae:.3f}s")
                    print(f"    RMSE: {performance.rmse:.3f}s")
                    print(f"    R²: {performance.r2_score:.3f}")
                    print(f"    CV Score: {performance.cv_score:.3f}")
                else:
                    print(f"  {model_name}: Training failed")
        else:
            print("No models trained successfully")
            
        print("\n2. Training tire degradation model...")
        print("-" * 40)
        
        degradation_performance = ml_models.train_tire_degradation_model(session_ids)
        if degradation_performance.mae != float('inf'):
            print(f"Tire Degradation Model:")
            print(f"  MAE: {degradation_performance.mae:.4f}")
            print(f"  RMSE: {degradation_performance.rmse:.4f}")
            print(f"  R²: {degradation_performance.r2_score:.3f}")
        else:
            print("Tire degradation model training failed")
        
        print("\n3. Making predictions for evaluation...")
        print("-" * 40)
        
        # Generate sample predictions
        predictions = []
        drivers = ["VER", "HAM", "LEC", "RUS", "NOR"]
        
        for i, driver in enumerate(drivers):
            prediction = ml_models.predict_lap_time(
                session_id="2024_abu_dhabi_grand_prix_r",
                driver_code=driver,
                lap_number=20 + i * 2,
                tire_compound="MEDIUM",
                tire_age=15 + i * 3,
                position=i + 1
            )
            
            if prediction:
                # Simulate actual lap time for evaluation (in real scenario, this would come from data)
                simulated_actual = prediction.predicted_value + np.random.normal(0, 0.5)
                
                predictions.append({
                    'driver': driver,
                    'predicted': prediction.predicted_value,
                    'actual': simulated_actual,  # Simulated for demo
                    'confidence_interval': prediction.confidence_interval,
                    'model_name': prediction.model_name,
                    'feature_importance': prediction.feature_importance
                })
                
                print(f"  {driver}: {prediction.predicted_value:.3f}s "
                      f"(±{(prediction.confidence_interval[1] - prediction.confidence_interval[0])/2:.3f}s)")
        
        print("\n4. Evaluating prediction accuracy...")
        print("-" * 40)
        
        if predictions:
            # Extract actual vs predicted for evaluation
            y_true = np.array([p['actual'] for p in predictions])
            y_pred = np.array([p['predicted'] for p in predictions])
            
            # Evaluate predictions
            metrics = evaluator.evaluate_predictions(y_true, y_pred, "Combined_Models")
            
            print("Prediction Evaluation Metrics:")
            for metric, value in metrics.items():
                if metric == 'mape':
                    print(f"  {metric.upper()}: {value:.2f}%")
                else:
                    print(f"  {metric.upper()}: {value:.3f}")
            
            # Generate prediction report
            report = evaluator.generate_prediction_report("demo_session", predictions)
            
            print(f"\nPrediction Report Summary:")
            print(f"  Session ID: {report.get('session_id', 'N/A')}")
            print(f"  Total Predictions: {report.get('total_predictions', 0)}")
            print(f"  Average Accuracy: {report.get('avg_accuracy', 0):.1%}")
            print(f"  Average Confidence Width: {report.get('avg_confidence_width', 0):.3f}s")
            
            # Show detailed predictions
            print(f"\nDetailed Predictions (first 3):")
            for i, pred in enumerate(predictions[:3]):
                print(f"  {i+1}. Driver {pred['driver']}:")
                print(f"     Predicted: {pred['predicted']:.3f}s")
                print(f"     Actual: {pred['actual']:.3f}s") 
                print(f"     Error: {abs(pred['actual'] - pred['predicted']):.3f}s")
                print(f"     Model: {pred['model_name']}")
                
                # Show top feature importance
                top_features = list(pred['feature_importance'].items())[:3]
                print(f"     Key Factors: {', '.join([f'{k}: {v:.1%}' for k, v in top_features])}")
        
        print("\n5. Testing tire degradation predictions...")
        print("-" * 40)
        
        # Test tire degradation predictions
        degradation_predictions = []
        compounds = ["SOFT", "MEDIUM", "HARD"]
        
        for compound in compounds:
            degradation = ml_models.predict_tire_degradation(
                session_id="2024_abu_dhabi_grand_prix_r",
                driver_code="VER",
                tire_compound=compound,
                stint_length=20,
                initial_pace=89.5,
                avg_track_temp=45.0
            )
            
            if degradation is not None:
                degradation_predictions.append({
                    'compound': compound,
                    'predicted_degradation': degradation,
                    'simulated_actual': degradation + np.random.normal(0, 0.01)  # Simulated
                })
                print(f"  {compound}: {degradation:.4f}s/lap degradation")
        
        # Evaluate degradation predictions
        if degradation_predictions:
            deg_y_true = np.array([d['simulated_actual'] for d in degradation_predictions])
            deg_y_pred = np.array([d['predicted_degradation'] for d in degradation_predictions])
            
            deg_metrics = evaluator.evaluate_predictions(deg_y_true, deg_y_pred, "Tire_Degradation")
            
            print("\nTire Degradation Evaluation:")
            for metric, value in deg_metrics.items():
                if metric == 'mape':
                    print(f"  {metric.upper()}: {value:.2f}%")
                else:
                    print(f"  {metric.upper()}: {value:.4f}")
        
        print("\n6. Model performance visualization...")
        print("-" * 40)
        
        # Create performance visualization
        if predictions:
            plt.figure(figsize=(12, 8))
            
            # Actual vs Predicted plot
            plt.subplot(2, 2, 1)
            y_true_plot = [p['actual'] for p in predictions]
            y_pred_plot = [p['predicted'] for p in predictions]
            
            plt.scatter(y_true_plot, y_pred_plot, alpha=0.7, color='blue')
            plt.plot([min(y_true_plot), max(y_true_plot)], 
                    [min(y_true_plot), max(y_true_plot)], 'r--', alpha=0.8)
            plt.xlabel('Actual Lap Time (s)')
            plt.ylabel('Predicted Lap Time (s)')
            plt.title('Actual vs Predicted Lap Times')
            plt.grid(True, alpha=0.3)
            
            # Error distribution
            plt.subplot(2, 2, 2)
            errors = [abs(p['actual'] - p['predicted']) for p in predictions]
            plt.hist(errors, bins=5, alpha=0.7, color='green', edgecolor='black')
            plt.xlabel('Absolute Error (s)')
            plt.ylabel('Frequency')
            plt.title('Prediction Error Distribution')
            plt.grid(True, alpha=0.3)
            
            # Model confidence intervals
            plt.subplot(2, 2, 3)
            drivers = [p['driver'] for p in predictions]
            predicted_values = [p['predicted'] for p in predictions]
            confidence_widths = [(p['confidence_interval'][1] - p['confidence_interval'][0])/2 
                                for p in predictions]
            
            plt.errorbar(drivers, predicted_values, yerr=confidence_widths, 
                        fmt='o', capsize=5, capthick=2, alpha=0.8)
            plt.xlabel('Driver')
            plt.ylabel('Predicted Lap Time (s)')
            plt.title('Predictions with Confidence Intervals')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Feature importance (average across predictions)
            plt.subplot(2, 2, 4)
            all_features = {}
            for pred in predictions:
                for feature, importance in pred['feature_importance'].items():
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(importance)
            
            avg_importance = {k: np.mean(v) for k, v in all_features.items()}
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            features, importances = zip(*sorted_features)
            plt.bar(features, importances, alpha=0.7, color='orange')
            plt.xlabel('Features')
            plt.ylabel('Average Importance')
            plt.title('Top Feature Importance')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = Path("model_evaluation_results.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Performance visualization saved to: {plot_path}")
            
            # Show the plot
            plt.show()
        
        print("\n7. Model summary and recommendations...")
        print("-" * 40)
        
        # Get model summary
        summary = ml_models.get_model_performance_summary()
        
        print("Model Summary:")
        print(f"  Available models: {summary.get('available_models', [])}")
        print(f"  Saved models: {summary.get('saved_models', [])}")
        print(f"  Best lap time model: {summary.get('best_lap_time_model', 'N/A')}")
        
        # Recommendations based on performance
        if performances:
            best_model = min(performances.items(), key=lambda x: x[1].mae if x[1].mae != float('inf') else float('inf'))
            if best_model[1].mae != float('inf'):
                print(f"\nRecommendations:")
                print(f"  - Best performing model: {best_model[0]} (MAE: {best_model[1].mae:.3f}s)")
                
                if best_model[1].r2_score > 0.8:
                    print(f"  - Model shows good predictive power (R² = {best_model[1].r2_score:.3f})")
                elif best_model[1].r2_score > 0.6:
                    print(f"  - Model shows moderate predictive power (R² = {best_model[1].r2_score:.3f})")
                else:
                    print(f"  - Model needs improvement (R² = {best_model[1].r2_score:.3f})")
                
                if best_model[1].mae < 0.5:
                    print(f"  - Prediction accuracy is excellent (±{best_model[1].mae:.3f}s)")
                elif best_model[1].mae < 1.0:
                    print(f"  - Prediction accuracy is good (±{best_model[1].mae:.3f}s)")
                else:
                    print(f"  - Consider additional feature engineering (±{best_model[1].mae:.3f}s)")
        
        print("\n=== Evaluation Complete ===")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required modules are available and paths are correct")
        
        # Fallback demo with synthetic data
        print("\nRunning fallback demo with synthetic data...")
        
        evaluator = ModelEvaluator()
        
        # Generate synthetic evaluation data
        np.random.seed(42)
        n_samples = 50
        
        # Simulate lap times
        true_lap_times = np.random.normal(90, 2, n_samples)  # Around 90s ± 2s
        predicted_lap_times = true_lap_times + np.random.normal(0, 0.5, n_samples)  # Add prediction error
        
        # Evaluate synthetic predictions
        metrics = evaluator.evaluate_predictions(true_lap_times, predicted_lap_times, "Synthetic_Model")
        
        print("Synthetic Data Evaluation:")
        for metric, value in metrics.items():
            if metric == 'mape':
                print(f"  {metric.upper()}: {value:.2f}%")
            else:
                print(f"  {metric.upper()}: {value:.3f}")
        
        # Generate synthetic prediction report
        synthetic_predictions = [
            {
                'driver': f'P{i+1}',
                'predicted': pred,
                'actual': actual,
                'confidence_interval': (pred-0.5, pred+0.5),
                'model_name': 'synthetic'
            }
            for i, (pred, actual) in enumerate(zip(predicted_lap_times[:10], true_lap_times[:10]))
        ]
        
        report = evaluator.generate_prediction_report("synthetic_session", synthetic_predictions)
        print(f"\nSynthetic Report:")
        print(f"  Total Predictions: {report.get('total_predictions', 0)}")
        print(f"  Average Accuracy: {report.get('avg_accuracy', 0):.1%}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()