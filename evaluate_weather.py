"""
Evaluation Module for Weather Comfort Fuzzy System
Includes metrics calculation, baseline comparison, and sensitivity analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

class WeatherComfortEvaluator:
    """
    Comprehensive evaluation for fuzzy weather comfort system
    """
    
    def __init__(self, predict_function):
        """
        Initialize evaluator
        
        Parameters:
        -----------
        predict_function : callable
            Function that takes (temperature, humidity) and returns (score, level)
        """
        self.predict = predict_function
    
    def evaluate_regression(self, df, temp_col='Temperature (C)', humid_col='Humidity', 
                           true_col='actual_comfort'):
        """
        Evaluate system as regression problem (predicting comfort score)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with temperature, humidity, and actual comfort scores
        
        Returns:
        --------
        dict : Evaluation metrics
        """
        predictions = []
        true_values = []
        
        for idx, row in df.iterrows():
            try:
                temp = row[temp_col]
                humid = row[humid_col]
                true_val = row[true_col]
                
                pred_score, _ = self.predict(temp, humid)
                
                predictions.append(pred_score)
                true_values.append(true_val)
                
            except Exception as e:
                print(f"Error at row {idx}: {e}")
                continue
        
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        # Calculate metrics
        mae = mean_absolute_error(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_values, predictions)
        
        # Calculate accuracy within thresholds
        within_10 = np.sum(np.abs(predictions - true_values) <= 10) / len(predictions) * 100
        within_5 = np.sum(np.abs(predictions - true_values) <= 5) / len(predictions) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'accuracy_within_10': within_10,
            'accuracy_within_5': within_5,
            'predictions': predictions,
            'true_values': true_values,
            'n_samples': len(predictions)
        }
        
        return metrics
    
    def evaluate_classification(self, df, temp_col='Temperature (C)', 
                                humid_col='Humidity', true_col='actual_level'):
        """
        Evaluate system as classification problem (predicting comfort level)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with temperature, humidity, and actual comfort levels
        
        Returns:
        --------
        dict : Classification metrics
        """
        predictions = []
        true_labels = []
        
        for idx, row in df.iterrows():
            try:
                temp = row[temp_col]
                humid = row[humid_col]
                true_label = row[true_col]
                
                _, pred_level = self.predict(temp, humid)
                
                predictions.append(pred_level)
                true_labels.append(true_label)
                
            except Exception as e:
                print(f"Error at row {idx}: {e}")
                continue
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Classification report
        report = classification_report(true_labels, predictions, 
                                      output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, 
                             labels=['Uncomfortable', 'Neutral', 'Comfortable'])
        
        metrics = {
            'accuracy': accuracy * 100,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'true_labels': true_labels,
            'n_samples': len(predictions)
        }
        
        return metrics
    
    def baseline_comparison(self, df, temp_col='Temperature (C)', 
                           humid_col='Humidity', true_col='actual_comfort'):
        """
        Compare fuzzy system with baseline methods
        """
        # Fuzzy predictions
        fuzzy_preds = []
        true_vals = []
        
        for idx, row in df.iterrows():
            try:
                temp = row[temp_col]
                humid = row[humid_col]
                true_val = row[true_col]
                
                pred_score, _ = self.predict(temp, humid)
                fuzzy_preds.append(pred_score)
                true_vals.append(true_val)
            except:
                continue
        
        fuzzy_preds = np.array(fuzzy_preds)
        true_vals = np.array(true_vals)
        
        # Baseline 1: Simple formula (inverse relationship with extremes)
        baseline1_preds = []
        for idx, row in df.iterrows():
            try:
                temp = row[temp_col]
                humid = row[humid_col]
                
                # Comfort decreases with distance from ideal temp (20-25°C)
                temp_comfort = 100 - abs(temp - 22.5) * 3
                temp_comfort = np.clip(temp_comfort, 0, 100)
                
                # Comfort optimal at 40-60% humidity
                humid_comfort = 100 - abs(humid - 50) * 1.5
                humid_comfort = np.clip(humid_comfort, 0, 100)
                
                # Average
                baseline1 = (temp_comfort + humid_comfort) / 2
                baseline1_preds.append(baseline1)
            except:
                continue
        
        baseline1_preds = np.array(baseline1_preds)
        
        # Baseline 2: Linear weighted combination
        baseline2_preds = []
        for idx, row in df.iterrows():
            try:
                temp = row[temp_col]
                humid = row[humid_col]
                
                # Normalize to 0-100 scale
                temp_norm = np.clip((temp + 10) / 60 * 100, 0, 100)
                humid_norm = humid
                
                baseline2 = 0.6 * (100 - abs(temp_norm - 50)) + 0.4 * (100 - abs(humid_norm - 50))
                baseline2 = np.clip(baseline2, 0, 100)
                baseline2_preds.append(baseline2)
            except:
                continue
        
        baseline2_preds = np.array(baseline2_preds)
        
        # Calculate metrics
        fuzzy_mae = mean_absolute_error(true_vals, fuzzy_preds)
        baseline1_mae = mean_absolute_error(true_vals, baseline1_preds)
        baseline2_mae = mean_absolute_error(true_vals, baseline2_preds)
        
        fuzzy_rmse = np.sqrt(mean_squared_error(true_vals, fuzzy_preds))
        baseline1_rmse = np.sqrt(mean_squared_error(true_vals, baseline1_preds))
        baseline2_rmse = np.sqrt(mean_squared_error(true_vals, baseline2_preds))
        
        improvement1 = ((baseline1_mae - fuzzy_mae) / baseline1_mae) * 100
        improvement2 = ((baseline2_mae - fuzzy_mae) / baseline2_mae) * 100
        
        comparison = {
            'fuzzy_mae': fuzzy_mae,
            'baseline1_mae': baseline1_mae,
            'baseline2_mae': baseline2_mae,
            'fuzzy_rmse': fuzzy_rmse,
            'baseline1_rmse': baseline1_rmse,
            'baseline2_rmse': baseline2_rmse,
            'improvement_vs_baseline1': improvement1,
            'improvement_vs_baseline2': improvement2
        }
        
        return comparison
    
    def sensitivity_analysis(self, base_temp=22, base_humid=50):
        """
        Analyze sensitivity to temperature and humidity changes
        """
        results = {}
        
        # Temperature sensitivity (varying temp, fixed humidity)
        temp_range = np.linspace(-10, 50, 100)
        temp_output = [self.predict(t, base_humid)[0] for t in temp_range]
        results['temperature'] = {'input': temp_range, 'output': temp_output}
        
        # Humidity sensitivity (varying humidity, fixed temp)
        humid_range = np.linspace(0, 100, 100)
        humid_output = [self.predict(base_temp, h)[0] for h in humid_range]
        results['humidity'] = {'input': humid_range, 'output': humid_output}
        
        return results
    
    def plot_evaluation_results(self, metrics, save_path='evaluation_results.png'):
        """
        Create comprehensive visualization of evaluation results
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Predicted vs Actual scatter plot
        axes[0, 0].scatter(metrics['true_values'], metrics['predictions'], 
                          alpha=0.6, edgecolors='k', s=50, c='steelblue')
        axes[0, 0].plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Comfort Index', fontsize=12)
        axes[0, 0].set_ylabel('Predicted Comfort Index', fontsize=12)
        axes[0, 0].set_title('Predicted vs Actual Comfort', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim([0, 100])
        axes[0, 0].set_ylim([0, 100])
        
        # 2. Error distribution
        errors = metrics['predictions'] - metrics['true_values']
        axes[0, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
        axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0, 1].set_xlabel('Prediction Error', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Metrics summary
        axes[1, 0].axis('off')
        metrics_text = f"""
        EVALUATION METRICS
        ═══════════════════════════════════
        
        Regression Metrics:
        ─────────────────────────────────
        MAE:               {metrics['mae']:.2f}
        RMSE:              {metrics['rmse']:.2f}
        R² Score:          {metrics['r2_score']:.3f}
        
        Accuracy Metrics:
        ─────────────────────────────────
        Within ±10 points: {metrics['accuracy_within_10']:.1f}%
        Within ±5 points:  {metrics['accuracy_within_5']:.1f}%
        
        Dataset Info:
        ─────────────────────────────────
        Total Samples:     {metrics['n_samples']}
        """
        axes[1, 0].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # 4. Residual plot
        axes[1, 1].scatter(metrics['predictions'], errors, alpha=0.6, 
                          edgecolors='k', s=50, c='purple')
        axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Predicted Comfort Index', fontsize=12)
        axes[1, 1].set_ylabel('Residuals', fontsize=12)
        axes[1, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Evaluation plot saved as '{save_path}'")
        plt.close()
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix for classification evaluation
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Uncomfortable', 'Neutral', 'Comfortable'],
                   yticklabels=['Uncomfortable', 'Neutral', 'Comfortable'],
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix - Comfort Level Classification', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved as '{save_path}'")
        plt.close()
    
    def plot_sensitivity(self, sensitivity_results, save_path='sensitivity_analysis.png'):
        """
        Plot sensitivity analysis results
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Temperature sensitivity
        temp_data = sensitivity_results['temperature']
        axes[0].plot(temp_data['input'], temp_data['output'], 
                    linewidth=3, color='#e74c3c', marker='o', markersize=3)
        axes[0].axhline(y=40, color='orange', linestyle='--', alpha=0.7, label='Uncomfortable Threshold')
        axes[0].axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Comfortable Threshold')
        axes[0].fill_between(temp_data['input'], 40, 70, alpha=0.2, color='yellow')
        axes[0].set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Comfort Index', fontsize=12)
        axes[0].set_title('Temperature Sensitivity Analysis', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim([0, 100])
        
        # Humidity sensitivity
        humid_data = sensitivity_results['humidity']
        axes[1].plot(humid_data['input'], humid_data['output'], 
                    linewidth=3, color='#3498db', marker='o', markersize=3)
        axes[1].axhline(y=40, color='orange', linestyle='--', alpha=0.7, label='Uncomfortable Threshold')
        axes[1].axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Comfortable Threshold')
        axes[1].fill_between(humid_data['input'], 40, 70, alpha=0.2, color='yellow')
        axes[1].set_xlabel('Humidity (%)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Comfort Index', fontsize=12)
        axes[1].set_title('Humidity Sensitivity Analysis', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_ylim([0, 100])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sensitivity analysis plot saved as '{save_path}'")
        plt.close()
    
    def generate_full_report(self, df, temp_col='Temperature (C)', 
                            humid_col='Humidity', output_dir='results'):
        """
        Generate comprehensive evaluation report with all visualizations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*70)
        
        # Run sensitivity analysis
        print("\n1. Running sensitivity analysis...")
        sensitivity = self.sensitivity_analysis()
        self.plot_sensitivity(sensitivity, 
                             save_path=os.path.join(output_dir, 'sensitivity_analysis.png'))
        
        print("\n✓ Evaluation report generated successfully!")
        print(f"✓ All results saved in '{output_dir}/' directory")


# ===== DEMONSTRATION SCRIPT =====

def run_evaluation_demo():
    """
    Demonstration of evaluation capabilities
    """
    print("\n" + "="*70)
    print("WEATHER COMFORT EVALUATOR - DEMONSTRATION")
    print("="*70)
    
    # Import the prediction function
    try:
        from fuzzy_weather_system import predict_comfort
    except ImportError:
        print("✗ Error: Could not import fuzzy_weather_system.py")
        print("  Make sure fuzzy_weather_system.py is in the same directory")
        return
    
    # Create evaluator
    evaluator = WeatherComfortEvaluator(predict_comfort)
    
    # Run sensitivity analysis
    print("\n1. Running Sensitivity Analysis...")
    sensitivity = evaluator.sensitivity_analysis(base_temp=22, base_humid=50)
    evaluator.plot_sensitivity(sensitivity)
    
    print("\n2. Creating test scenarios...")
    # Create synthetic test data for demonstration
    test_data = {
        'Temperature (C)': [5, 10, 15, 20, 25, 30, 35, 40],
        'Humidity': [30, 40, 50, 60, 70, 80, 90, 95]
    }
    
    print("\n3. Making predictions...")
    for temp, humid in zip(test_data['Temperature (C)'], test_data['Humidity']):
        score, level = predict_comfort(temp, humid)
        print(f"   Temp: {temp:>3}°C, Humidity: {humid:>3}% → Comfort: {score:>5.2f} ({level})")
    
    print("\n" + "="*70)
    print("✓ DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  • sensitivity_analysis.png")
    print("\nTo use with real data:")
    print("  from evaluate_weather import WeatherComfortEvaluator")
    print("  evaluator = WeatherComfortEvaluator(predict_comfort)")
    print("  metrics = evaluator.evaluate_regression(your_dataframe)")
    print("="*70)


# Example usage and testing
if __name__ == "__main__":
    run_evaluation_demo()