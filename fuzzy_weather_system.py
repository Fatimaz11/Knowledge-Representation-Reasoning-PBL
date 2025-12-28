"""
Fuzzy Logic Weather Comfort Index Prediction System
Course: Knowledge Representation & Reasoning (KRR)
Dataset: Weather Dataset (Kaggle)

This module implements a Mamdani fuzzy inference system to predict
comfort index based on temperature and humidity.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ===== 1. DEFINE FUZZY VARIABLES =====

# Input Variables
temperature = ctrl.Antecedent(np.arange(-10, 51, 0.1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')

# Output Variable
comfort_index = ctrl.Consequent(np.arange(0, 101, 1), 'comfort_index')

# ===== 2. DEFINE MEMBERSHIP FUNCTIONS =====

# Temperature: Cold, Mild, Warm, Hot
# Based on typical comfort ranges in Celsius
temperature['cold'] = fuzz.trimf(temperature.universe, [-10, -10, 15])
temperature['mild'] = fuzz.trimf(temperature.universe, [10, 20, 27])
temperature['warm'] = fuzz.trimf(temperature.universe, [22, 28, 35])
temperature['hot'] = fuzz.trimf(temperature.universe, [30, 50, 50])

# Humidity: Low, Moderate, High, Very High
humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 35])
humidity['moderate'] = fuzz.trimf(humidity.universe, [25, 45, 65])
humidity['high'] = fuzz.trimf(humidity.universe, [55, 70, 85])
humidity['very_high'] = fuzz.trimf(humidity.universe, [75, 100, 100])

# Comfort Index: Uncomfortable, Neutral, Comfortable
# Scale: 0-100 (0 = very uncomfortable, 100 = very comfortable)
comfort_index['uncomfortable'] = fuzz.trimf(comfort_index.universe, [0, 0, 40])
comfort_index['neutral'] = fuzz.trimf(comfort_index.universe, [30, 50, 70])
comfort_index['comfortable'] = fuzz.trimf(comfort_index.universe, [60, 100, 100])

# ===== 3. DEFINE FUZZY RULES =====
# Total: 16 rules covering all major combinations

# Rule Group 1: Cold Temperature (4 rules)
rule1 = ctrl.Rule(temperature['cold'] & humidity['low'], comfort_index['uncomfortable'])
rule2 = ctrl.Rule(temperature['cold'] & humidity['moderate'], comfort_index['uncomfortable'])
rule3 = ctrl.Rule(temperature['cold'] & humidity['high'], comfort_index['uncomfortable'])
rule4 = ctrl.Rule(temperature['cold'] & humidity['very_high'], comfort_index['uncomfortable'])

# Rule Group 2: Mild Temperature (4 rules) - Most comfortable range
rule5 = ctrl.Rule(temperature['mild'] & humidity['low'], comfort_index['comfortable'])
rule6 = ctrl.Rule(temperature['mild'] & humidity['moderate'], comfort_index['comfortable'])
rule7 = ctrl.Rule(temperature['mild'] & humidity['high'], comfort_index['neutral'])
rule8 = ctrl.Rule(temperature['mild'] & humidity['very_high'], comfort_index['uncomfortable'])

# Rule Group 3: Warm Temperature (4 rules)
rule9 = ctrl.Rule(temperature['warm'] & humidity['low'], comfort_index['comfortable'])
rule10 = ctrl.Rule(temperature['warm'] & humidity['moderate'], comfort_index['neutral'])
rule11 = ctrl.Rule(temperature['warm'] & humidity['high'], comfort_index['uncomfortable'])
rule12 = ctrl.Rule(temperature['warm'] & humidity['very_high'], comfort_index['uncomfortable'])

# Rule Group 4: Hot Temperature (4 rules)
rule13 = ctrl.Rule(temperature['hot'] & humidity['low'], comfort_index['neutral'])
rule14 = ctrl.Rule(temperature['hot'] & humidity['moderate'], comfort_index['uncomfortable'])
rule15 = ctrl.Rule(temperature['hot'] & humidity['high'], comfort_index['uncomfortable'])
rule16 = ctrl.Rule(temperature['hot'] & humidity['very_high'], comfort_index['uncomfortable'])

# ===== 4. CREATE CONTROL SYSTEM =====

comfort_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8,
    rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16
])

comfort_sim = ctrl.ControlSystemSimulation(comfort_ctrl)

# ===== 5. PREDICTION FUNCTION =====

def predict_comfort(temp, humid):
    """
    Predict comfort index using fuzzy logic
    
    Parameters:
    -----------
    temp : float
        Temperature in Celsius (-10 to 50)
    humid : float
        Humidity percentage (0 to 100)
    
    Returns:
    --------
    comfort_score : float
        Comfort index (0-100)
    comfort_level : str
        Textual comfort level (Uncomfortable/Neutral/Comfortable)
    """
    # Input bounds checking
    temp = np.clip(temp, -10, 50)
    humid = np.clip(humid, 0, 100)
    
    # Set inputs
    comfort_sim.input['temperature'] = temp
    comfort_sim.input['humidity'] = humid
    
    # Compute output
    comfort_sim.compute()
    
    comfort_score = comfort_sim.output['comfort_index']
    
    # Determine comfort level
    if comfort_score < 40:
        comfort_level = "Uncomfortable"
    elif comfort_score < 70:
        comfort_level = "Neutral"
    else:
        comfort_level = "Comfortable"
    
    return comfort_score, comfort_level


# ===== 6. BATCH PROCESSING FUNCTION =====

def process_weather_dataset(csv_file, temp_col='Temperature (C)', humid_col='Humidity'):
    """
    Process entire weather dataset using fuzzy logic
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file containing weather data
    temp_col : str
        Name of temperature column
    humid_col : str
        Name of humidity column
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with added comfort predictions
    """
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded dataset: {len(df)} records")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Handle different possible column names
        if temp_col not in df.columns:
            # Try to find temperature column
            temp_candidates = [col for col in df.columns if 'temp' in col.lower()]
            if temp_candidates:
                temp_col = temp_candidates[0]
                print(f"  Using temperature column: {temp_col}")
        
        if humid_col not in df.columns:
            # Try to find humidity column
            humid_candidates = [col for col in df.columns if 'humid' in col.lower()]
            if humid_candidates:
                humid_col = humid_candidates[0]
                print(f"  Using humidity column: {humid_col}")
        
        # Initialize prediction columns
        comfort_scores = []
        comfort_levels = []
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                temp = float(row[temp_col])
                humid = float(row[humid_col])
                
                score, level = predict_comfort(temp, humid)
                comfort_scores.append(score)
                comfort_levels.append(level)
                
            except Exception as e:
                print(f"  Warning: Error at row {idx}: {e}")
                comfort_scores.append(None)
                comfort_levels.append(None)
        
        # Add predictions to dataframe
        df['comfort_score'] = comfort_scores
        df['comfort_level'] = comfort_levels
        
        # Save results
        output_file = 'weather_predictions.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Predictions saved to '{output_file}'")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("COMFORT LEVEL DISTRIBUTION")
        print("="*60)
        print(df['comfort_level'].value_counts())
        print("\n" + "="*60)
        print("COMFORT SCORE STATISTICS")
        print("="*60)
        print(f"Mean:   {df['comfort_score'].mean():.2f}")
        print(f"Median: {df['comfort_score'].median():.2f}")
        print(f"Std:    {df['comfort_score'].std():.2f}")
        print(f"Min:    {df['comfort_score'].min():.2f}")
        print(f"Max:    {df['comfort_score'].max():.2f}")
        
        return df
        
    except FileNotFoundError:
        print(f"✗ Error: File '{csv_file}' not found!")
        print("  Please ensure the weather dataset is in the same directory.")
        return None
    except Exception as e:
        print(f"✗ Error processing dataset: {e}")
        return None


# ===== 7. VISUALIZATION FUNCTIONS =====

def visualize_membership_functions():
    """
    Generate plots of all membership functions
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temperature MFs
    temperature.view(ax=axes[0, 0])
    axes[0, 0].set_title('Temperature Membership Functions', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Temperature (°C)', fontsize=12)
    axes[0, 0].set_ylabel('Membership Degree', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(['Cold', 'Mild', 'Warm', 'Hot'], loc='upper right')
    
    # Humidity MFs
    humidity.view(ax=axes[0, 1])
    axes[0, 1].set_title('Humidity Membership Functions', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Humidity (%)', fontsize=12)
    axes[0, 1].set_ylabel('Membership Degree', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(['Low', 'Moderate', 'High', 'Very High'], loc='upper right')
    
    # Comfort Index MFs
    comfort_index.view(ax=axes[1, 0])
    axes[1, 0].set_title('Comfort Index Membership Functions', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Comfort Index (0-100)', fontsize=12)
    axes[1, 0].set_ylabel('Membership Degree', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(['Uncomfortable', 'Neutral', 'Comfortable'], loc='upper right')
    
    # Hide the fourth subplot
    axes[1, 1].axis('off')
    
    # Add system info in the fourth subplot
    info_text = """
    FUZZY WEATHER COMFORT SYSTEM
    ═══════════════════════════════════
    
    Inputs:
    • Temperature: -10°C to 50°C
    • Humidity: 0% to 100%
    
    Output:
    • Comfort Index: 0 to 100
    
    Rules: 16 fuzzy IF-THEN rules
    Method: Mamdani Inference
    Defuzzification: Centroid
    
    Comfort Levels:
    • Uncomfortable: 0-40
    • Neutral: 40-70
    • Comfortable: 70-100
    """
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('membership_functions.png', dpi=300, bbox_inches='tight')
    print("✓ Membership functions plot saved as 'membership_functions.png'")
    plt.close()


def create_comfort_heatmap():
    """
    Create a 2D heatmap showing comfort index across temperature and humidity ranges
    """
    # Create meshgrid
    temp_range = np.linspace(-10, 50, 60)
    humid_range = np.linspace(0, 100, 50)
    
    comfort_grid = np.zeros((len(humid_range), len(temp_range)))
    
    print("\nGenerating comfort heatmap...")
    for i, h in enumerate(humid_range):
        for j, t in enumerate(temp_range):
            try:
                score, _ = predict_comfort(t, h)
                comfort_grid[i, j] = score
            except:
                comfort_grid[i, j] = 50  # Default neutral
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(comfort_grid, extent=[-10, 50, 0, 100], origin='lower', 
               cmap='RdYlGn', aspect='auto')
    plt.colorbar(label='Comfort Index (0-100)')
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Humidity (%)', fontsize=12)
    plt.title('Weather Comfort Index Heatmap', fontsize=14, fontweight='bold')
    
    # Add contour lines
    plt.contour(temp_range, humid_range, comfort_grid, 
                levels=[40, 70], colors='black', linewidths=2, linestyles='dashed')
    
    # Add text annotations
    plt.text(20, 50, 'Optimal\nComfort Zone', fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comfort_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Comfort heatmap saved as 'comfort_heatmap.png'")
    plt.close()


# ===== 8. DEMONSTRATION AND TESTING =====

def run_demo():
    """
    Run demonstration with various test cases
    """
    print("\n" + "="*70)
    print("FUZZY WEATHER COMFORT INDEX PREDICTION SYSTEM")
    print("="*70)
    
    # Test cases covering different scenarios
    test_cases = [
        # (Temperature, Humidity, Description)
        (5, 30, "Cold winter day"),
        (20, 50, "Ideal spring day"),
        (25, 40, "Pleasant summer day"),
        (35, 80, "Hot and humid summer"),
        (10, 70, "Cool and humid"),
        (28, 30, "Warm and dry"),
        (40, 90, "Extremely hot and humid"),
        (-5, 60, "Cold winter with moderate humidity"),
        (22, 65, "Moderate conditions"),
        (15, 85, "Cool with high humidity"),
    ]
    
    print("\nTest Cases:")
    print("-" * 70)
    print(f"{'#':<4} {'Temp(°C)':<10} {'Humidity(%)':<14} {'Comfort':<10} {'Level':<15} {'Description':<20}")
    print("-" * 70)
    
    for i, (temp, humid, desc) in enumerate(test_cases, 1):
        score, level = predict_comfort(temp, humid)
        print(f"{i:<4} {temp:<10.1f} {humid:<14.1f} {score:<10.2f} {level:<15} {desc:<20}")
    
    print("="*70)


# ===== 9. MAIN EXECUTION =====

if __name__ == "__main__":
    print("\n" + "="*70)
    print("INITIALIZING FUZZY WEATHER COMFORT SYSTEM")
    print("="*70)
    
    # Run demonstration
    run_demo()
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    visualize_membership_functions()
    create_comfort_heatmap()
    
    # Process dataset if available
    print("\n" + "="*70)
    print("PROCESSING WEATHER DATASET")
    print("="*70)
    print("\nLooking for 'weather_data.csv' in current directory...")
    
    # Try to process the dataset
    df = process_weather_dataset('weather_data.csv')
    
    if df is None:
        print("\n" + "="*70)
        print("DATASET PROCESSING INSTRUCTIONS")
        print("="*70)
        print("1. Download the Weather Dataset from Kaggle:")
        print("   https://www.kaggle.com/datasets/muthuj7/weather-dataset")
        print("\n2. Save the CSV file as 'weather_data.csv' in this directory")
        print("\n3. Run this script again to process the dataset")
        print("\n4. Alternatively, use the function programmatically:")
        print("   df = process_weather_dataset('your_file.csv')")
    
    print("\n" + "="*70)
    print("SYSTEM READY")
    print("="*70)
    print("\nYou can now use predict_comfort(temp, humidity) for predictions!")
    print("\nExample:")
    print("  score, level = predict_comfort(25, 60)")
    print(f"  Result: Comfort Score = {predict_comfort(25, 60)[0]:.2f}, Level = {predict_comfort(25, 60)[1]}")
    print("\n" + "="*70)