# training/train_models.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os
import json
from datetime import datetime

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def load_real_data():
    """
    Replace this with actual data loading from:
    - Database
    - CSV files
    - APIs
    - Research papers
    """
    try:
        # Try to load from database or file
        # For now, use sample data
        return create_sample_data()
    except:
        print("No real data found. Using sample data for initial training.")
        return create_sample_data()

def create_sample_data():
    """Create realistic sample data based on research findings"""
    # Runoff coefficient data based on research
    roof_coefficients = {
        'Concrete': (0.75, 0.85),
        'Tiled': (0.65, 0.75),
        'Metal': (0.85, 0.95),
        'Asbestos': (0.75, 0.85),
        'Thatched': (0.55, 0.65),
        'Plastic': (0.80, 0.90)
    }
    
    runoff_data = pd.DataFrame()
    for roof_type, (min_coeff, max_coeff) in roof_coefficients.items():
        for _ in range(50):
            age = np.random.randint(0, 30)
            is_urban = np.random.choice([0, 1])
            # Coefficient decreases with age and urban pollution
            base_coeff = np.random.uniform(min_coeff, max_coeff)
            age_impact = age * 0.002  # 0.2% decrease per year
            urban_impact = 0.03 if is_urban else 0.0
            final_coeff = max(min_coeff * 0.7, base_coeff - age_impact - urban_impact)
            
            runoff_data = pd.concat([runoff_data, pd.DataFrame({
                'roof_type': [roof_type],
                'roof_age': [age],
                'is_urban': [is_urban],
                'runoff_coeff': [final_coeff]
            })])
    
    # Structure recommendation data based on CGWB guidelines
    structure_rules = [
        # (roof_area, open_space, soil_type, water_depth, structure)
        (100, 50, 'Sandy', 10, 'Recharge_Pit'),
        (80, 30, 'Sandy Loam', 15, 'Recharge_Trench'),
        (120, 20, 'Clay', 25, 'Recharge_Shaft'),
        (60, 10, 'Clay Loam', 8, 'Storage_Tank'),
    ]
    
    structure_data = pd.DataFrame()
    for roof_area, open_space, soil_type, water_depth, structure in structure_rules:
        for _ in range(25):
            # Add some variation
            area_var = roof_area * np.random.uniform(0.8, 1.2)
            space_var = open_space * np.random.uniform(0.8, 1.2)
            depth_var = water_depth * np.random.uniform(0.9, 1.1)
            
            structure_data = pd.concat([structure_data, pd.DataFrame({
                'roof_area': [area_var],
                'open_space': [space_var],
                'soil_type': [soil_type],
                'aquifer_type': ['Alluvial'],  # Most common in India
                'water_depth': [depth_var],
                'structure': [structure]
            })])
    
    return runoff_data, structure_data

def train_models():
    print("Starting model training...")
    print(f"Timestamp: {datetime.now()}")
    
    # Load or create data
    runoff_data, structure_data = load_real_data()
    
    print(f"Runoff data samples: {len(runoff_data)}")
    print(f"Structure data samples: {len(structure_data)}")
    
    # Train runoff coefficient model
    print("Training runoff coefficient model...")
    le_roof = LabelEncoder()
    runoff_data['roof_type_encoded'] = le_roof.fit_transform(runoff_data['roof_type'])
    
    X_runoff = runoff_data[['roof_type_encoded', 'roof_age', 'is_urban']]
    y_runoff = runoff_data['runoff_coeff']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_runoff, y_runoff, test_size=0.2, random_state=42
    )
    
    runoff_model = RandomForestRegressor(n_estimators=100, random_state=42)
    runoff_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = runoff_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Runoff model MSE: {mse:.4f}")
    
    # Train structure recommendation model
    print("Training structure recommendation model...")
    le_soil = LabelEncoder()
    le_aquifer = LabelEncoder()
    le_structure = LabelEncoder()
    
    structure_data['soil_encoded'] = le_soil.fit_transform(structure_data['soil_type'])
    structure_data['aquifer_encoded'] = le_aquifer.fit_transform(structure_data['aquifer_type'])
    structure_data['structure_encoded'] = le_structure.fit_transform(structure_data['structure'])
    
    X_structure = structure_data[['roof_area', 'open_space', 'soil_encoded', 'aquifer_encoded', 'water_depth']]
    y_structure = structure_data['structure_encoded']
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_structure, y_structure, test_size=0.2, random_state=42
    )
    
    structure_model = RandomForestClassifier(n_estimators=100, random_state=42)
    structure_model.fit(X_train_s, y_train_s)
    
    # Evaluate
    y_pred_s = structure_model.predict(X_test_s)
    accuracy = accuracy_score(y_test_s, y_pred_s)
    print(f"Structure model accuracy: {accuracy:.4f}")
    
    # Save models and encoders
    print("Saving models...")
    joblib.dump(runoff_model, 'models/runoff_model.pkl')
    joblib.dump(structure_model, 'models/structure_model.pkl')
    joblib.dump(le_roof, 'models/roof_type_encoder.pkl')
    joblib.dump(le_soil, 'models/soil_type_encoder.pkl')
    joblib.dump(le_aquifer, 'models/aquifer_type_encoder.pkl')
    joblib.dump(le_structure, 'models/structure_encoder.pkl')
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'runoff_samples': len(runoff_data),
        'structure_samples': len(structure_data),
        'runoff_mse': float(mse),
        'structure_accuracy': float(accuracy),
        'feature_columns': {
            'runoff': list(X_runoff.columns),
            'structure': list(X_structure.columns)
        }
    }
    
    with open('models/training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Models trained and saved successfully!")
    print(f"Runoff model performance: MSE = {mse:.4f}")
    print(f"Structure model performance: Accuracy = {accuracy:.4f}")

if __name__ == "__main__":
    train_models()