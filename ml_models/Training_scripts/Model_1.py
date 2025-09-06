import pandas as pd
import numpy as np
import lightgbm as lgb # type: ignore
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_runoff_model():
    """
    Loads the synthetic dataset, trains a LightGBM model to predict the
    runoff coefficient, evaluates it, and saves the trained model to a PKL file.
    """
    print("--- Training Model 1: Runoff Coefficient Predictor ---")
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv("Datasets/Runoff_coeff_dataset.csv")
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: 'Runoff_coeff_dataset.csv' not found.")
        print("Please run the 'Synthetic_data.py' script first.")
        return

    # --- 2. Preprocessing ---
    # Define features (X) and target (y)
    X = df[['roof_type', 'roof_age', 'region']]
    y = df['runoff_coefficient']

    # Convert categorical features into a numerical format
    # Using one-hot encoding for better model performance
    X = pd.get_dummies(X, columns=['roof_type', 'region'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=50
    )
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # --- 3. Model Training ---
    print("Training the LightGBM model...")
    # Initialize the LightGBM Regressor
    model = lgb.LGBMRegressor(
        random_state=50,
        n_estimators=1000,        # Increased from 100 to build more trees
        learning_rate=0.01,      # Reduced from 0.1 for more careful learning
        num_leaves=31,           # Controls tree complexity
        max_depth=-1,            # No limit on tree depth
        n_jobs=-1)
    
    # Train the model
    model.fit(X_train, y_train)
    print("Training complete.")

    # --- 4. Model Evaluation ---
    y_pred = model.predict(X_test)

    # FIX: Calculate RMSE manually for compatibility with older scikit-learn versions
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print("------------------------")

    # --- 5. Save the Model ---
    model_filename = "runoff_model.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"\nModel successfully saved to '{model_filename}'")


if __name__ == "__main__":
    train_runoff_model()

