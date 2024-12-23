import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def train_and_save_model():
    print("Loading and preprocessing data...")
    # Load data
    df = pd.read_csv('fitness_claim_dataset.csv')
    df = df.dropna()

    # Define column types
    id_columns = ['Name']
    integer_columns = ['Age', 'Sleep Quality', 'Stress Levels', 'Active Minutes']
    float_columns = [
        'Blood Pressure (Systolic)', 'Blood Pressure (Diastolic)', 
        'Heart Beats', 'BMI', 'Cholesterol', 'Steps Taken',
        'Sleep Duration', 'VO2 Max', 'Calories Burned', 'SpO2 Levels'
    ]
    target_column = 'Claim Amount'

    # Verify all expected columns are present
    expected_columns = id_columns + integer_columns + float_columns + [target_column]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    # Convert columns to appropriate types
    for col in integer_columns:
        df[col] = df[col].astype(int)
    for col in float_columns:
        df[col] = df[col].astype(float)

    # Initialize scaler
    scaler = StandardScaler()
    
    # Scale only the float columns
    scaled_features = scaler.fit_transform(df[float_columns])
    scaled_df = pd.DataFrame(scaled_features, columns=float_columns)
    
    # Combine scaled and unscaled features for ML
    X = pd.concat([
        df[integer_columns],  # Keep integers as is
        scaled_df  # Add scaled float features
    ], axis=1)

    # Calculate fitness score using scaled features
    df['Fitness Score'] = (
        0.1 * scaled_df['Blood Pressure (Systolic)'] +
        0.1 * scaled_df['Blood Pressure (Diastolic)'] +
        0.15 * scaled_df['Heart Beats'] +
        0.15 * scaled_df['BMI'] +
        0.1 * scaled_df['Cholesterol'] +
        0.2 * scaled_df['Steps Taken'] +
        0.1 * df['Active Minutes'].astype(float)/df['Active Minutes'].max() +
        0.1 * scaled_df['Sleep Duration'] +
        0.05 * df['Sleep Quality'].astype(float)/df['Sleep Quality'].max() +
        0.15 * scaled_df['VO2 Max'] +
        0.1 * scaled_df['Calories Burned'] +
        0.15 * scaled_df['SpO2 Levels'] +
        -0.2 * df['Stress Levels'].astype(float)/df['Stress Levels'].max()
    )

    # Normalize fitness score to 0-100 range
    df['Fitness Score'] = (df['Fitness Score'] - df['Fitness Score'].min()) / (df['Fitness Score'].max() - df['Fitness Score'].min()) * 100

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save two versions of the processed data:
    
    # 1. Normalized version for ML tasks
    ml_df = pd.concat([
        df[id_columns],  # Original ID columns
        X,  # Scaled features
        df[['Fitness Score', target_column]]  # Calculated and target columns
    ], axis=1)
    ml_df.to_csv('models/ml_processed_dataset.csv', index=False)

    # 2. Original values version for visualization
    viz_df = pd.concat([
        df[id_columns + integer_columns + float_columns],  # Original unscaled columns
        df[['Fitness Score', target_column]]  # Calculated and target columns
    ], axis=1)
    viz_df.to_csv('models/viz_processed_dataset.csv', index=False)

    # Prepare feature names in correct order
    feature_names = integer_columns + float_columns
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(
        X[feature_names], 
        df['Fitness Score'], 
        test_size=0.2, 
        random_state=42
    )
    
    print("Training model...")
    rf_regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_regressor.fit(X_train, y_train)

    print("Saving model and preprocessors...")
    joblib.dump(rf_regressor, 'models/fitness_model_rf.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(feature_names, 'models/feature_names.joblib')
    joblib.dump(float_columns, 'models/float_columns.joblib')
    joblib.dump(integer_columns, 'models/integer_columns.joblib')

    print("Model training and saving completed!")

if __name__ == "__main__":
    train_and_save_model()