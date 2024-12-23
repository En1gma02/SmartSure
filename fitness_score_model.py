import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def train_and_save_model():
    print("Loading and preprocessing data...")
    # Load data
    df = pd.read_csv('fitness_claim_dataset.csv')
    df = df.dropna()

    # Process categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.difference(['Name'])
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=[np.number]).columns.difference(['Age'])
    scaled_features = scaler.fit_transform(df[numerical_columns])
    df[numerical_columns] = scaled_features

    # Calculate fitness score
    df['Fitness Score'] = (
        0.1 * df['Blood Pressure (Systolic)'] +
        0.1 * df['Blood Pressure (Diastolic)'] +
        0.15 * df['Heart Beats'] +
        0.15 * df['BMI'] +
        0.1 * df['Cholesterol'] +
        0.2 * df['Steps Taken'] +
        0.1 * df['Active Minutes'] +
        0.1 * df['Sleep Duration'] +
        0.05 * df['Sleep Quality'] +
        0.15 * df['VO2 Max'] +
        0.1 * df['Calories Burned'] +
        0.15 * df['SpO2 Levels'] +
        -0.2 * df['Stress Levels']
    )

    # Normalize fitness score to 0-100 range
    df['Fitness Score'] = (df['Fitness Score'] - df['Fitness Score'].min()) / (df['Fitness Score'].max() - df['Fitness Score'].min()) * 100

    # Save the processed DataFrame
    df.to_csv('models/processed_fitness_claim_dataset.csv', index=False)

    # Prepare features and target
    X = df.drop(['Name', 'Fitness Score'], axis=1)
    y = df['Fitness Score']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    # Train model
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    print("Saving model and preprocessors...")
    # Save model and preprocessors
    joblib.dump(rf_regressor, 'models/fitness_model.joblib')
    joblib.dump(label_encoders, 'models/label_encoders.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save the feature names
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names.joblib')

    print("Model training and saving completed!")

if __name__ == "__main__":
    train_and_save_model()
