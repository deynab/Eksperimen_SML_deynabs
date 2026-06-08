import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape}")
    return df

def preprocess_data(df):
    # Pisah fitur dan target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def save_results(X_train, X_test, y_train, y_test, columns):
    os.makedirs('preprocessing/wine_preprocessing', exist_ok=True)
    
    pd.DataFrame(X_train, columns=columns).to_csv('preprocessing/wine_preprocessing/X_train.csv', index=False)
    pd.DataFrame(X_test, columns=columns).to_csv('preprocessing/wine_preprocessing/X_test.csv', index=False)
    y_train.to_csv('preprocessing/wine_preprocessing/y_train.csv', index=False)
    y_test.to_csv('preprocessing/wine_preprocessing/y_test.csv', index=False)
    
    print("Data preprocessing tersimpan!")

if __name__ == "__main__":
    df = load_data('preprocessing/wine_dataset.csv')
    X_train, X_test, y_train, y_test, columns = preprocess_data(df)
    save_results(X_train, X_test, y_train, y_test, columns)
    print("Automate preprocessing selesai!")