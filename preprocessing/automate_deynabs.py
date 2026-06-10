import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os, warnings
warnings.filterwarnings('ignore')

def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    print(f'[LOAD] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom')
    return df

def preprocess(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def save_results(X_train, X_test, y_train, y_test, columns):
    os.makedirs('breast_cancer_preprocessing', exist_ok=True)
    pd.DataFrame(X_train, columns=columns).to_csv('breast_cancer_preprocessing/X_train.csv', index=False)
    pd.DataFrame(X_test, columns=columns).to_csv('breast_cancer_preprocessing/X_test.csv', index=False)
    y_train.to_csv('breast_cancer_preprocessing/y_train.csv', index=False)
    y_test.to_csv('breast_cancer_preprocessing/y_test.csv', index=False)
    print('[SAVE] Data disimpan ke breast_cancer_preprocessing/')

if __name__ == '__main__':
    print('=== AUTOMATE PREPROCESSING — BREAST CANCER DATASET ===')
    df = load_data()
    X_train, X_test, y_train, y_test, columns = preprocess(df)
    save_results(X_train, X_test, y_train, y_test, columns)
    print('=== PREPROCESSING SELESAI! ===')