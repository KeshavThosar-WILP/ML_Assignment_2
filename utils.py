from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Constants saved from the training StandardScaler
# Features order: ['tenure', 'MonthlyCharges', 'TotalCharges']
SCALER_MEAN = np.array([32.37114866, 64.76169246, 2279.73430356])
SCALER_SCALE = np.array([24.55773827, 30.08791192, 2266.63363378])

# Categorical features in order
CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
    'PaperlessBilling', 'PaymentMethod'
]

# Exact columns expected by the trained models after one-hot encoding
EXPECTED_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

AVAILABLE_MODELS = ['LogisticRegression', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'MultinomialNB', 'SVC']

def evaluate_dataset(test_data, model_name):
    data = test_data.copy()

    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
    
    # Separate features and target
    X_test_raw = data.drop(columns=['Churn'])
    y_test_raw = data['Churn'].map({'Yes': 1, 'No': 0})
    
    # Numerical scaling using saved constants
    X_num = X_test_raw[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
    X_num_scaled = (X_num - SCALER_MEAN) / SCALER_SCALE
    
    # One-hot encoding using pd.get_dummies to match training schema
    X_cat_encoded = pd.get_dummies(X_test_raw[CATEGORICAL_COLS], drop_first=True, dtype=float)
    
    # Combine processed features
    X_combined = pd.concat([X_num_scaled, X_cat_encoded], axis=1)
    
    # Align columns to match the trained dataset feature space
    for col in EXPECTED_FEATURES:
        if col not in X_combined.columns:
            X_combined[col] = 0.0
    X_final = X_combined[EXPECTED_FEATURES]

    parent_dir = Path(__file__).resolve().parent
    
    model_path = f'{parent_dir}/model/{model_name}.joblib'
    model = joblib.load(model_path)

    if model_name == 'MultinomialNB':
        nb_scaler = MinMaxScaler()
        X_final = nb_scaler.fit_transform(X_final)
    
    y_pred = model.predict(X_final)
    y_prob = model.predict_proba(X_final)[:, 1]
    auc = roc_auc_score(y_test_raw, y_prob)
        
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test_raw, y_pred),
        'AUC': auc,
        'Precision': precision_score(y_test_raw, y_pred),
        'Recall': recall_score(y_test_raw, y_pred),
        'F1 Score': f1_score(y_test_raw, y_pred),
        'MCC': matthews_corrcoef(y_test_raw, y_pred)
    }
    
    cm = confusion_matrix(y_test_raw, y_pred)
    
    return metrics, cm
