import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_and_clean(filepath):
    df = pd.read_csv(filepath)

    df.drop_duplicates(inplace=True)

    df['person_emp_exp'] = df['person_emp_exp'].fillna(df['person_emp_exp'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    df = df[df['person_age'] < 100]

    return df


def engineer_features(df):
    df['debt_to_income'] = df['loan_amnt'] / (df['person_income'] + 1)
    df['income_to_loan_ratio'] = df['person_income'] / (df['loan_amnt'] + 1)
    df['credit_to_age_ratio'] = df['cb_person_cred_hist_length'] / (df['person_age'] + 1)

    df['label'] = df.apply(assign_label, axis=1)

    return df


def assign_label(row):
    if row['loan_status'] == 0 and row['loan_int_rate'] <= 12 and row['credit_score'] >= 700:
        return 'Approved'
    elif row['loan_status'] == 1 or row['loan_int_rate'] > 18 or row['credit_score'] < 580:
        return 'Denied'
    else:
        return 'Review'


def encode_and_scale(df):
    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent',
                                     'previous_loan_defaults_on_file',
                                     'person_gender', 'person_education'])

    df.drop(columns=['loan_status'], inplace=True)

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    joblib.dump(le, 'models/label_encoder.pkl')

    feature_cols = [c for c in df.columns if c not in ['label', 'label_encoded']]
    X = df[feature_cols]
    y = df['label_encoded']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')

    return X_scaled, y, le


def preprocess(filepath='data/loan_data.csv'):
    os.makedirs('models', exist_ok=True)

    df = load_and_clean(filepath)
    df = engineer_features(df)
    X, y, le = encode_and_scale(df)

    print(f"Dataset shape: {X.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    return X, y