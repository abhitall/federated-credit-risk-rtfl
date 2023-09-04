import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

DATA_FILE_APPLICATION = 'application_record.csv'
DATA_FILE_CREDIT = 'credit_record.csv'

def load_and_preprocess_data():
    try:
        app_df = pd.read_csv(DATA_FILE_APPLICATION)
        cred_df = pd.read_csv(DATA_FILE_CREDIT)
    except FileNotFoundError:
        print(f"Error: Ensure '{DATA_FILE_APPLICATION}' and '{DATA_FILE_CREDIT}' are in the current directory.")
        print("Download from:")
        print("https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction")
        return None, None, None, None

    df = pd.merge(app_df, cred_df, on='ID', how='inner')

    def determine_credit_risk(group):
        if any(s in ['2', '3', '4', '5'] for s in group['STATUS'].astype(str)):
            return 1 # Bad risk
        return 0 # Good risk

    target_df = df.groupby('ID').apply(determine_credit_risk).reset_index(name='TARGET')
    unique_app_df = df.drop_duplicates(subset=['ID'], keep='first')
    df = pd.merge(unique_app_df, target_df, on='ID', how='left')
    
    df = df.drop(['ID', 'MONTHS_BALANCE', 'STATUS', 'FLAG_MOBIL'], axis=1)
    df = df.dropna(subset=['TARGET'])
    
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
        
    df['DAYS_BIRTH'] = np.abs(df['DAYS_BIRTH']) / 365
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: 0 if x > 0 else np.abs(x) / 365)

    X = df.drop('TARGET', axis=1)
    y = df['TARGET'].astype(int)

    categorical_features = X.select_dtypes(include='object').columns
    numerical_features = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    try:
        feature_names_cat = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    except AttributeError:
         feature_names_cat = list(preprocessor.named_transformers_['cat'].get_feature_names_out())

    feature_names = numerical_features.tolist() + feature_names_cat.tolist()
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    # Reset indices to avoid KeyError when indexing with numpy arrays
    if hasattr(X_train_full, 'reset_index'):
        X_train_full = pd.DataFrame(X_train_full).reset_index(drop=True)
    if hasattr(y_train_full, 'reset_index'):
        y_train_full = pd.Series(y_train_full).reset_index(drop=True)
    if hasattr(X_test, 'reset_index'):
        X_test = pd.DataFrame(X_test).reset_index(drop=True)
    if hasattr(y_test, 'reset_index'):
        y_test = pd.Series(y_test).reset_index(drop=True)
    return X_train_full, y_train_full, X_test, y_test, feature_names

def split_data_for_clients(X_train_full, y_train_full, num_clients):
    client_data = []
    if len(y_train_full) == 0:
        for i in range(num_clients):
            client_data.append((np.array([]).reshape(0, X_train_full.shape[1]), np.array([])))
        return client_data

    shuffled_indices = np.random.permutation(len(y_train_full))
    split_indices = np.array_split(shuffled_indices, num_clients)

    for i in range(num_clients):
        client_idx = split_indices[i]
        if len(client_idx) == 0:
            client_data.append((np.array([]).reshape(0, X_train_full.shape[1]), np.array([])))
        else:
            client_data.append((X_train_full[client_idx], y_train_full[client_idx]))
    return client_data
