import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
import pickle


def load_and_prepare_koi_data():
    print("Usando o serviço TAP da NASA para carregar dados do Kepler (KOI)...")
    url_koi = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select%20%2A%20from%20cumulative&format=csv"
    
    koi = pd.read_csv(url_koi, comment='#')
    print(f"KOI shape inicial: {koi.shape}")

    # === 2. Prepare KOI data ===
    # Target column
    target_col = "koi_pdisposition"

    # Binary target encoding: 1 = Exoplanet, 0 = False Positive
    koi["label"] = koi[target_col].apply(lambda x: 1 if x in ["CANDIDATE"] else 0)

    # Select relevant numeric features
    features = [
        "koi_period", "koi_duration", "koi_depth", "koi_impact",
        "koi_steff", "koi_slogg", "koi_srad",
        "koi_prad", "koi_insol"
    ]

    # Keep only available features
    available_features = [f for f in features if f in koi.columns]
    print(f"Using {len(available_features)} features: {available_features}")

    data = koi[available_features + ["label"]].copy()
    data = data.dropna()
    
    return data, available_features

def train_random_forest(df, feature_cols):
    X = df[feature_cols]
    y = df["label"]

    # === 3. Normalize numeric data ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === 4. Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # === 5. Train a Random Forest model ===
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # === 6. Evaluate ===
    y_pred = rf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # === 7. Feature Importances ===
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,6))
    plt.title("Feature Importance in Exoplanet Classification (KOI)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("./feature_importance_koi.png")

    return rf, scaler, feature_cols

if __name__ == '__main__':
    dataframe, features = load_and_prepare_koi_data()
    if not dataframe.empty:
        # Treina o modelo e obtém os artefatos
        trained_model, fitted_scaler, feature_list = train_random_forest(dataframe, features)

        # Salva os artefatos para a aplicação Streamlit
        print("\nSalvando o modelo, scaler e lista de features...")
        joblib.dump(trained_model, 'random_forest_model.joblib')
        joblib.dump(fitted_scaler, 'scaler_rf.joblib')
        with open('feature_columns_rf.pkl', 'wb') as f:
            pickle.dump(feature_list, f)
        
        print("Artefatos de IA (RandomForest) salvos com sucesso!")