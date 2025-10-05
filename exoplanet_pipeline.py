import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib
import pickle

# --- 1. Carregamento e Preparação dos Dados ---

def load_and_prepare_koi_data():
    """
    Carrega e prepara os dados do KOI, usando a lógica do script de exemplo.
    """
    print("Usando o serviço TAP da NASA para carregar dados do Kepler (KOI)...")
    url_koi = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select%20%2A%20from%20cumulative&format=csv"
    
    try:
        koi = pd.read_csv(url_koi, comment='#')
        print(f"KOI shape inicial: {koi.shape}")
    except Exception as e:
        print(f"ERRO: Falha ao carregar dados. {e}")
        return pd.DataFrame(), []

    # Coluna alvo e mapeamento
    target_col = "koi_disposition" # Usando koi_disposition por ser mais completo
    koi["label"] = koi[target_col].apply(lambda x: 1 if x in ["CONFIRMED", "CANDIDATE"] else 0)

    # Seleção de features do script de exemplo
    features = [
        "koi_period", "koi_duration", "koi_depth", "koi_impact", 
        "koi_model_snr", # 'koi_snr' é um alias, 'koi_model_snr' é a coluna principal
        "koi_steff", "koi_slogg", "koi_srad",
        "koi_prad", "koi_insol"
    ]

    available_features = [f for f in features if f in koi.columns]
    print(f"Usando {len(available_features)} features: {available_features}")

    data = koi[available_features + ["label"]].copy()

    # Usando SimpleImputer para preencher dados faltantes em vez de dropar
    imputer = SimpleImputer(strategy='median')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    return data_imputed, available_features

# --- 2. Treinamento do Modelo ---

def train_random_forest(df, feature_cols):
    """
    Treina um classificador RandomForest com os dados processados.
    """
    X = df[feature_cols]
    y = df["label"]

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Escalonamento dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinamento do modelo RandomForest
    print("\nIniciando treinamento do modelo RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    print("Treinamento concluído.")

    # Avaliação
    y_pred = rf.predict(X_test_scaled)
    print("\n=== Relatório de Classificação ===")
    print(classification_report(y_test, y_pred, target_names=['Falso Positivo', 'Planeta']))
    
    # Gerar e salvar o gráfico de importância de features
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,6))
    plt.title("Importância das Features na Classificação (KOI - RandomForest)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("./feature_importance_rf.png")
    print("\nGráfico 'feature_importance_rf.png' salvo.")

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
