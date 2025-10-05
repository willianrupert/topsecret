import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle

# --- 1. Carregamento e Pré-processamento dos Dados (Apenas KOI) ---

def load_koi_data():
    """
    Carrega e processa dados apenas do catálogo Kepler (KOI).
    """
    print("Usando o serviço TAP da NASA para carregar dados do Kepler (KOI)...")
    url_koi = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select%20%2A%20from%20cumulative&format=csv"

    column_mapping = {
        'period': 'koi_period', 'duration': 'koi_duration',
        'depth': 'koi_depth', 'prad': 'koi_prad',
        'teq': 'koi_teq', 'insol': 'koi_insol',
        'model_snr': 'koi_model_snr', 'steff': 'koi_steff',
        'slogg': 'koi_slogg', 'srad': 'koi_srad'
    }
    
    disposition_col = ('koi_disposition', ['CONFIRMED', 'FALSE POSITIVE'])

    print("Processando dados...")
    
    try:
        df = pd.read_csv(url_koi, comment='#')
        df_processed = pd.DataFrame()
        
        for unified_name, source_name in column_mapping.items():
            if source_name in df.columns:
                df_processed[unified_name] = pd.to_numeric(df[source_name], errors='coerce')

        disp_col_name, valid_disps = disposition_col
        if disp_col_name in df.columns:
            df_processed['disposition'] = df[disp_col_name]
            df_processed = df_processed[df_processed['disposition'].isin(valid_disps)]
            df_processed['is_planet'] = df_processed['disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)
            df_processed = df_processed.drop(columns=['disposition'])

        for col in df_processed.columns:
            if df_processed[col].isnull().any():
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)

        final_feature_cols = list(column_mapping.keys())
        
        print(f"\nDados do KOI processados. Shape final: {df_processed.shape}")
        print("\nDistribuição das classes:")
        print(df_processed['is_planet'].value_counts())
        return df_processed, final_feature_cols

    except Exception as e:
        print(f"    ERRO: Falha ao carregar ou processar dados do KOI. Erro: {e}.")
        return pd.DataFrame(), []

# --- 2. Treinamento Otimizado do Modelo ---
def train_exoplanet_classifier(df, feature_cols):
    X = df[feature_cols]
    y = df['is_planet']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("\nEscalonando os dados...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scale_pos_weight = 1 if y_train.value_counts().get(1, 0) == 0 else y_train.value_counts()[0] / y_train.value_counts()[1]
    
    print("\nIniciando busca por hiperparâmetros para o modelo XGBoost...")
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200]
    }

    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, scale_pos_weight=scale_pos_weight, random_state=42)
    
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    print("\nBusca por hiperparâmetros concluída.")
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    
    print("\nAvaliação do Modelo Otimizado no conjunto de teste:")
    preds = best_model.predict(X_test_scaled)
    print(f"Acurácia: {accuracy_score(y_test, preds):.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, preds, target_names=['Falso Positivo', 'Planeta']))
    
    return best_model, X_test_scaled, y_test, scaler, feature_cols, pd.DataFrame(X_test_scaled, columns=feature_cols)


if __name__ == '__main__':
    koi_df, feature_columns = load_koi_data()
    if not koi_df.empty and feature_columns:
        trained_model, X_test_data, y_test_data, fitted_scaler, feature_cols, X_test_df = train_exoplanet_classifier(koi_df, feature_columns)
        
        # O explainer não é mais salvo, apenas o modelo, scaler e colunas.
        print("\nSalvando o modelo, scaler e lista de features...")
        joblib.dump(trained_model, 'xgboost_model_koi_only.joblib')
        joblib.dump(fitted_scaler, 'scaler_koi_only.joblib')
            
        with open('feature_columns_koi_only.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)

        print("Artefatos de IA salvos com sucesso!")
        
    else:
        print("Nenhum dado foi carregado ou processado. Encerrando o script.")

