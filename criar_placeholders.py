import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump

# 1. Criar um conjunto de dados falso com a estrutura esperada
#    Vamos criar 10 colunas de 'FLUX' como exemplo.
colunas_falsas = [f'FLUX.{i+1}' for i in range(10)]
dados_falsos = pd.DataFrame([[1] * 10], columns=colunas_falsas)
labels_falsos = pd.Series([1]) # Apenas um label de exemplo

# 2. Criar um Scaler "treinado" nos dados falsos
print("Criando scaler de placeholder...")
scaler_placeholder = StandardScaler()
scaler_placeholder.fit(dados_falsos)

# 3. Criar um Modelo de IA "treinado"
#    O DummyClassifier é um modelo falso do scikit-learn, perfeito para isso.
#    Ele sempre prevê a classe mais frequente dos dados de treino.
print("Criando modelo de placeholder...")
modelo_placeholder = DummyClassifier(strategy='most_frequent')
modelo_placeholder.fit(dados_falsos, labels_falsos)

# 4. Salvar os placeholders com os nomes corretos
dump(scaler_placeholder, 'scaler.joblib')
dump(modelo_placeholder, 'modelo_exoplaneta.joblib')

print("\nSucesso! Arquivos 'scaler.joblib' e 'modelo_exoplaneta.joblib' criados.")
print("Eles podem ser usados para o deploy inicial.")

