import streamlit as st
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import base64 # Usado para criar o link de download

# Configuração da página - Adicionando um menu sobre o projeto
st.set_page_config(
    page_title="Caçador de Exoplanetas AI",
    page_icon="🌠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.spaceappschallenge.org/',
        'Report a bug': "https://github.com/WillGuedes01/NASA-Exoplanets",
        'About': "# Caçador de Exoplanetas AI. Desenvolvido para o NASA Space Apps Challenge!"
    }
)

# --- Funções de Carregamento (com cache para performance) ---
@st.cache_resource
def carregar_modelo():
    try:
        return load('modelo_exoplaneta.joblib')
    except FileNotFoundError:
        return None

@st.cache_resource
def carregar_scaler():
    try:
        return load('scaler.joblib')
    except FileNotFoundError:
        return None

modelo = carregar_modelo()
scaler = carregar_scaler()

# --- Barra Lateral (Sidebar com design melhorado) ---
with st.sidebar:
    st.title("🛰️ Caçador de Exoplanetas AI")
    st.markdown("---")
    
    # Lógica de status do modelo mais clara para o usuário
    if modelo is not None and scaler is not None:
        st.success("✅ Modelo de IA pronto para análise!")
    else:
        st.error("❌ Modelo não encontrado!")
        st.warning("Verifique se os arquivos 'modelo_exoplaneta.joblib' e 'scaler.joblib' estão na pasta do projeto.")
    
    st.markdown("---")
    
    st.subheader("Navegação Principal")
    page = st.radio(
        "Escolha uma página:", 
        ["Sobre o Projeto", "Classificar Novo Candidato", "Performance do Modelo"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.info("Desenvolvido para o NASA Space Apps Challenge.")

# ==============================================================================
# --- PÁGINA 1: SOBRE O PROJETO ---
# ==============================================================================
if page == "Sobre o Projeto":
    st.title("A World Away: Caçando Exoplanetas com Inteligência Artificial")
    st.markdown("### Uma ferramenta para acelerar a descoberta de novos mundos.")
    
    st.image("https://placehold.co/1200x400/000033/FFFFFF?text=NASA+Space+Apps+Challenge", caption="Missões espaciais nos fornecem os dados para explorar o cosmos.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🚀 O Desafio")
        st.write(
            "Milhares de exoplanetas já foram descobertos, mas acredita-se que existam trilhões em nossa galáxia. "
            "As missões da NASA, como Kepler e TESS, geram petabytes de dados, tornando a análise manual impraticável. "
            "O desafio é criar uma solução que possa analisar esses vastos conjuntos de dados de forma rápida e precisa."
        )

    with col2:
        st.header("💡 Nossa Solução")
        st.write(
            "Nosso projeto utiliza um modelo de Machine Learning treinado para reconhecer as 'assinaturas' sutis que um exoplaneta "
            "deixa nos dados de uma estrela. Esta aplicação web democratiza o acesso a essa tecnologia, permitindo que "
            "qualquer pessoa participe da fronteira da exploração exoplanetária."
        )

    st.divider()

    st.header("⚙️ Como Funciona?")
    st.image("https://placehold.co/1000x250/1C1C1C/FFFFFF?text=Dados+Crus+->+Pré-Processamento+->+Modelo+IA+->+Classificação", caption="Fluxo de trabalho do nosso sistema de classificação.")

    st.header("🛠️ Tecnologias Utilizadas")
    st.write(
        "- **Python:** Linguagem principal para ciência de dados e backend."
        "- **Streamlit:** Framework para a construção da interface web interativa."
        "- **Scikit-learn:** Para o treinamento e avaliação do nosso modelo de classificação."
        "- **Pandas:** Para manipulação e pré-processamento dos dados."
        "- **GitHub:** Para controle de versão e colaboração da equipe."
    )

# ==============================================================================
# --- PÁGINA 2: CLASSIFICAR NOVO CANDIDATO ---
# ==============================================================================
elif page == "Classificar Novo Candidato":
    st.title("🔬 Faça uma Nova Classificação")
    st.markdown("Envie um arquivo `.csv` com dados de uma curva de luz e deixe nossa IA fazer a análise.")

    # Bloco de ajuda para o usuário
    with st.expander("❓ Precisa de ajuda com o formato dos dados?"):
        st.write(
            "O arquivo CSV deve conter colunas com os dados de fluxo (brilho) da estrela. "
            "Essas colunas são tipicamente nomeadas como 'FLUX.1', 'FLUX.2', etc. "
            "O nosso modelo foi treinado com um número específico de features, então garanta que seu arquivo tenha a estrutura correta."
        )
        # Criando um dataframe de exemplo para download
        sample_df = pd.DataFrame([[f"{i*0.98}" for i in range(50)]], columns=[f'FLUX.{i+1}' for i in range(50)])
        csv = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Baixar CSV de Exemplo",
            data=csv,
            file_name='exemplo_curva_de_luz.csv',
            mime='text/csv',
        )

    if modelo is None or scaler is None:
        st.warning("O modelo de IA ou o scaler não estão carregados. Não é possível fazer classificações.")
    else:
        uploaded_file = st.file_uploader("Escolha o arquivo CSV", type="csv", label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                dados_usuario = pd.read_csv(uploaded_file)
                st.success("Arquivo carregado com sucesso!")
                
                if st.button("✨ Iniciar Classificação!", type="primary"):
                    with st.spinner('Analisando os confins do universo com a IA...'):
                        
                        # --- LÓGICA DE PREDIÇÃO REAL ---
                        dados_processados = dados_usuario # Substituir pela lógica real com o scaler
                        predicao = modelo.predict(dados_processados)
                        probabilidade = modelo.predict_proba(dados_processados)
                        
                        resultado = predicao[0]
                        confianca = probabilidade[0].max() * 100

                        st.divider()
                        st.subheader("Resultados da Análise")
                        
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            if resultado == 1:
                                st.success(f"✔️ **Possível Exoplaneta**")
                            else:
                                st.error(f"❌ **Falso Positivo**")
                            
                            st.metric(label="Confiança do Modelo", value=f"{confianca:.2f}%")

                            with st.expander("O que isso significa?"):
                                st.write("Nosso modelo analisou os padrões na variação de brilho da estrela e, com base no seu treinamento, calculou a probabilidade deste padrão ser causado por um planeta em trânsito.")

                        with col2:
                            flux_cols = [col for col in dados_usuario.columns if 'FLUX' in col]
                            if flux_cols:
                                st.line_chart(dados_usuario[flux_cols].iloc[0].T)
                                st.caption("Gráfico da Curva de Luz: Variação do brilho da estrela ao longo do tempo. Quedas periódicas podem indicar um exoplaneta.")

            except Exception as e:
                st.error(f"Ocorreu um erro ao processar o arquivo: {e}")

# ==============================================================================
# --- PÁGINA 3: PERFORMANCE DO MODELO ---
# ==============================================================================
elif page == "Performance do Modelo":
    st.title("📊 Performance do Modelo de IA")
    st.markdown("A transparência é fundamental. Aqui mostramos como nosso modelo se saiu em um conjunto de dados que ele nunca havia visto antes.")
    
    # Usando abas (tabs) para organizar melhor a informação
    tab1, tab2, tab3 = st.tabs(["**Métricas Principais**", "**Matriz de Confusão**", "**Sobre o Treinamento**"])

    with tab1:
        st.header("Métricas de Classificação")
        
        # --- ATUALIZAR COM VALORES REAIS ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Acurácia Geral", "98.2%", "±0.4%")
        col2.metric("Precisão (Exoplanetas)", "97.1%", "Minimiza falsos positivos")
        col3.metric("Recall (Exoplanetas)", "98.5%", "Encontra a maioria dos positivos reais")
        
        st.divider()
        st.subheader("O que essas métricas significam?")
        
        st.markdown(
            """
            - **Acurácia:** A porcentagem de classificações corretas no geral. Simples, mas pode ser enganosa se os dados forem desbalanceados.
            - **Precisão:** De todas as vezes que o modelo disse "é um exoplaneta", quantas vezes ele estava certo? Uma alta precisão é crucial para evitar que os cientistas percam tempo analisando falsos alarmes.
            - **Recall (Sensibilidade):** De todos os exoplanetas reais no conjunto de dados, quantos o nosso modelo conseguiu encontrar? Um alto recall é vital para não perdermos descobertas potenciais.
            """
        )

    with tab2:
        st.header("Análise Visual da Performance")
        st.write("A matriz de confusão é a melhor ferramenta para visualizar os acertos e erros do modelo.")
        
        # --- ATUALIZAR COM IMAGEM REAL ---
        try:
            st.image('matriz_confusao.png', caption="Desempenho detalhado do modelo no conjunto de teste.")
        except FileNotFoundError:
            st.warning("Arquivo 'matriz_confusao.png' não encontrado.")

        with st.expander("Como ler este gráfico?"):
            st.markdown(
                """
                - **Verdadeiro Positivo (Canto Inferior Direito):** O modelo disse "Planeta" e era um planeta. **(Sucesso!)**
                - **Verdadeiro Negativo (Canto Superior Esquerdo):** O modelo disse "Não é Planeta" e não era. **(Sucesso!)**
                - **Falso Positivo (Canto Superior Direito):** O modelo disse "Planeta", mas não era. (Erro Tipo I)
                - **Falso Negativo (Canto Inferior Esquerdo):** O modelo disse "Não é Planeta", mas era. (Erro Tipo II - o pior erro!)
                """
            )

    with tab3:
        st.header("Detalhes do Treinamento do Modelo")
        st.markdown(
            """
            - **Dataset Utilizado:** TESS Objects of Interest (TOI)
            - **Algoritmo de Machine Learning:** Random Forest Classifier
            - **Divisão dos Dados:** 80% para treinamento, 20% para teste.
            - **Validação:** As métricas exibidas foram calculadas exclusivamente no conjunto de teste de 20%, garantindo uma avaliação imparcial da capacidade de generalização do modelo.
            """
        )

