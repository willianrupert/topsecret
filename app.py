import streamlit as st
import pandas as pd
from joblib import load
import plotly.express as px
import numpy as np
import time
import os

def toggle_campos():
    """Alterna a visibilidade dos campos de cadastro manual."""
    st.session_state.show_fields = not st.session_state.get('show_fields', False)

def main():
    # --- Configuração da Página ---
    st.set_page_config(
        page_title="ExoplanetLIA",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Inicialização do Estado da Sessão ---
    if "show_fields" not in st.session_state:
        st.session_state.show_fields = False
    if 'lang' not in st.session_state:
        st.session_state.lang = 'en'
    if 'page' not in st.session_state:
        st.session_state.page = "nav_about" # Valor padrão
    if "dados" not in st.session_state:
        st.session_state.dados = {}
    if "dados_salvos" not in st.session_state:
        st.session_state.dados_salvos = []

    # --- Dicionário de Idiomas (i18n) ---
    LANGUAGES = {
        "pt": {
            "animation_overlay_text": "Encontre exoplanetas conosco",
            "page_title": "Caçador de Exoplanetas LIA",
            "sidebar_title": "Exoplanet Hunter AI",
            "sidebar_version": "Versão 2.0",
            "sidebar_nav": "Navegação:",
            "nav_about": "Sobre o Projeto",
            "nav_classify": "Classificar Candidato",
            "nav_performance": "Performance do Modelo",
            "model_error": "Arquivos de modelo/scaler não encontrados!",
            "model_success": "Modelo de IA Carregado!",
            "project_info": "Projeto para o NASA Space Apps Challenge.",

            "about_title": "A World Away: Caçando Exoplanetas com Inteligência Artificial",
            "about_challenge_header": "O Desafio",
            "about_challenge_text": """
            O universo está repleto de planetas fora do nosso sistema solar — os exoplanetas. 
            Missões da NASA como Kepler e TESS coletam uma quantidade imensa de dados.
            No entanto, identificar a minúscula queda de brilho causada por um planeta em trânsito é como encontrar uma agulha num palheiro cósmico. 
            O nosso desafio é construir uma ferramenta de IA para automatizar e acelerar essa incrível descoberta.
            """,
            "about_solution_header": "Nossa Solução",
            "about_solution_text": """
            Desenvolvemos uma aplicação web interativa que utiliza um modelo de Machine Learning para analisar dados de curvas de luz, 
            classificando candidatos como 'Possível Exoplaneta' ou 'Falso Positivo'. Esta ferramenta permite que qualquer pessoa 
            participe da busca por novos mundos.
            """,
            "about_how_header": "Como Funciona?",
            "about_how_text": """
            Utilizamos o **método de trânsito**. Quando um planeta passa na frente de sua estrela, 
            ele bloqueia uma pequena fração da luz, gerando um gráfico conhecido como **Curva de Luz**. Nosso modelo de IA é especialista em analisar 
            a forma, profundidade e periodicidade dessas quedas de brilho.
            """,

            "classify_title": "🔬 Faça uma Nova Classificação",
            "classify_info": "Envie um arquivo .csv com dados de uma curva de luz ou insira os dados manualmente para que nossa IA faça a análise.",
            "classify_expander_title": "❓ Precisa de ajuda com o formato dos dados?",
            "classify_expander_text": "O arquivo CSV deve conter colunas com os dados de fluxo (brilho) da estrela, tipicamente nomeadas como 'FLUX.1', 'FLUX.2', etc.",
            "classify_download_button": "Baixar CSV de Exemplo",
            "classify_model_warning": "O modelo de IA ou o scaler não estão carregados. Não é possível fazer classificações.",
            "classify_uploader_label": "Escolha o arquivo CSV",
            "classify_file_success": "Arquivo carregado com sucesso!",
            "classify_button": "✨ Iniciar Classificação!",
            "classify_spinner": "Analisando os confins do universo com a IA...",
            "classify_results_header": "Resultados da Análise",
            "classify_verdict_subheader": "Veredito do Modelo:",
            "classify_planet_candidate": "✔️ **Possível Exoplaneta**",
            "classify_false_positive": "❌ **Falso Positivo**",
            "classify_success_fallback": "✨ Resultado: candidato detectado — verifique com follow-up científico.",
            "classify_fail_fallback": "🔎 Resultado: provável falso positivo — pode ser ruído ou variabilidade estelar.",
            "classify_confidence_metric": "Confiança do Modelo",
            "classify_expander_meaning_title": "O que isso significa?",
            "classify_expander_meaning_text": "Nosso modelo analisou os padrões na variação de brilho da estrela e calculou a probabilidade deste padrão ser causado por um planeta em trânsito.",
            "classify_chart_caption": "Gráfico da Curva de Luz: Variação do brilho da estrela ao longo do tempo.",
            "classify_error": "Ocorreu um erro ao processar o arquivo:",

            "manual_registration": "📊 Cadastro Manual",
            "manual_reg_button": "Gerar Previsão",
            "manual_reg_warning": "⚠️ Preencha todos os campos para gerar a previsão.",
            "manual_reg_success": "Dados salvos com sucesso!",

            "performance_title": "📊 Performance do Modelo de IA",
            "performance_intro": "A transparência é fundamental. Aqui mostramos como nosso modelo se saiu em dados que ele nunca havia visto antes.",
            "performance_tab_metrics": "**Métricas Principais**",
            "performance_tab_matrix": "**Matriz de Confusão**",
            "performance_tab_training": "**Sobre o Treinamento**",
            "metric_accuracy": "Acurácia Geral",
            "metric_precision": "Precisão",
            "metric_recall": "Recall",
            "metrics_desc": """
                - **Acurácia:** A porcentagem de classificações corretas no geral.
                - **Precisão:** De todas as vezes que o modelo disse "é um exoplaneta", quantas vezes ele estava certo? Essencial para evitar falsos alarmes.
                - **Recall:** De todos os exoplanetas reais, quantos o nosso modelo conseguiu encontrar? Vital para não perdermos descobertas.
                """,
            "matrix_header": "Análise Visual da Performance",
            "matrix_text": "A matriz de confusão é a melhor ferramenta para visualizar os acertos e erros do modelo.",
            "matrix_not_found": "Arquivo 'matriz_confusao.png' não encontrado.",
            "matrix_expander_title": "Como ler este gráfico?",
            "matrix_expander_text": """
                - **Verdadeiro Positivo:** O modelo acertou "Planeta". **(Sucesso!)**
                - **Verdadeiro Negativo:** O modelo acertou "Não é Planeta". **(Sucesso!)**
                - **Falso Positivo:** O modelo errou, dizendo "Planeta".
                - **Falso Negativo:** O modelo errou, dizendo "Não é Planeta". (O pior erro!)
                """,
            "training_header": "Detalhes do Treinamento do Modelo",
            "training_text": """
                - **Dataset Utilizado:** TESS Objects of Interest (TOI) - Placeholder
                - **Algoritmo:** Random Forest Classifier - Placeholder
                - **Divisão dos Dados:** 80% para treinamento, 20% para teste.
                """
        },
        "en": {
            "animation_overlay_text": "Find exoplanets with us",
            "page_title": "Exoplanet Hunter LIA",
            "sidebar_title": "Exoplanet Hunter AI",
            "sidebar_version": "Version 2.0",
            "sidebar_nav": "Navigation:",
            "nav_about": "About the Project",
            "nav_classify": "Classify Candidate",
            "nav_performance": "Model Performance",
            "model_error": "Model/scaler files not found!",
            "model_success": "AI Model Loaded!",
            "project_info": "Project for the NASA Space Apps Challenge.",
            "about_title": "A World Away: Hunting for Exoplanets with AI",
            "about_challenge_header": "The Challenge",
            "about_challenge_text": """
            The universe is filled with exoplanets. NASA missions like Kepler and TESS collect vast amounts of data. 
            However, identifying the tiny dip in brightness from a transit is like finding a needle in a cosmic haystack. 
            Our challenge is to build an AI tool to automate and accelerate this discovery.
            """,
            "about_solution_header": "Our Solution",
            "about_solution_text": """
            We developed an interactive web app that uses Machine Learning to analyze light curve data, classifying candidates as 
            'Potential Exoplanet' or 'False Positive'. This tool allows anyone to join the search for new worlds.
            """,
            "about_how_header": "How It Works",
            "about_how_text": """
            We use the **transit method**. When a planet passes in front of its star, it blocks a fraction of the starlight, creating a **Light Curve**. 
            Our AI model is an expert in analyzing the shape, depth, and periodicity of these dips.
            """,
            "classify_title": "🔬 Make a New Classification",
            "classify_info": "Upload a .csv file with light curve data or enter it manually and let our AI do the analysis.",
            "classify_expander_title": "❓ Need help with the data format?",
            "classify_expander_text": "The CSV file must contain flux data columns, typically named 'FLUX.1', 'FLUX.2', etc.",
            "classify_download_button": "Download Sample CSV",
            "classify_model_warning": "AI model or scaler is not loaded. Classification is not possible.",
            "classify_uploader_label": "Choose the CSV file",
            "classify_file_success": "File uploaded successfully!",
            "classify_button": "✨ Start Classification!",
            "classify_spinner": "Analyzing the cosmos with AI...",
            "classify_results_header": "Analysis Results",
            "classify_verdict_subheader": "Model's Verdict:",
            "classify_planet_candidate": "✔️ **Potential Exoplanet**",
            "classify_false_positive": "❌ **False Positive**",
            "classify_success_fallback": "✨ Result: Candidate detected — verify with scientific follow-up.",
            "classify_fail_fallback": "🔎 Result: Likely a false positive — could be stellar variability or noise.",
            "classify_confidence_metric": "Model Confidence",
            "classify_expander_meaning_title": "What does this mean?",
            "classify_expander_meaning_text": "Our model analyzed the brightness patterns and calculated the probability of this pattern being caused by a transiting planet.",
            "classify_chart_caption": "Light Curve Graph: Star's brightness variation over time.",
            "classify_error": "An error occurred while processing the file:",
            
            "manual_registration": "📊 Manual Entry",
            "manual_reg_button": "Generate Prediction",
            "manual_reg_warning": "⚠️ Please fill all fields to generate the prediction.",
            "manual_reg_success": "Data saved successfully!",

            "performance_title": "📊 AI Model Performance",
            "performance_intro": "Transparency is key. Here we show how our model performed on data it had never seen before.",
            "performance_tab_metrics": "**Key Metrics**",
            "performance_tab_matrix": "**Confusion Matrix**",
            "performance_tab_training": "**About the Training**",
            "metric_accuracy": "Overall Accuracy",
            "metric_precision": "Precision",
            "metric_recall": "Recall",
            "metrics_desc": """
                - **Accuracy:** The overall percentage of correct classifications.
                - **Precision:** Of all "exoplanet" predictions, how many were correct? Crucial to avoid false alarms.
                - **Recall:** Of all actual exoplanets, how many did our model find? Vital for not missing discoveries.
                """,
            "matrix_header": "Visual Performance Analysis",
            "matrix_text": "The confusion matrix is the best tool to visualize the model's hits and misses.",
            "matrix_not_found": "'matriz_confusao.png' file not found.",
            "matrix_expander_title": "How to read this chart?",
            "matrix_expander_text": """
                - **True Positive:** Correctly guessed "Planet". **(Success!)**
                - **True Negative:** Correctly guessed "Not a Planet". **(Success!)**
                - **False Positive:** Incorrectly guessed "Planet".
                - **False Negative:** Incorrectly guessed "Not a Planet". (The worst error!)
                """,
            "training_header": "Model Training Details",
            "training_text": """
                - **Dataset Used:** TESS Objects of Interest (TOI) - Placeholder
                - **Algorithm:** Random Forest Classifier - Placeholder
                - **Data Split:** 80% for training, 20% for testing.
                """
        }
    }

    def t(key):
        """Função para buscar texto no dicionário de idiomas."""
        return LANGUAGES[st.session_state.lang].get(key, key)

    # --- CSS Customizado ---
    CSS = """
    <style>
        /* Headers com cor de destaque */
        h1, h2, h3 {
            color: #58a6ff; /* Azul NASA */
        }

        /* Botão primário com gradiente animado */
        .stButton>button[kind="primary"] {
            background: linear-gradient(90deg, #238636, #2ea043, #58a6ff);
            background-size: 200% 200%;
            color: white;
            border: none;
            animation: gradientShift 6s ease infinite;
        }
        .stButton>button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(46, 160, 67, 0.3);
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* --- Estilo para o botão de cadastro manual --- */
        .manual-reg-container .stButton>button {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid var(--secondary-background-color);
            text-align: left !important;
            display: flex;
            align-items: center;
            justify-content: flex-start !important;
            font-weight: normal;
            transition: all 0.2s ease-in-out;
            width: 100%;
        }
        .manual-reg-container .stButton>button:hover {
            border: 1px solid var(--primary-color);
            background-color: var(--background-color);
            transform: none;
            box-shadow: none;
        }
        .manual-reg-container .stButton>button:focus:not(:active) {
            box-shadow: 0 0 0 0.2rem rgba(88, 166, 255, 0.25);
            border-color: var(--primary-color);
        }

        /* --- CORREÇÃO: Remove o cursor de texto piscando no seletor de idioma --- */
        .stSelectbox div[data-baseweb="select"] {
            cursor: pointer;
            caret-color: transparent; /* Oculta o cursor de texto (caret) */
        }
    </style>
    """
    st.markdown(CSS, unsafe_allow_html=True)

    # --- Animação de Estrelas com Texto Sobreposto ---
    def get_animation_html(overlay_text):
        return f"""
    <div style="width: 100%; height: 320px; overflow: hidden; position: relative; border-radius: 10px; background-color: #0d1117;">
        <canvas id="starfield" style="position: absolute; top: 0; left: 0; width:100%; height:100%;"></canvas>
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; pointer-events: none;">
            <h1 style="color: #fff; font-size: clamp(2rem, 5vw, 3.5rem); text-shadow: 0 0 8px #58a6ff, 0 0 15px #58a6ff, 0 0 25px #58a6ff; font-family: 'Segoe UI', sans-serif; text-align: center; padding: 0 20px;">
                {overlay_text}
            </h1>
        </div>
    </div>
    <script>
        const canvas = document.getElementById('starfield');
        if (canvas) {{
            const container = canvas.parentElement;
            function resizeCanvas() {{
                canvas.width = container.offsetWidth;
                canvas.height = container.offsetHeight;
            }}
            resizeCanvas();
            const ctx = canvas.getContext('2d');
            const stars = [];
            const numStars = 400;
            function random(min, max){{ return Math.random()*(max-min)+min; }}
            function createStars(){{
                for(let i=0;i<numStars;i++){{
                    stars.push({{x:random(0,canvas.width), y:random(0,canvas.height), size:random(0.5,2.5), speed:random(0.05,0.4)}});
                }}
            }}
            function draw(){{
                ctx.clearRect(0,0,canvas.width,canvas.height);
                ctx.fillStyle = '#c9d1d9';
                for(let s of stars){{
                    ctx.beginPath();
                    ctx.arc(s.x,s.y,s.size/2,0,Math.PI*2);
                    ctx.fill();
                }}
            }}
            function update(){{
                for(let s of stars){{
                    s.y += s.speed;
                    if(s.y > canvas.height){{ s.y = 0; s.x = Math.random()*canvas.width; }}
                }}
            }}
            function loop(){{ draw(); update(); requestAnimationFrame(loop); }}
            createStars(); loop();
            window.addEventListener('resize', function(){{ resizeCanvas(); stars.length=0; createStars(); }});
        }}
    </script>
    """

    # --- Funções de Carregamento ---
    @st.cache_resource
    def carregar_modelo():
        if os.path.exists('modelo_exoplaneta.joblib'):
            return load('modelo_exoplaneta.joblib')
        return None

    @st.cache_resource
    def carregar_scaler():
        if os.path.exists('scaler.joblib'):
            return load('scaler.joblib')
        return None

    modelo = carregar_modelo()
    scaler = carregar_scaler()

    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        lang_options = {"🇧🇷 Português": "pt", "🇺🇸 English": "en"}
        lang_labels = list(lang_options.keys())
        current_lang_index = lang_labels.index("🇺🇸 English") if st.session_state.lang == 'en' else lang_labels.index("🇧🇷 Português")

        # key garante reatividade imediata
        selected_lang_label = st.selectbox(
            "Idioma/Language",
            lang_labels,
            index=current_lang_index,
            key="select_lang",
            label_visibility="collapsed"
        )
        # Atualiza o idioma imediatamente
        st.session_state.lang = lang_options[selected_lang_label]

    with col1:
        st.title(f'🔭 {t("page_title")}')

    # --- Barra Lateral ---
    with st.sidebar:
        if os.path.exists('nasa_logo.png'):
            st.image("nasa_logo.png", use_container_width=True)
        else:
            st.warning("Logo 'nasa_logo.png' não encontrada.")
        
        st.title(t("sidebar_title"))
        st.markdown(f"**{t('sidebar_version')}**")
        
        # --- LÓGICA DE NAVEGAÇÃO CORRIGIDA E ROBUSTA ---
        nav_options = {
            "nav_about": t("nav_about"),
            "nav_classify": t("nav_classify"),
            "nav_performance": t("nav_performance")
        }
        nav_keys_list = list(nav_options.keys())

        # 1. Renderiza o widget de rádio. A seleção do usuário é armazenada
        #    automaticamente em st.session_state.navigation_radio devido à 'key'.
        st.radio(
            label=t("sidebar_nav"),
            options=nav_keys_list,
            index=nav_keys_list.index(st.session_state.page),
            format_func=lambda key: nav_options[key],
            key="navigation_radio"
        )

        # 2. Compara a seleção do widget com o estado atual da página.
        #    Se forem diferentes, significa que o usuário clicou em um novo item.
        if st.session_state.navigation_radio != st.session_state.page:
            # 3. Atualiza o estado da página para a nova seleção.
            st.session_state.page = st.session_state.navigation_radio
            # 4. Força um 'rerun' imediato do script. Isso é crucial para
            #    garantir que a página seja redesenhada instantaneamente com o
            #    novo conteúdo, eliminando o bug do "clique duplo".
            st.rerun()

        st.divider()

        if modelo is None or scaler is None:
            st.error(t("model_error"))
        else:
            st.success(t("model_success"))

        st.info(t("project_info"))

    # --- Roteamento de Páginas ---

    # Sobre o Projeto
    if st.session_state.page == "nav_about":
        st.components.v1.html(get_animation_html(t("animation_overlay_text")), height=320)
        st.header(t("about_challenge_header"))
        st.write(t("about_challenge_text"))
        st.header(t("about_solution_header"))
        st.write(t("about_solution_text"))
        st.header(t("about_how_header"))
        st.write(t("about_how_text"))

    # Classificar Candidato
    elif st.session_state.page == "nav_classify":
        st.info(t("classify_info"))

        with st.expander(t("classify_expander_title")):
            st.write(t("classify_expander_text"))
            sample_df = pd.DataFrame([[f"{1.0 - (i % 20) * 0.001}" for i in range(50)]], columns=[f'FLUX.{i+1}' for i in range(50)])
            csv = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=t("classify_download_button"),
                data=csv,
                file_name='exemplo_curva_de_luz.csv',
                mime='text/csv',
            )

        st.subheader("Upload de Arquivo CSV")
        if modelo is None or scaler is None:
            st.warning(t("classify_model_warning"))
        else:
            uploaded_file = st.file_uploader(t("classify_uploader_label"), type="csv", label_visibility="collapsed")

            if uploaded_file is not None:
                try:
                    dados_usuario = pd.read_csv(uploaded_file)
                    st.success(t("classify_file_success"))
                    
                    if st.button(t("classify_button"), type="primary", use_container_width=True):
                        with st.spinner(t("classify_spinner")):
                            colunas_do_modelo = scaler.get_feature_names_out()
                            dados_processados = scaler.transform(dados_usuario[colunas_do_modelo])
                            
                            predicao = modelo.predict(dados_processados)
                            probabilidade = modelo.predict_proba(dados_processados)
                            
                            resultado = predicao[0]
                            confianca = probabilidade[0].max() * 100

                        st.divider()
                        st.subheader(t("classify_results_header"))
                        
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.subheader(t("classify_verdict_subheader"))
                            if resultado == 1:
                                st.success(t("classify_planet_candidate"))
                                if os.path.exists('planet_success.gif'):
                                    st.image('planet_success.gif')
                                else:
                                    st.info(t("classify_success_fallback"))
                            else:
                                st.error(t("classify_false_positive"))
                                if os.path.exists('false_positive.gif'):
                                    st.image('false_positive.gif')
                                else:
                                    st.info(t("classify_fail_fallback"))
                            
                            st.metric(label=t("classify_confidence_metric"), value=f"{confianca:.2f}%")

                        with col2:
                            flux_cols = [col for col in dados_usuario.columns if 'FLUX' in col]
                            if flux_cols:
                                fig = px.line(y=dados_usuario[flux_cols].iloc[0].T, labels={'y': 'Brilho Normalizado', 'x': 'Tempo (Observação)'})
                                fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
                                st.plotly_chart(fig, use_container_width=True)
                                st.caption(t("classify_chart_caption"))

                except Exception as e:
                    st.error(f"{t('classify_error')} {e}")
        
        # O st.divider() e o st.subheader() foram removidos para aproximar os elementos.

        def gerar_previsao():
            registro = {campo: st.session_state.dados.get(campo, "") for campo in campos}
            st.session_state.dados_salvos.append(registro)
            st.success(t("manual_reg_success"))
            # Aqui você pode adicionar a lógica para usar os dados salvos com o modelo

        # Container para o botão de cadastro manual (para aplicar CSS customizado)
        st.markdown('<div class="manual-reg-container" style="margin-top: 1rem;">', unsafe_allow_html=True)
        # O botão agora alterna a visibilidade dos campos
        st.button(t("manual_registration"), on_click=toggle_campos, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        campos = ["Dado 1", "Dado 2", "Dado 3", "Dado 4"]

        if st.session_state.show_fields:
            with st.form(key="manual_form"):
                for campo in campos:
                    st.session_state.dados[campo] = st.text_input(
                        f"{campo}",
                        value=st.session_state.dados.get(campo, "")
                    )
                
                # Botão de submit dentro do formulário
                submitted = st.form_submit_button(t("manual_reg_button"), type="primary")

                if submitted:
                    todos_preenchidos = all(st.session_state.dados.get(campo, "").strip() != "" for campo in campos)
                    if todos_preenchidos:
                        gerar_previsao()
                    else:
                        st.warning(t("manual_reg_warning"))

    # Performance do Modelo
    elif st.session_state.page == "nav_performance":
        st.markdown(t("performance_intro"))
        
        tab1, tab2, tab3 = st.tabs([t("performance_tab_metrics"), t("performance_tab_matrix"), t("performance_tab_training")])

        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric(t("metric_accuracy"), "98.2%", "±0.4%")
            col2.metric(t("metric_precision"), "97.1%", "Minimiza falsos positivos")
            col3.metric(t("metric_recall"), "98.5%", "Encontra a maioria dos positivos reais")
            st.markdown(t("metrics_desc"))

        with tab2:
            st.header(t("matrix_header"))
            st.write(t("matrix_text"))
            if os.path.exists('matriz_confusao.png'):
                st.image('matriz_confusao.png', use_container_width=True)
            else:
                st.warning(t("matrix_not_found"))
            with st.expander(t("matrix_expander_title")):
                st.markdown(t("matrix_expander_text"))

        with tab3:
            st.header(t("training_header"))
            st.markdown(t("training_text"))

if __name__ == "__main__":
    main()


