import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg') # Essencial para rodar Matplotlib no Streamlit
import matplotlib.pyplot as plt
import os
from io import StringIO

def main():
    # --- Configuração da Página ---
    st.set_page_config(
        page_title="ExoplanetLIA",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Inicialização do Estado da Sessão ---
    if 'lang' not in st.session_state:
        st.session_state.lang = 'pt'
    if 'page' not in st.session_state:
        st.session_state.page = "nav_about"

    valid_pages = ["nav_about", "nav_classify", "nav_performance"]
    if st.session_state.page not in valid_pages:
        st.session_state.page = "nav_about"

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
            "model_error": "Arquivos do modelo/explainer não encontrados!",
            "model_success": "Modelo de IA Carregado!",
            "project_info": "Projeto para o NASA Space Apps Challenge.",
            "logo_not_found_warning": "Arquivo 'nasa_logo.png' não encontrado.",
            "about_title": "A World Away: Caçando Exoplanetas com Inteligência Artificial",
            "about_challenge_header": "O Desafio",
            "about_challenge_text": """
            O universo está repleto de planetas fora do nosso sistema solar — os exoplanetas. 
            Missões da NASA como Kepler e TESS coletam uma quantidade imensa de dados.
            O nosso desafio é construir uma ferramenta de IA para automatizar e acelerar essa incrível descoberta.
            """,
            "about_solution_header": "Nossa Solução",
            "about_solution_text": """
            Desenvolvemos uma aplicação web interativa que utiliza um modelo de Machine Learning para analisar dados tabulares de candidatos a exoplanetas, 
            classificando-os como 'Planeta Confirmado' ou 'Falso Positivo'. Nossa ferramenta não apenas classifica, mas também **explica suas decisões**,
            trazendo transparência para a ciência de dados.
            """,
            "about_how_header": "Como Funciona?",
            "about_how_text": """
            A partir de dados processados de missões como Kepler, nosso modelo de IA (baseado em XGBoost) aprende a identificar os padrões sutis que diferenciam um trânsito planetário real de outros fenômenos astrofísicos. 
            Utilizamos a tecnologia **SHAP** para visualizar exatamente quais parâmetros mais influenciaram cada predição.
            """,

            "classify_title": "🔬 Classifique um Candidato a Exoplaneta",
            "classify_intro": "Escolha um método: envie um arquivo CSV com múltiplos candidatos ou insira os dados de um único candidato manualmente.",
            "upload_header": "Opção 1: Upload de Arquivo CSV",
            "manual_header": "Opção 2: Entrada Manual de Dados",
            "classify_uploader_label": "Escolha o arquivo CSV",
            "classify_expander_title": "❓ Precisa de ajuda com o formato do CSV?",
            "classify_expander_text": "O arquivo CSV deve conter as 10 colunas de features que o modelo espera. O sistema tentará reconhecer nomes alternativos comuns (ex: 'koi_period' será mapeado para 'period').",
            "classify_download_button": "Baixar CSV de Exemplo",
            "classify_button": "✨ Classificar!",
            "classify_spinner": "Analisando os confins do universo com a IA...",
            "classify_results_header": "Resultados da Análise",
            "classify_verdict_subheader": "Veredito do Modelo:",
            "classify_planet_candidate": "✔️ **Planeta Confirmado**",
            "classify_false_positive": "❌ **Falso Positivo**",
            "classify_confidence_metric": "Confiança do Modelo",
            "xai_header": "Explicação da IA (XAI com SHAP)",
            "xai_text": "Este gráfico mostra quais fatores influenciaram a decisão do modelo. Forças em **vermelho** empurram a decisão para 'Planeta', enquanto forças em **azul** empurram para 'Falso Positivo'.",
            "results_table_header": "Resultados para o arquivo enviado",
            "performance_title": "📊 Performance do Modelo de IA",
            "performance_intro": "A transparência é fundamental. Aqui mostramos como nosso modelo se saiu em dados que ele nunca havia visto antes.",
            "performance_tab_metrics": "**Métricas Principais**",
            "performance_tab_matrix": "**Importância das Features**",
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
            "matrix_text": "O gráfico de importância de features SHAP mostra o impacto médio de cada característica nas predições do modelo.",
            "matrix_not_found": "Arquivo 'shap_global_importance_optimized.png' não encontrado.",
            "matrix_expander_title": "Como ler este gráfico?",
            "matrix_expander_text": """
                - As barras mais longas representam as features mais influentes.
                - Este gráfico nos ajuda a entender em quais dados o modelo mais confia para tomar suas decisões.
                """,
            "training_header": "Detalhes do Treinamento do Modelo",
            "training_text": """
                - **Dataset Utilizado:** Kepler Objects of Interest (KOI) Cumulative
                - **Algoritmo:** XGBoost (com otimização GridSearchCV)
                - **Features:** 10 características principais do trânsito e da estrela hospedeira.
                - **Divisão dos Dados:** 70% para treinamento, 30% para teste.
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
            "model_error": "Model/explainer files not found!",
            "model_success": "AI Model Loaded!",
            "project_info": "Project for the NASA Space Apps Challenge.",
            "logo_not_found_warning": "'nasa_logo.png' file not found.",
            "about_title": "A World Away: Hunting for Exoplanets with AI",
            "about_challenge_header": "The Challenge",
            "about_challenge_text": """
            The universe is filled with exoplanets. NASA missions like Kepler and TESS collect vast amounts of data. 
            Our challenge is to build an AI tool to automate and accelerate this discovery.
            """,
            "about_solution_header": "Our Solution",
            "about_solution_text": """
            We developed an interactive web app using a Machine Learning model to analyze tabular data of exoplanet candidates, classifying them as 'Confirmed Planet' or 'False Positive'. 
            Our tool not only classifies but also **explains its decisions**, bringing transparency to data science.
            """,
            "about_how_header": "How It Works",
            "about_how_text": """
            Using processed data from missions like Kepler, our AI model (based on XGBoost) learns to identify the subtle patterns that differentiate a real planetary transit from other astrophysical phenomena. 
            We use **SHAP** technology to visualize exactly which parameters most influenced each prediction.
            """,
            "classify_title": "🔬 Classify an Exoplanet Candidate",
            "classify_intro": "Choose a method: upload a CSV file with multiple candidates or enter the data for a single candidate manually.",
            "upload_header": "Option 1: CSV File Upload",
            "manual_header": "Option 2: Manual Data Entry",
            "classify_uploader_label": "Choose the CSV file",
            "classify_expander_title": "❓ Need help with the CSV format?",
            "classify_expander_text": "The CSV file must contain the 10 feature columns the model expects. The system will try to recognize common alternative names (e.g., 'koi_period' will be mapped to 'period').",
            "classify_download_button": "Download Sample CSV",
            "classify_button": "✨ Classify!",
            "classify_spinner": "Analyzing the cosmos with AI...",
            "classify_results_header": "Analysis Results",
            "classify_verdict_subheader": "Model's Verdict:",
            "classify_planet_candidate": "✔️ **Confirmed Planet**",
            "classify_false_positive": "❌ **False Positive**",
            "classify_confidence_metric": "Model Confidence",
            "xai_header": "AI Explanation (XAI with SHAP)",
            "xai_text": "This chart shows what factors influenced the model's decision. Forces in **red** push the prediction toward 'Planet', while forces in **blue** push it toward 'False Positive'.",
            "results_table_header": "Results for the uploaded file",
            "performance_title": "📊 AI Model Performance",
            "performance_intro": "Transparency is key. Here we show how our model performed on data it had never seen before.",
            "performance_tab_metrics": "**Key Metrics**",
            "performance_tab_matrix": "**Feature Importance**",
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
            "matrix_text": "The SHAP feature importance chart shows the average impact of each feature on the model's predictions.",
            "matrix_not_found": "'shap_global_importance_optimized.png' file not found.",
            "matrix_expander_title": "How to read this chart?",
            "matrix_expander_text": """
                - Longer bars represent more influential features.
                - This chart helps us understand which data the model relies on most to make its decisions.
                """,
            "training_header": "Model Training Details",
            "training_text": """
                - **Dataset Used:** Kepler Objects of Interest (KOI) Cumulative
                - **Algorithm:** XGBoost (with GridSearchCV optimization)
                - **Features:** 10 key characteristics of the transit and host star.
                - **Data Split:** 70% for training, 30% for testing.
                """
        }
    }

    def t(key):
        return LANGUAGES[st.session_state.lang].get(key, key)

    COLUMN_ALIASES = {
        'period': ['koi_period', 'pl_orbper', 'toi_period'],
        'duration': ['koi_duration', 'pl_trandur', 'toi_duration'],
        'depth': ['koi_depth', 'pl_trandep', 'toi_depth'],
        'prad': ['koi_prad', 'pl_rade', 'toi_prad'],
        'teq': ['koi_teq', 'pl_eqt', 'toi_steq'],
        'insol': ['koi_insol', 'pl_insol', 'toi_insol'],
        'model_snr': ['koi_model_snr', 'toi_model_snr'],
        'steff': ['koi_steff', 'st_teff', 'toi_steff'],
        'slogg': ['koi_slogg', 'st_logg', 'toi_slogg'],
        'srad': ['koi_srad', 'st_rad', 'toi_srad']
    }

    def rename_uploaded_columns(df):
        rename_map = {}
        df_columns_lower = [str(col).lower() for col in df.columns]

        for standard_name, alias_list in COLUMN_ALIASES.items():
            for alias in alias_list:
                if alias.lower() in df_columns_lower:
                    original_col_index = df_columns_lower.index(alias.lower())
                    original_col_name = df.columns[original_col_index]
                    rename_map[original_col_name] = standard_name
                    break
        
        df.rename(columns=rename_map, inplace=True)
        return df

    @st.cache_resource
    def load_artifacts():
        try:
            model = joblib.load('xgboost_model_koi_only.joblib')
            scaler = joblib.load('scaler_koi_only.joblib')
            explainer = shap.TreeExplainer(model)
            with open('feature_columns_koi_only.pkl', 'rb') as f:
                feature_columns = pickle.load(f)
            return model, scaler, explainer, feature_columns
        except FileNotFoundError:
            return None, None, None, None

    model, scaler, explainer, feature_columns = load_artifacts()

    st.title(f'🔭 {t("page_title")}')

    with st.sidebar:
        # **CORREÇÃO: Verifica se a imagem existe antes de exibi-la**
        if os.path.exists('nasa_logo.png'):
            st.image("nasa_logo.png") # Removido use_container_width
        else:
            st.warning(t("logo_not_found_warning"))

        st.title(t("sidebar_title"))
        st.markdown(f"**{t('sidebar_version')}**")
        
        nav_options = {
            "nav_about": t("nav_about"),
            "nav_classify": t("nav_classify"),
            "nav_performance": t("nav_performance")
        }
        nav_keys_list = list(nav_options.keys())

        st.radio(
            label=t("sidebar_nav"),
            options=nav_keys_list,
            index=nav_keys_list.index(st.session_state.page),
            format_func=lambda key: nav_options[key],
            key="navigation_radio"
        )

        if st.session_state.navigation_radio != st.session_state.page:
            st.session_state.page = st.session_state.navigation_radio
            st.rerun()

        st.divider()

        if model is None:
            st.error(t("model_error"))
        else:
            st.success(t("model_success"))

        st.info(t("project_info"))

    if st.session_state.page == "nav_about":
        st.components.v1.html(get_animation_html(t("animation_overlay_text")), height=320)
        st.header(t("about_challenge_header"))
        st.write(t("about_challenge_text"))
        st.header(t("about_solution_header"))
        st.write(t("about_solution_text"))
        st.header(t("about_how_header"))
        st.write(t("about_how_text"))

    elif st.session_state.page == "nav_classify":
        st.header(t("classify_title"))
        st.write(t("classify_intro"))

        if model is None or scaler is None or explainer is None:
            st.error(t("model_error"))
        else:
            st.subheader(t("upload_header"))
            with st.expander(t("classify_expander_title")):
                st.write(t("classify_expander_text"))
                sample_data = {col: [np.random.rand()*100] for col in feature_columns}
                sample_df = pd.DataFrame(sample_data)
                csv = sample_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=t("classify_download_button"),
                    data=csv,
                    file_name='exemplo_candidatos_koi.csv',
                    mime='text/csv',
                )

            uploaded_file = st.file_uploader(t("classify_uploader_label"), type="csv", label_visibility="collapsed")

            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file, comment='#', engine='python')
                    
                    df_renamed = rename_uploaded_columns(df_upload.copy())
                    
                    if set(feature_columns).issubset(df_renamed.columns):
                        df_to_predict = df_renamed[feature_columns]
                        df_scaled = scaler.transform(df_to_predict)
                        
                        if st.button(t("classify_button"), key="csv_classify", type="primary"):
                             with st.spinner(t("classify_spinner")):
                                predictions = model.predict(df_scaled)
                                probas = model.predict_proba(df_scaled)

                                df_renamed['Predição'] = ['Planeta Confirmado' if p == 1 else 'Falso Positivo' for p in predictions]
                                df_renamed['Confiança'] = [f"{p.max()*100:.2f}%" for p in probas]
                             
                             st.subheader(t("results_table_header"))
                             st.dataframe(df_renamed)
                    else:
                        st.error(f"O arquivo CSV não contém todas as colunas necessárias ou seus sinônimos. Faltando: {list(set(feature_columns) - set(df_renamed.columns))}")
                except Exception as e:
                    st.error(f"Erro ao processar o arquivo CSV: {e}")

            st.divider()

            st.subheader(t("manual_header"))
            
            with st.form(key="manual_form"):
                default_values = {
                    'period': 8.7, 'duration': 2.4, 'depth': 846.0, 'prad': 2.2, 'teq': 921.0, 
                    'insol': 163.7, 'model_snr': 24.8, 'steff': 5912.0, 'slogg': 4.4, 'srad': 0.9
                }
                
                input_data = {}
                cols = st.columns(3)
                
                for i, col_name in enumerate(feature_columns):
                    input_data[col_name] = cols[i % 3].number_input(
                        label=col_name,
                        value=default_values.get(col_name, 0.0),
                        format="%.4f"
                    )

                submitted = st.form_submit_button(t("classify_button"), type="primary")

                if submitted:
                    with st.spinner(t("classify_spinner")):
                        input_df = pd.DataFrame([input_data])
                        input_scaled = scaler.transform(input_df)
                        
                        prediction_proba = model.predict_proba(input_scaled)[0]
                        prediction = int(np.argmax(prediction_proba))
                        confidence = float(max(prediction_proba))
                        
                        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_columns)
                        shap_values = explainer.shap_values(input_scaled_df)

                    st.subheader(t("classify_results_header"))
                    col1, col2 = st.columns([0.4, 0.6])

                    with col1:
                        st.subheader(t("classify_verdict_subheader"))
                        if prediction == 1:
                            st.success(t("classify_planet_candidate"))
                        else:
                            st.error(t("classify_false_positive"))
                        
                        st.metric(label=t("classify_confidence_metric"), value=f"{confidence*100:.2f}%")

                    with col2:
                        st.subheader(t("xai_header"))
                        st.write(t("xai_text"))
                        
                        fig, ax = plt.subplots(figsize=(8, 2))
                        shap.force_plot(
                            explainer.expected_value,
                            shap_values[0,:],
                            input_scaled_df.iloc[0,:],
                            matplotlib=True,
                            show=False,
                            text_rotation=10
                        )
                        st.pyplot(fig, bbox_inches='tight')
                        plt.close(fig)

    elif st.session_state.page == "nav_performance":
        st.header(t("performance_title"))
        st.write(t("performance_intro"))
        
        tab1, tab2, tab3 = st.tabs([t("performance_tab_metrics"), t("performance_tab_matrix"), t("performance_tab_training")])

        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric(t("metric_accuracy"), "94.0%")
            col2.metric(t("metric_precision"), "91.0%")
            col3.metric(t("metric_recall"), "94.0%")
            st.markdown(t("metrics_desc"))

        with tab2:
            st.header(t("matrix_header"))
            st.write(t("matrix_text"))
            if os.path.exists('shap_global_importance_optimized.png'):
                st.image('shap_global_importance_optimized.png')
            else:
                st.warning(t("matrix_not_found"))
            with st.expander(t("matrix_expander_title")):
                st.markdown(t("matrix_expander_text"))

        with tab3:
            st.header(t("training_header"))
            st.markdown(t("training_text"))

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

if __name__ == "__main__":
    main()

