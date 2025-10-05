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
from PIL import Image


def main():
    # --- Configuração da Página ---
    st.set_page_config(
        page_title="Exoplanet Hunter LIA",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Dicionário de Idiomas (i18n) ---
    LANGUAGES = {
        "pt": {
            "lang_button": "English",
            "animation_overlay_text": "Encontre exoplanetas conosco",
            "page_title": "Caçador de Exoplanetas LIA",
            "sidebar_title": "Exoplanet Hunter AI",
            "sidebar_version": "Versão RandomForest",
            "sidebar_nav": "Navegação:",
            "nav_about": "Sobre o Projeto",
            "nav_classify": "Classificar Candidato",
            "nav_performance": "Performance do Modelo",
            "model_error": "Arquivos do modelo não encontrados! Por favor, execute o script de treinamento.",
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
            classificando-os como 'Planeta' ou 'Falso Positivo'. Nossa ferramenta não apenas classifica, mas também **explica suas decisões**,
            trazendo transparência para a ciência de dados.
            """,
            "about_how_header": "Como Funciona?",
            "about_how_text": """
            A partir de dados da missão Kepler (KOI), nosso modelo de IA (baseado em RandomForest) aprende a identificar os padrões sutis que diferenciam um trânsito planetário real de outros fenômenos astrofísicos. 
            Utilizamos a tecnologia **SHAP** para visualizar exatamente quais parâmetros mais influenciaram cada predição.
            """,
            "classify_title": "🔬 Classifique um Candidato a Exoplaneta",
            "classify_intro": "Escolha um método: envie um arquivo CSV ou insira os dados de um único candidato manualmente.",
            "upload_header": "Opção 1: Upload de Arquivo CSV",
            "manual_header": "Opção 2: Entrada Manual de Dados",
            "classify_expander_title": "❓ Precisa de ajuda com o formato do CSV?",
            "classify_expander_text": "O arquivo CSV deve conter as features que o modelo espera. O sistema tentará reconhecer nomes alternativos.",
            "classify_uploader_label": "Selecione o arquivo CSV",
            "classify_download_button": "Baixar CSV de Exemplo",
            "classify_button": "✨ Classificar!",
            "classify_spinner": "Analisando os confins do universo com a IA...",
            "classify_results_header": "Resultados da Análise",
            "classify_verdict_subheader": "Veredito do Modelo:",
            "classify_planet_candidate": "✔️ **Planeta**",
            "classify_false_positive": "❌ **Falso Positivo**",
            "classify_confidence_metric": "Confiança do Modelo",
            "xai_header": "Explicação da IA (XAI com SHAP)",
            "results_table_header": "Resultados para o arquivo enviado",
            "prediction_col": "Predição",
            "confidence_col": "Confiança",
            "planet_val": "Planeta",
            "false_positive_val": "Falso Positivo",
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
            - **Precisão:** De todas as vezes que o modelo disse "é um planeta", quantas vezes ele estava certo?
            - **Recall:** De todos os planetas reais, quantos o nosso modelo conseguiu encontrar?
            """,
            "matrix_header": "Análise Visual da Performance",
            "matrix_text": "O gráfico de importância de features mostra o impacto médio de cada característica nas predições do modelo.",
            "matrix_not_found": "Arquivo 'feature_importance_rf.png' não encontrado.",
            "matrix_expander_title": "Como ler este gráfico?",
            "matrix_expander_text": "As barras mais longas representam as features mais influentes para o modelo.",
            "training_header": "Detalhes do Treinamento do Modelo",
            "training_text": """
            - **Dataset Utilizado:** Kepler Objects of Interest (KOI)
            - **Algoritmo:** RandomForestClassifier
            - **Features:** As features utilizadas são carregadas dinamicamente a partir do modelo treinado.
            - **Divisão dos Dados:** 80% para treinamento, 20% para teste.
            """
        },
        "en": {
            "lang_button": "Português",
            "animation_overlay_text": "Find exoplanets with us",
            "page_title": "Exoplanet Hunter LIA",
            "sidebar_title": "Exoplanet Hunter AI",
            "sidebar_version": "RandomForest Version",
            "sidebar_nav": "Navigation:",
            "nav_about": "About the Project",
            "nav_classify": "Classify Candidate",
            "nav_performance": "Model Performance",
            "model_error": "Model files not found! Please run the training script.",
            "model_success": "AI Model Loaded!",
            "project_info": "Project for the NASA Space Apps Challenge.",
            "logo_not_found_warning": "'nasa_logo.png' file not found.",
            "about_title": "A World Away: Hunting for Exoplanets with AI",
            "about_challenge_header": "The Challenge",
            "about_challenge_text": "The universe is filled with exoplanets. NASA missions like Kepler collect vast amounts of data. Our challenge is to build an AI tool to automate and accelerate this discovery.",
            "about_solution_header": "Our Solution",
            "about_solution_text": "We developed an interactive web app using a Machine Learning model to analyze tabular data, classifying candidates as 'Candidate' or 'False Positive'. Our tool not only classifies but also **explains its decisions**.",
            "about_how_header": "How It Works",
            "about_how_text": "Using data from the Kepler mission (KOI), our RandomForest-based AI model learns to identify patterns that differentiate a real transit from other phenomena. We use **SHAP** to visualize what parameters influenced each prediction.",
            "classify_title": "🔬 Classify an Exoplanet Candidate",
            "classify_intro": "Choose a method: upload a CSV file or enter data manually.",
            "upload_header": "Option 1: CSV File Upload",
            "manual_header": "Option 2: Manual Data Entry",
            "classify_expander_title": "❓ Need help with the CSV format?",
            "classify_expander_text": "The CSV must contain the features the model expects. The system will try to recognize alternative names.",
            "classify_uploader_label": "Select CSV file",
            "classify_download_button": "Download Sample CSV",
            "classify_button": "✨ Classify!",
            "classify_spinner": "Analyzing the cosmos with AI...",
            "classify_results_header": "Analysis Results",
            "classify_verdict_subheader": "Model's Verdict:",
            "classify_planet_candidate": "✔️ **Planet**",
            "classify_false_positive": "❌ **False Positive**",
            "classify_confidence_metric": "Model Confidence",
            "xai_header": "AI Explanation (XAI with SHAP)",
            "results_table_header": "Results for the uploaded file",
            "prediction_col": "Prediction",
            "confidence_col": "Confidence",
            "planet_val": "Planet",
            "false_positive_val": "False Positive",
            "performance_title": "📊 AI Model Performance",
            "performance_intro": "Transparency is key. Here we show how our model performed on unseen data.",
            "performance_tab_metrics": "**Key Metrics**",
            "performance_tab_matrix": "**Feature Importance**",
            "performance_tab_training": "**About the Training**",
            "metric_accuracy": "Overall Accuracy",
            "metric_precision": "Precision",
            "metric_recall": "Recall",
            "metrics_desc": """
            - **Accuracy:** The overall percentage of correct classifications.
            - **Precision:** Of all "planet" predictions, how many were correct?
            - **Recall:** Of all actual planets, how many did our model find?
            """,
            "matrix_header": "Visual Performance Analysis",
            "matrix_text": "The feature importance chart shows the average impact of each feature.",
            "matrix_not_found": "'feature_importance_rf.png' file not found.",
            "matrix_expander_title": "How to read this chart?",
            "matrix_expander_text": "Longer bars represent more influential features for the model.",
            "training_header": "Model Training Details",
            "training_text": """
            - **Dataset Used:** Kepler Objects of Interest (KOI)
            - **Algorithm:** RandomForestClassifier
            - **Features:** The features used are dynamically loaded from the trained model.
            - **Data Split:** 80% for training, 20% for testing.
            """
        }
    }

    # --- Inicialização do Estado da Sessão ---
    if 'lang' not in st.session_state:
        st.session_state.lang = 'en' # Default em Inglês
    if 'page' not in st.session_state:
        st.session_state.page = "nav_about"

    valid_pages = ["nav_about", "nav_classify", "nav_performance"]
    if st.session_state.page not in valid_pages:
        st.session_state.page = "nav_about"
    def t(key):
        return LANGUAGES[st.session_state.lang].get(key, key)

    def switch_language():
        st.session_state.lang = 'pt' if st.session_state.lang == 'en' else 'en'

    # --- Botão de troca de idioma ---
    _, col2 = st.columns([0.8, 0.2])
    with col2:
        if st.button(f"🇧🇷 / 🇺🇸 {t('lang_button')}", use_container_width=True):
            switch_language()
            st.rerun()

    # Este dicionário contém aliases para as colunas que o modelo utiliza.
    COLUMN_ALIASES = {
        'koi_period': ['period'], 
        'koi_duration': ['duration'], 
        'koi_depth': ['depth'],
        'koi_prad': ['prad'], 
        'koi_insol': ['insol'], 
        'koi_model_snr': ['model_snr'],
        'koi_steff': ['steff', 'tce_steff'], 
        'koi_slogg': ['slogg', 'tce_slogg'], 
        'koi_srad': ['srad', 'tce_srad'],
        'koi_impact': ['impact', 'tce_impact']
    }

    def rename_uploaded_columns(df, expected_features):
        rename_map = {}
        df_columns_lower = [str(col).lower() for col in df.columns]
        for standard_name in expected_features:
            alias_list = COLUMN_ALIASES.get(standard_name, [])
            full_alias_list = alias_list + [standard_name]
            for alias in full_alias_list:
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
            model = joblib.load('random_forest_model.joblib')
            scaler = joblib.load('scaler_rf.joblib')
            with open('feature_columns_rf.pkl', 'rb') as f:
                feature_columns = pickle.load(f)
            explainer = shap.TreeExplainer(model)
            return model, scaler, explainer, feature_columns
        except FileNotFoundError:
            return None, None, None, None

    model, scaler, explainer, feature_columns = load_artifacts()

    st.title(f'🔭 {t("page_title")}')

    with st.sidebar:
        if os.path.exists('nasa_logo.png'):
            st.image("nasa_logo.png")
        else:
            st.warning(t("logo_not_found_warning"))
        st.title(t("sidebar_title"))
        st.markdown(f"**{t('sidebar_version')}**")
        nav_options = {"nav_about": t("nav_about"), "nav_classify": t("nav_classify"), "nav_performance": t("nav_performance")}
        nav_keys_list = list(nav_options.keys())
        # Usando o st.session_state.page para garantir que o rádio button reflita a página atual
        page_index = nav_keys_list.index(st.session_state.page) if st.session_state.page in nav_keys_list else 0
        chosen_page = st.radio(label=t("sidebar_nav"), options=nav_keys_list, index=page_index, format_func=lambda key: nav_options[key])
        if chosen_page != st.session_state.page:
            st.session_state.page = chosen_page
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
        if model is None or scaler is None or explainer is None or feature_columns is None:
            st.error(t("model_error"))
        else:
            st.subheader(t("upload_header"))
            with st.expander(t("classify_expander_title")):
                st.write(t("classify_expander_text"))
                sample_data = {col: [np.random.rand()*100] for col in feature_columns}
                sample_df = pd.DataFrame(sample_data)
                csv = sample_df.to_csv(index=False).encode('utf-8')
                st.download_button(label=t("classify_download_button"), data=csv, file_name='exemplo_candidatos_rf.csv', mime='text/csv')
            
            uploaded_file = st.file_uploader(t("classify_uploader_label"), type="csv", label_visibility="collapsed")
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file, comment='#', engine='python')
                    df_renamed = rename_uploaded_columns(df_upload.copy(), feature_columns)
                    missing_cols = list(set(feature_columns) - set(df_renamed.columns))

                    if not missing_cols:
                        df_to_predict = df_renamed[feature_columns]

                        # --- INÍCIO DA CORREÇÃO ---
                        # Verifica se existe algum valor NaN no DataFrame a ser previsto.
                        if df_to_predict.isnull().sum().sum() > 0:
                            # Informa ao usuário que valores ausentes foram encontrados e serão tratados.
                            st.warning(
                                "Atenção: Seu arquivo CSV contém valores ausentes (NaN). "
                                "Para prosseguir com a análise, eles foram preenchidos com a média de suas "
                                "respectivas colunas. Para resultados mais precisos, considere preencher "
                                "esses valores no arquivo original."
                            )
                            # Preenche os valores NaN com a média de cada coluna.
                            # Isso deve ser feito ANTES de passar os dados para o scaler.
                            df_to_predict = df_to_predict.fillna(df_to_predict.mean())
                        # --- FIM DA CORREÇÃO ---
                        
                        # Agora o df_to_predict está limpo, sem NaNs.
                        df_scaled = scaler.transform(df_to_predict)

                        if st.button(t("classify_button"), key="csv_classify", type="primary"):
                            with st.spinner(t("classify_spinner")):
                                predictions = model.predict(df_scaled)
                                probas = model.predict_proba(df_scaled)
                                results_df = pd.DataFrame({
                                    t('prediction_col'): [t('planet_val') if p == 1 else t('false_positive_val') for p in predictions],
                                    t('confidence_col'): [f"{p.max()*100:.2f}%" for p in probas]
                                })
                                # Mantém uma cópia do upload original para exibição
                                df_display = pd.concat([results_df, df_upload.copy()], axis=1)
                                st.subheader(t("results_table_header"))
                                st.dataframe(df_display)
                    else:
                        st.error(f"O arquivo CSV não contém todas as colunas necessárias ou seus sinônimos. Faltando: {', '.join(missing_cols)}")
                except Exception as e:
                    st.error(f"Erro ao processar o arquivo CSV: {e}")
            
            st.divider()
            
            st.subheader(t("manual_header"))
            with st.form(key="manual_form"):
                default_values = {
                    'koi_period': 8.7, 'koi_duration': 2.4, 'koi_depth': 846.0, 
                    'koi_impact': 0.7, 'koi_model_snr': 24.8, 'koi_steff': 5912.0, 
                    'koi_slogg': 4.4, 'koi_srad': 0.9, 'koi_prad': 2.2, 'koi_insol': 163.7
                }
                input_data = {}
                cols = st.columns(3)
                for i, col_name in enumerate(feature_columns):
                    input_data[col_name] = cols[i % 3].number_input(label=col_name, value=default_values.get(col_name, 0.0), format="%.4f")
                submitted = st.form_submit_button(t("classify_button"), type="primary")
                if submitted:
                    with st.spinner(t("classify_spinner")):
                        input_df = pd.DataFrame([input_data])
                        input_scaled = scaler.transform(input_df[feature_columns])
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
                            # Correção robusta para plotagem SHAP
                            if isinstance(shap_values, list):
                                shap_values_to_plot = shap_values[1]
                            else:
                                shap_values_to_plot = shap_values

                            expected_value_raw = explainer.expected_value
                            if isinstance(expected_value_raw, (list, np.ndarray)):
                                expected_value_to_plot = expected_value_raw[1] if len(expected_value_raw) > 1 else expected_value_raw[0]
                            else:
                                expected_value_to_plot = expected_value_raw

                            shap_values_sample = shap_values_to_plot[0] if getattr(shap_values_to_plot, "ndim", 1) > 1 else shap_values_to_plot

                            if isinstance(expected_value_to_plot, (list, np.ndarray)):
                                expected_value_to_plot = float(np.ravel(expected_value_to_plot)[0])

                            force_plot_html = shap.force_plot(
                                expected_value_to_plot,
                                shap_values_sample,
                                input_scaled_df.iloc[0, :],
                                matplotlib=False
                            )
                            st.components.v1.html(force_plot_html.html(), height=300)

    elif st.session_state.page == "nav_performance":
        st.header(t("performance_title"))
        st.write(t("performance_intro"))
        tab1, tab2, tab3 = st.tabs([t("performance_tab_metrics"), t("performance_tab_matrix"), t("performance_tab_training")])
        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric(t("metric_accuracy"), "86.0%") 
            col2.metric(t("metric_precision"), "86.0%")
            col3.metric(t("metric_recall"), "86.0%")
            st.markdown(t("metrics_desc"))
        with tab2:
            st.header(t("matrix_header"))
            st.write(t("matrix_text"))
            if os.path.exists('feature_importance_rf.png'):
                st.image('feature_importance_rf.png')
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
