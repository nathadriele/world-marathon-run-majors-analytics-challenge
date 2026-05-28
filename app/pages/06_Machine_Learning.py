import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_utils import load_css

load_css()

try:
    from src.models.train_model import (
        prepare_regression_data,
        prepare_classification_data,
        train_regression_models,
        train_classification_models,
        train_clustering,
        get_feature_importance,
    )
except ImportError:
    prepare_regression_data = None
    prepare_classification_data = None
    train_regression_models = None
    train_classification_models = None
    train_clustering = None
    get_feature_importance = None

try:
    from src.models.evaluate_model import (
        regression_metrics,
        classification_metrics,
        create_model_comparison_chart,
        plot_confusion_matrix,
        plot_regression_results,
        plot_feature_importance,
        generate_model_report,
    )
except ImportError:
    regression_metrics = None
    classification_metrics = None
    create_model_comparison_chart = None
    plot_confusion_matrix = None
    plot_regression_results = None
    plot_feature_importance = None
    generate_model_report = None

try:
    from src.utils.config import (
        MARATHON_DISTANCE_KM,
        RANDOM_SEED,
        TEST_SIZE,
        CV_FOLDS,
        RACE_NAMES,
        PERFORMANCE_CATEGORIES,
        COLOR_PALETTE,
        PLOTLY_TEMPLATE,
    )
except ImportError:
    MARATHON_DISTANCE_KM = 42.195
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    RACE_NAMES = [
        "Tokyo Marathon",
        "Boston Marathon",
        "London Marathon",
        "Berlin Marathon",
        "Chicago Marathon",
        "New York City Marathon",
    ]
    PERFORMANCE_CATEGORIES = ["Elite", "Advanced", "Intermediate", "Recreational"]
    COLOR_PALETTE = {
        "primary": "#2563EB",
        "secondary": "#14B8A6",
        "accent": "#F97316",
        "success": "#22C55E",
        "warning": "#F59E0B",
    }
    PLOTLY_TEMPLATE = "plotly_white"

try:
    from src.utils.helpers import (
        seconds_to_time,
        calculate_pace_per_km,
        calculate_average_speed_kmh,
        format_pace_str,
    )
except ImportError:
    def seconds_to_time(total_seconds):
        if total_seconds is None or total_seconds <= 0:
            return "00:00:00"
        total_seconds = int(round(total_seconds))
        hours = total_seconds // 3600
        remaining = total_seconds % 3600
        minutes = remaining // 60
        seconds = remaining % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def calculate_pace_per_km(finish_seconds, distance_km=42.195):
        if finish_seconds <= 0 or distance_km <= 0:
            return 0.0
        return (finish_seconds / 60.0) / distance_km

    def calculate_average_speed_kmh(finish_seconds, distance_km=42.195):
        if finish_seconds <= 0:
            return 0.0
        return distance_km / (finish_seconds / 3600.0)

    def format_pace_str(pace_per_km):
        if pace_per_km is None or pace_per_km < 0:
            return "0:00"
        minutes = int(pace_per_km)
        seconds = round((pace_per_km - minutes) * 60)
        if seconds == 60:
            minutes += 1
            seconds = 0
        return f"{minutes}:{seconds:02d}"


REGRESSION_FEATURES = [
    "year",
    "gender_encoded",
    "age",
    "race_encoded",
    "runner_country_encoded",
    "first_half_seconds",
    "negative_split_flag",
    "positive_split_flag",
    "pace_variation",
]

REGRESSION_TARGET = "finish_seconds"

CLASSIFICATION_TARGET = "performance_category_encoded"

FEATURE_DISPLAY_NAMES_PT = {
    "year": "Ano da Corrida",
    "gender_encoded": "Genero (Codificado)",
    "age": "Idade do Corredor",
    "race_encoded": "Corrida (Codificada)",
    "runner_country_encoded": "Pais do Corredor (Codificado)",
    "first_half_seconds": "Tempo da Primeira Metade (segundos)",
    "negative_split_flag": "Flag de Split Negativo",
    "positive_split_flag": "Flag de Split Positivo",
    "pace_variation": "Variacao de Ritmo",
}


@st.cache_data
def prepare_ml_data(results_df):
    df = results_df.copy()
    required_cols = [
        "finish_seconds",
        "gender_encoded",
        "age",
        "race_encoded",
        "runner_country_encoded",
        "first_half_seconds",
        "negative_split_flag",
        "positive_split_flag",
        "pace_variation",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    df = df.dropna(subset=["finish_seconds"])
    df = df[df["finish_seconds"] > 0]
    if "performance_category" in df.columns and "performance_category_encoded" not in df.columns:
        unique_cats = df["performance_category"].dropna().unique().tolist()
        cat_mapping = {val: idx for idx, val in enumerate(unique_cats)}
        df["performance_category_encoded"] = df["performance_category"].map(cat_mapping)
    elif "performance_category_encoded" not in df.columns:
        df["performance_category_encoded"] = 0
    df = df.fillna(0)
    return df


@st.cache_resource
def run_regression_training(_df):
    if prepare_regression_data is None or train_regression_models is None:
        return None
    X_train, X_test, y_train, y_test = prepare_regression_data(_df)
    results = train_regression_models(X_train, y_train, X_test, y_test)
    return results, X_train, X_test, y_train, y_test


@st.cache_resource
def run_classification_training(_df):
    if prepare_classification_data is None or train_classification_models is None:
        return None
    X_train, X_test, y_train, y_test = prepare_classification_data(_df)
    results = train_classification_models(X_train, y_train, X_test, y_test)
    return results, X_train, X_test, y_train, y_test


@st.cache_data
def run_kmeans_clustering(_df, n_clusters):
    if train_clustering is None:
        return None
    model, labels, scaled_data = train_clustering(_df, n_clusters=n_clusters)
    return model, labels, scaled_data


def render_ml_page(results_df):
    st.title("Aprendizado de Maquina")
    st.markdown(
        "Esta pagina apresenta abordagens de modelagem preditiva para analise de desempenho em maratonas. "
        "Tres tarefas de aprendizado de maquina sao exploradas: **Regressao** para prever tempos de chegada, "
        "**Classificacao** para categorizar niveis de desempenho dos corredores e **Agrupamento** para identificar "
        "perfis naturais de corredores. Os modelos sao treinados com features derivadas dos dados de resultados "
        "de maratona, incluindo demografia, tempos parciais e caracteristicas de ritmo."
    )

    with st.expander("Contexto da Competicao", expanded=False):
        st.markdown("##### Declaracao do Problema")
        st.markdown(
            "O objetivo principal de aprendizado de maquina e construir modelos capazes de prever com precisao "
            "os tempos de chegada de maratona e classificar as categorias de desempenho dos corredores com base "
            "nas caracteristicas demograficas e da corrida disponiveis. Alem disso, tecnicas de aprendizado "
            "nao supervisionado sao aplicadas para descobrir agrupamentos naturais dentro da populacao de corredores."
        )
        st.markdown("##### Variaveis Alvo")
        st.markdown(
            "- **Alvo de Regressao**: `finish_seconds` -- o tempo total de chegada em segundos, "
            "permitindo a previsao continua do tempo de conclusao da maratona.\n"
            "- **Alvo de Classificacao**: `performance_category` -- um rotulo categorico "
            "(Elite, Avancado, Intermediario, Recreativo) atribuido com base no tempo de chegada e genero."
        )
        st.markdown("##### Metricas de Avaliacao")
        st.markdown(
            "**Modelos de Regressao**: Erro Absoluto Medio (MAE), Raiz do Erro Quadratico Medio (RMSE), "
            "R-quadrado (R2), Erro Percentual Absoluto Medio (MAPE)\n\n"
            "**Modelos de Classificacao**: Acuracia, Precisao, Revocacao, F1-Score (media ponderada)"
        )
        st.markdown("##### Nota Metodologica")
        st.markdown(
            "A feature `first_half_seconds` e um forte preditor do tempo de chegada, pois o tempo da segunda "
            "metade e diretamente derivado dele. Isso cria um risco de vazamento de dados. Os modelos "
            "apresentados aqui destinam-se a demonstracao analitica e comparacao, e nao a previsao pronta "
            "para producao. Para um modelo de producao, esta feature precisaria ser excluida ou substituida "
            "por features de dados historicos de treinamento."
        )

    task_selection = st.radio(
        "Selecionar Tarefa de Aprendizado de Maquina",
        options=["Regressao", "Classificacao", "Agrupamento"],
        horizontal=True,
        key="ml_task_selection",
    )

    st.divider()

    if task_selection == "Regressao":
        render_regression_section(results_df)
    elif task_selection == "Classificacao":
        render_classification_section(results_df)
    elif task_selection == "Agrupamento":
        render_clustering_section(results_df)

    st.divider()
    render_model_leaderboard()


def render_regression_section(results_df):
    st.subheader("Previsao do Tempo de Chegada")
    st.markdown(
        "Modelos de regressao preveem o tempo de chegada continuo em segundos para cada corredor. "
        "As features incluem demografia do corredor, identificador da corrida, tempo parcial de "
        "meia-maratona, flags de direcao de split e metricas de variacao de ritmo."
    )

    df = prepare_ml_data(results_df)

    with st.expander("Detalhes da Preparacao dos Dados", expanded=False):
        st.markdown("##### Features Utilizadas")
        feature_df_data = []
        for feat in REGRESSION_FEATURES:
            feature_df_data.append({
                "Feature": feat,
                "Descricao": FEATURE_DISPLAY_NAMES_PT.get(feat, feat),
                "Tipo": "Numerica",
            })
        st.dataframe(pd.DataFrame(feature_df_data), use_container_width=True, hide_index=True)

        st.markdown("##### Variavel Alvo")
        st.markdown(f"**`{REGRESSION_TARGET}`** -- Tempo total de chegada em segundos.")

        st.markdown("##### Divisao Treino / Teste")
        st.markdown(
            f"O conjunto de dados e dividido em **{int((1 - TEST_SIZE) * 100)}% treino** e "
            f"**{int(TEST_SIZE * 100)}% teste** com semente aleatoria de **{RANDOM_SEED}**. "
            f"Total de registros disponiveis: **{len(df):,}**."
        )

    if st.button("Treinar Modelos de Regressao", key="train_regression_btn"):
        with st.spinner("Treinando modelos de regressao... Isso pode levar um momento."):
            training_output = run_regression_training(df)
        if training_output is None:
            st.error("Funcoes de treinamento de regressao nao estao disponiveis. Verifique se scikit-learn esta instalado.")
            return
        reg_results, X_train, X_test, y_train, y_test = training_output
        st.session_state["reg_results"] = reg_results
        st.session_state["reg_X_test"] = X_test
        st.session_state["reg_y_test"] = y_test
        st.session_state["reg_trained"] = True
        st.success("Todos os modelos de regressao foram treinados com sucesso.")

    if st.session_state.get("reg_trained", False) and st.session_state.get("reg_results") is not None:
        reg_results = st.session_state["reg_results"]
        X_test = st.session_state.get("reg_X_test")
        y_test = st.session_state.get("reg_y_test")

        st.markdown("#### Comparacao dos Modelos")
        comparison_rows = []
        for model_name, model_data in reg_results.items():
            comparison_rows.append({
                "Modelo": model_name,
                "MAE (segundos)": round(model_data["mae"], 2),
                "RMSE (segundos)": round(model_data["rmse"], 2),
                "R2 Score": round(model_data["r2"], 4),
                "MAPE (%)": round(model_data["mape"] * 100, 2),
            })
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df = comparison_df.sort_values("R2 Score", ascending=False).reset_index(drop=True)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        metrics_to_plot = ["MAE (segundos)", "RMSE (segundos)", "R2 Score"]
        selected_metric_display = st.selectbox(
            "Selecionar Metrica para Grafico de Comparacao",
            options=metrics_to_plot,
            index=2,
            key="reg_metric_select",
        )
        metric_key_map = {
            "MAE (segundos)": "mae",
            "RMSE (segundos)": "rmse",
            "R2 Score": "r2",
        }
        selected_metric_key = metric_key_map[selected_metric_display]

        model_names = list(reg_results.keys())
        metric_values = [reg_results[m].get(selected_metric_key, 0) for m in model_names]

        fig_comparison = go.Figure()
        fig_comparison.add_trace(
            go.Bar(
                x=model_names,
                y=metric_values,
                marker_color=[COLOR_PALETTE.get("primary", "#2563EB")] * len(model_names),
                text=[f"{v:.4f}" for v in metric_values],
                textposition="outside",
            )
        )
        fig_comparison.update_layout(
            title=f"Comparacao dos Modelos -- {selected_metric_display}",
            xaxis_title="Modelo",
            yaxis_title=selected_metric_display,
            template=PLOTLY_TEMPLATE,
            height=450,
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

        best_model_name = comparison_df.iloc[0]["Modelo"]
        best_r2 = comparison_df.iloc[0]["R2 Score"]
        best_mae = comparison_df.iloc[0]["MAE (segundos)"]
        st.success(
            f"Melhor Modelo: **{best_model_name}** -- "
            f"R2: {best_r2:.4f}, MAE: {best_mae:.2f} segundos "
            f"({seconds_to_time(best_mae)})"
        )

        st.markdown("#### Importancia das Features")
        tree_models = ["RandomForestRegressor", "GradientBoostingRegressor"]
        best_tree = None
        for tm in tree_models:
            if tm in reg_results:
                best_tree = tm
                break
        if best_tree is not None:
            model_obj = reg_results[best_tree]["model"]
            if get_feature_importance is not None:
                importance_df = get_feature_importance(model_obj, REGRESSION_FEATURES)
            else:
                if hasattr(model_obj, "feature_importances_"):
                    importances = model_obj.feature_importances_
                elif hasattr(model_obj, "coef_"):
                    importances = np.abs(model_obj.coef_)
                    if importances.ndim > 1:
                        importances = importances.mean(axis=0)
                else:
                    importances = np.zeros(len(REGRESSION_FEATURES))
                importance_df = pd.DataFrame({
                    "feature": REGRESSION_FEATURES,
                    "importance": importances,
                })
                importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

            display_importance = importance_df.copy()
            display_importance["feature"] = display_importance["feature"].map(
                lambda f: FEATURE_DISPLAY_NAMES_PT.get(f, f)
            )

            fig_importance = go.Figure()
            sorted_imp = display_importance.sort_values("importance", ascending=True)
            fig_importance.add_trace(
                go.Bar(
                    x=sorted_imp["importance"],
                    y=sorted_imp["feature"],
                    orientation="h",
                    marker_color=sorted_imp["importance"],
                    marker_colorscale="Viridis",
                    text=sorted_imp["importance"].apply(lambda x: f"{x:.4f}"),
                    textposition="outside",
                )
            )
            fig_importance.update_layout(
                title=f"Importancia das Features -- {best_tree}",
                xaxis_title="Importancia",
                yaxis_title="Feature",
                template=PLOTLY_TEMPLATE,
                height=500,
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Nenhum modelo baseado em arvores disponivel para extracao de importancia das features.")

        st.markdown("#### Previsoes vs Real")
        pred_model_name = st.selectbox(
            "Selecionar Modelo para Grafico de Previsao",
            options=list(reg_results.keys()),
            index=0,
            key="reg_pred_model_select",
        )
        preds = reg_results[pred_model_name]["predictions"]
        if y_test is not None and preds is not None:
            y_test_arr = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
            preds_arr = np.array(preds) if not isinstance(preds, np.ndarray) else preds
            fig_scatter = go.Figure()
            fig_scatter.add_trace(
                go.Scatter(
                    x=y_test_arr,
                    y=preds_arr,
                    mode="markers",
                    name="Previsoes",
                    marker=dict(color=COLOR_PALETTE.get("primary", "#2563EB"), opacity=0.4, size=5),
                )
            )
            min_val = min(y_test_arr.min(), preds_arr.min())
            max_val = max(y_test_arr.max(), preds_arr.max())
            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Previsao Perfeita",
                    line=dict(color="red", dash="dash"),
                )
            )
            fig_scatter.update_layout(
                title=f"Real vs Previsto -- {pred_model_name}",
                xaxis_title="Tempo de Chegada Real (segundos)",
                yaxis_title="Tempo de Chegada Previsto (segundos)",
                template=PLOTLY_TEMPLATE,
                height=550,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### Previsao Interativa")
        st.markdown(
            "Insira os dados do corredor abaixo para prever o tempo de chegada usando o melhor modelo treinado."
        )

        pred_col1, pred_col2, pred_col3 = st.columns(3)
        with pred_col1:
            input_age = st.number_input(
                "Idade",
                min_value=18,
                max_value=80,
                value=35,
                step=1,
                key="ml_pred_age",
            )
            input_gender = st.selectbox(
                "Genero",
                options=["M", "F"],
                index=0,
                key="ml_pred_gender",
            )
            input_gender_encoded = 0 if input_gender == "M" else 1
        with pred_col2:
            input_race = st.selectbox(
                "Corrida",
                options=RACE_NAMES,
                index=0,
                key="ml_pred_race",
            )
            input_race_encoded = RACE_NAMES.index(input_race)
            input_year = st.number_input(
                "Ano",
                min_value=2018,
                max_value=2025,
                value=2024,
                step=1,
                key="ml_pred_year",
            )
        with pred_col3:
            input_first_half_min = st.number_input(
                "Tempo da Primeira Metade (minutos)",
                min_value=30.0,
                max_value=180.0,
                value=105.0,
                step=0.5,
                key="ml_pred_first_half_min",
            )
            input_first_half_seconds = int(input_first_half_min * 60)
            input_country_encoded = st.number_input(
                "Codigo do Pais (codificado)",
                min_value=0,
                max_value=200,
                value=0,
                step=1,
                key="ml_pred_country",
            )

        input_negative_split = st.checkbox("Split Negativo", value=False, key="ml_pred_neg_split")
        input_positive_split = not input_negative_split
        input_pace_variation = st.number_input(
            "Variacao de Ritmo",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.1,
            key="ml_pred_pace_var",
        )

        runner_input = {
            "year": input_year,
            "gender_encoded": input_gender_encoded,
            "age": input_age,
            "race_encoded": input_race_encoded,
            "runner_country_encoded": input_country_encoded,
            "first_half_seconds": input_first_half_seconds,
            "negative_split_flag": int(input_negative_split),
            "positive_split_flag": int(input_positive_split),
            "pace_variation": input_pace_variation,
        }

        if st.button("Prever Tempo de Chegada", key="predict_finish_btn"):
            best_model_obj = reg_results[best_model_name]["model"]
            input_df = pd.DataFrame([runner_input])
            for col in REGRESSION_FEATURES:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[REGRESSION_FEATURES]
            predicted_seconds = float(best_model_obj.predict(input_df)[0])
            predicted_time_str = seconds_to_time(predicted_seconds)
            predicted_pace = calculate_pace_per_km(predicted_seconds)
            predicted_pace_str = format_pace_str(predicted_pace)
            predicted_speed = calculate_average_speed_kmh(predicted_seconds)

            result_col1, result_col2, result_col3 = st.columns(3)
            with result_col1:
                st.metric("Tempo de Chegada Previsto", predicted_time_str)
            with result_col2:
                st.metric("Ritmo Previsto", f"{predicted_pace_str} /km")
            with result_col3:
                st.metric("Velocidade Prevista", f"{predicted_speed:.2f} km/h")


def render_classification_section(results_df):
    st.subheader("Classificacao de Categoria de Desempenho")
    st.markdown(
        "Modelos de classificacao preveem a categoria de desempenho do corredor com base em features "
        "demograficas e da corrida. As categorias sao: **Elite**, **Avancado**, **Intermediario** e **Recreativo**, "
        "determinadas por limiares de tempo de chegada que variam por genero."
    )

    df = prepare_ml_data(results_df)

    if "performance_category" in df.columns:
        st.markdown("##### Distribuicao das Categorias")
        cat_counts = df["performance_category"].value_counts().reset_index()
        cat_counts.columns = ["Categoria", "Quantidade"]
        fig_cat = px.bar(
            cat_counts,
            x="Categoria",
            y="Quantidade",
            color="Categoria",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Distribuicao das Categorias de Desempenho dos Corredores",
        )
        fig_cat.update_layout(template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig_cat, use_container_width=True)

        total_runners = len(df)
        dist_rows = []
        for _, row in cat_counts.iterrows():
            pct = (row["Quantidade"] / total_runners) * 100 if total_runners > 0 else 0
            dist_rows.append({
                "Categoria": row["Categoria"],
                "Quantidade": row["Quantidade"],
                "Percentual": f"{pct:.1f}%",
            })
        st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Coluna de categoria de desempenho nao encontrada no conjunto de dados.")

    if st.button("Treinar Modelos de Classificacao", key="train_classification_btn"):
        with st.spinner("Treinando modelos de classificacao... Isso pode levar um momento."):
            training_output = run_classification_training(df)
        if training_output is None:
            st.error("Funcoes de treinamento de classificacao nao estao disponiveis. Verifique se scikit-learn esta instalado.")
            return
        clf_results, X_train, X_test, y_train, y_test = training_output
        st.session_state["clf_results"] = clf_results
        st.session_state["clf_X_test"] = X_test
        st.session_state["clf_y_test"] = y_test
        st.session_state["clf_trained"] = True
        st.success("Todos os modelos de classificacao foram treinados com sucesso.")

    if st.session_state.get("clf_trained", False) and st.session_state.get("clf_results") is not None:
        clf_results = st.session_state["clf_results"]
        X_test = st.session_state.get("clf_X_test")
        y_test = st.session_state.get("clf_y_test")

        st.markdown("#### Comparacao dos Modelos")
        clf_comparison_rows = []
        for model_name, model_data in clf_results.items():
            clf_comparison_rows.append({
                "Modelo": model_name,
                "Acuracia": round(model_data.get("accuracy", 0), 4),
                "Precisao": round(model_data.get("precision", 0), 4),
                "Revocacao": round(model_data.get("recall", 0), 4),
                "F1 Score": round(model_data.get("f1", 0), 4),
            })
        clf_comparison_df = pd.DataFrame(clf_comparison_rows)
        clf_comparison_df = clf_comparison_df.sort_values("F1 Score", ascending=False).reset_index(drop=True)
        st.dataframe(clf_comparison_df, use_container_width=True, hide_index=True)

        clf_metrics = ["Acuracia", "Precisao", "Revocacao", "F1 Score"]
        selected_clf_metric = st.selectbox(
            "Selecionar Metrica para Grafico de Comparacao",
            options=clf_metrics,
            index=3,
            key="clf_metric_select",
        )
        clf_metric_key_map = {
            "Acuracia": "accuracy",
            "Precisao": "precision",
            "Revocacao": "recall",
            "F1 Score": "f1",
        }
        selected_clf_metric_key = clf_metric_key_map[selected_clf_metric]

        clf_model_names = list(clf_results.keys())
        clf_metric_values = [clf_results[m].get(selected_clf_metric_key, 0) for m in clf_model_names]

        fig_clf_comp = go.Figure()
        fig_clf_comp.add_trace(
            go.Bar(
                x=clf_model_names,
                y=clf_metric_values,
                marker_color=[COLOR_PALETTE.get("secondary", "#14B8A6")] * len(clf_model_names),
                text=[f"{v:.4f}" for v in clf_metric_values],
                textposition="outside",
            )
        )
        fig_clf_comp.update_layout(
            title=f"Comparacao dos Modelos de Classificacao -- {selected_clf_metric}",
            xaxis_title="Modelo",
            yaxis_title=selected_clf_metric,
            template=PLOTLY_TEMPLATE,
            height=450,
        )
        st.plotly_chart(fig_clf_comp, use_container_width=True)

        best_clf_name = clf_comparison_df.iloc[0]["Modelo"]
        best_clf_f1 = clf_comparison_df.iloc[0]["F1 Score"]
        best_clf_acc = clf_comparison_df.iloc[0]["Acuracia"]
        st.success(
            f"Melhor Modelo de Classificacao: **{best_clf_name}** -- "
            f"Acuracia: {best_clf_acc:.4f}, F1 Score: {best_clf_f1:.4f}"
        )

        st.markdown("#### Matriz de Confusao")
        selected_cm_model = st.selectbox(
            "Selecionar Modelo para Matriz de Confusao",
            options=list(clf_results.keys()),
            index=0,
            key="clf_cm_model_select",
        )
        cm_preds = clf_results[selected_cm_model]["predictions"]
        if y_test is not None and cm_preds is not None:
            y_test_arr = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
            cm_preds_arr = np.array(cm_preds) if not isinstance(cm_preds, np.ndarray) else cm_preds
            unique_labels = sorted(list(set(y_test_arr.tolist()) | set(cm_preds_arr.tolist())))
            if "performance_category" in df.columns:
                unique_cats = df["performance_category"].dropna().unique().tolist()
                label_names = []
                for lbl in unique_labels:
                    if lbl < len(unique_cats):
                        label_names.append(unique_cats[lbl])
                    else:
                        label_names.append(str(lbl))
            else:
                label_names = [str(lbl) for lbl in unique_labels]

            if plot_confusion_matrix is not None:
                fig_cm = plot_confusion_matrix(y_test_arr, cm_preds_arr, label_names)
            else:
                from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
                cm_array = sklearn_confusion_matrix(y_test_arr, cm_preds_arr)
                fig_cm = go.Figure()
                fig_cm.add_trace(
                    go.Heatmap(
                        z=cm_array,
                        x=label_names,
                        y=label_names,
                        colorscale="Blues",
                        text=cm_array,
                        texttemplate="%{text}",
                        textfont={"size": 12},
                        showscale=True,
                    )
                )
                fig_cm.update_layout(
                    title="Matriz de Confusao",
                    xaxis_title="Rotulo Previsto",
                    yaxis_title="Rotulo Real",
                )
            fig_cm.update_layout(template=PLOTLY_TEMPLATE, height=500)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("#### Relatorio de Classificacao")
        from sklearn.metrics import classification_report as sklearn_classification_report
        report_clf_model = st.selectbox(
            "Selecionar Modelo para Relatorio de Classificacao",
            options=list(clf_results.keys()),
            index=0,
            key="clf_report_model_select",
        )
        report_preds = clf_results[report_clf_model]["predictions"]
        if y_test is not None and report_preds is not None:
            y_test_arr2 = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
            report_preds_arr = np.array(report_preds) if not isinstance(report_preds, np.ndarray) else report_preds
            unique_labels2 = sorted(list(set(y_test_arr2.tolist()) | set(report_preds_arr.tolist())))
            if "performance_category" in df.columns:
                unique_cats2 = df["performance_category"].dropna().unique().tolist()
                target_names2 = []
                for lbl in unique_labels2:
                    if lbl < len(unique_cats2):
                        target_names2.append(unique_cats2[lbl])
                    else:
                        target_names2.append(str(lbl))
            else:
                target_names2 = [str(lbl) for lbl in unique_labels2]
            report_text = sklearn_classification_report(
                y_test_arr2, report_preds_arr, target_names=target_names2, zero_division=0
            )
            st.text(report_text)


def render_clustering_section(results_df):
    st.subheader("Agrupamento de Perfis de Corredores")
    st.markdown(
        "Agrupamento nao supervisionado usando KMeans agrupa corredores em perfis distintos com base no "
        "tempo de chegada, ritmo, idade e velocidade. Esta analise revela padroes naturais na populacao "
        "de corredores sem usar rotulos predefinidos."
    )

    df = prepare_ml_data(results_df)

    n_clusters = st.slider(
        "Selecionar Numero de Clusters",
        min_value=2,
        max_value=6,
        value=4,
        step=1,
        key="cluster_count_slider",
    )

    if st.button("Executar Agrupamento KMeans", key="run_clustering_btn"):
        with st.spinner(f"Executando KMeans com {n_clusters} clusters..."):
            cluster_output = run_kmeans_clustering(df, n_clusters)
        if cluster_output is None:
            st.error("Funcoes de agrupamento nao estao disponiveis. Verifique se scikit-learn esta instalado.")
            return
        cluster_model, cluster_labels, scaled_data = cluster_output
        df_clustered = df.copy()
        df_clustered["cluster"] = cluster_labels
        st.session_state["cluster_df"] = df_clustered
        st.session_state["cluster_n"] = n_clusters
        st.session_state["cluster_trained"] = True
        st.success(f"Agrupamento KMeans concluido com {n_clusters} clusters.")

    if st.session_state.get("cluster_trained", False) and st.session_state.get("cluster_df") is not None:
        df_clustered = st.session_state["cluster_df"]
        n_cl = st.session_state.get("cluster_n", n_clusters)

        st.markdown("#### Estatisticas dos Clusters")
        cluster_stats_rows = []
        for cluster_id in range(n_cl):
            cluster_data = df_clustered[df_clustered["cluster"] == cluster_id]
            avg_finish = cluster_data["finish_seconds"].mean() if "finish_seconds" in cluster_data.columns else 0
            avg_pace = calculate_pace_per_km(avg_finish) if avg_finish > 0 else 0
            avg_age = cluster_data["age"].mean() if "age" in cluster_data.columns else 0
            avg_speed = calculate_average_speed_kmh(avg_finish) if avg_finish > 0 else 0
            count = len(cluster_data)
            pct = (count / len(df_clustered)) * 100 if len(df_clustered) > 0 else 0
            cluster_stats_rows.append({
                "Cluster": f"Cluster {cluster_id}",
                "Quantidade": count,
                "Percentual": f"{pct:.1f}%",
                "Tempo Medio de Chegada": seconds_to_time(avg_finish),
                "Ritmo Medio (/km)": format_pace_str(avg_pace),
                "Idade Media": f"{avg_age:.1f}",
                "Velocidade Media (km/h)": f"{avg_speed:.2f}",
            })
        cluster_stats_df = pd.DataFrame(cluster_stats_rows)
        st.dataframe(cluster_stats_df, use_container_width=True, hide_index=True)

        st.markdown("#### Visualizacao dos Clusters")
        scatter_x_col = "finish_seconds"
        scatter_y_col = "pace_per_km" if "pace_per_km" in df_clustered.columns else "finish_seconds"
        if scatter_y_col not in df_clustered.columns:
            scatter_y_col = "finish_seconds"

        fig_clusters = px.scatter(
            df_clustered,
            x=scatter_x_col,
            y=scatter_y_col,
            color="cluster",
            title="Clusters de Corredores -- Tempo de Chegada vs Ritmo",
            labels={
                scatter_x_col: "Tempo de Chegada (segundos)",
                scatter_y_col: "Ritmo (min/km)" if scatter_y_col == "pace_per_km" else scatter_y_col,
                "cluster": "Cluster",
            },
            color_continuous_scale="Viridis",
            opacity=0.6,
        )
        fig_clusters.update_layout(
            template=PLOTLY_TEMPLATE,
            height=550,
        )
        st.plotly_chart(fig_clusters, use_container_width=True)

        st.markdown("#### Descricoes dos Perfis dos Clusters")
        for cluster_id in range(n_cl):
            cluster_data = df_clustered[df_clustered["cluster"] == cluster_id]
            avg_finish = cluster_data["finish_seconds"].mean() if "finish_seconds" in cluster_data.columns else 0
            avg_pace = calculate_pace_per_km(avg_finish) if avg_finish > 0 else 0
            avg_age = cluster_data["age"].mean() if "age" in cluster_data.columns else 0
            count = len(cluster_data)
            pct = (count / len(df_clustered)) * 100 if len(df_clustered) > 0 else 0

            if avg_finish < 9000:
                profile_label = "Corredores Rapidos"
            elif avg_finish < 12600:
                profile_label = "Corredores Intermediarios"
            elif avg_finish < 18000:
                profile_label = "Corredores Recreativos"
            else:
                profile_label = "Participantes Caminhada/Corrida"

            with st.expander(f"Cluster {cluster_id} -- {profile_label} ({count} corredores, {pct:.1f}%)"):
                st.markdown(
                    f"**Tempo Medio de Chegada**: {seconds_to_time(avg_finish)}\n\n"
                    f"**Ritmo Medio**: {format_pace_str(avg_pace)} /km\n\n"
                    f"**Idade Media**: {avg_age:.1f} anos\n\n"
                    f"**Tamanho do Cluster**: {count} corredores ({pct:.1f}% do total)"
                )


def render_model_leaderboard():
    st.markdown("### Tabela de Classificacao dos Modelos")
    st.markdown(
        "Resumo combinado de todos os modelos treinados e suas respectivas metricas de desempenho."
    )

    leaderboard_rows = []
    if st.session_state.get("reg_trained", False) and st.session_state.get("reg_results") is not None:
        reg_results = st.session_state["reg_results"]
        for model_name, model_data in reg_results.items():
            leaderboard_rows.append({
                "Tarefa": "Regressao",
                "Modelo": model_name,
                "Metrica Principal": f"R2 = {model_data['r2']:.4f}",
                "MAE": f"{model_data['mae']:.2f}",
                "RMSE": f"{model_data['rmse']:.2f}",
                "R2": f"{model_data['r2']:.4f}",
                "Acuracia": "--",
                "F1 Score": "--",
            })

    if st.session_state.get("clf_trained", False) and st.session_state.get("clf_results") is not None:
        clf_results = st.session_state["clf_results"]
        for model_name, model_data in clf_results.items():
            leaderboard_rows.append({
                "Tarefa": "Classificacao",
                "Modelo": model_name,
                "Metrica Principal": f"F1 = {model_data.get('f1', 0):.4f}",
                "MAE": "--",
                "RMSE": "--",
                "R2": "--",
                "Acuracia": f"{model_data.get('accuracy', 0):.4f}",
                "F1 Score": f"{model_data.get('f1', 0):.4f}",
            })

    if st.session_state.get("cluster_trained", False):
        leaderboard_rows.append({
            "Tarefa": "Agrupamento",
            "Modelo": f"KMeans (k={st.session_state.get('cluster_n', 4)})",
            "Metrica Principal": "Nao Supervisionado",
            "MAE": "--",
            "RMSE": "--",
            "R2": "--",
            "Acuracia": "--",
            "F1 Score": "--",
        })

    if len(leaderboard_rows) > 0:
        leaderboard_df = pd.DataFrame(leaderboard_rows)
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

        best_reg_row = None
        best_clf_row = None
        for row in leaderboard_rows:
            if row["Tarefa"] == "Regressao" and (best_reg_row is None or float(row["R2"]) > float(best_reg_row["R2"])):
                best_reg_row = row
            if row["Tarefa"] == "Classificacao" and (best_clf_row is None or float(row["F1 Score"]) > float(best_clf_row["F1 Score"])):
                best_clf_row = row

        highlight_col1, highlight_col2 = st.columns(2)
        with highlight_col1:
            if best_reg_row is not None:
                st.success(
                    f"Melhor Modelo de Regressao: **{best_reg_row['Modelo']}** -- {best_reg_row['Metrica Principal']}"
                )
        with highlight_col2:
            if best_clf_row is not None:
                st.success(
                    f"Melhor Modelo de Classificacao: **{best_clf_row['Modelo']}** -- {best_clf_row['Metrica Principal']}"
                )
    else:
        st.info("Nenhum modelo foi treinado ainda. Use os botoes acima para treinar modelos e popular a tabela de classificacao.")


if __name__ == "__main__" or "streamlit" in os.path.basename(sys.argv[0]).lower():
    results_df = None
    data_loaded = False

    if "marathon_results" in st.session_state and st.session_state["marathon_results"] is not None:
        results_df = st.session_state["marathon_results"]
        data_loaded = True
    elif "results" in st.session_state and st.session_state["results"] is not None:
        results_df = st.session_state["results"]
        data_loaded = True
    elif "combined_data" in st.session_state and st.session_state["combined_data"] is not None:
        results_df = st.session_state["combined_data"]
        data_loaded = True

    if not data_loaded:
        try:
            from src.data.load_data import load_csv, get_data_filepaths
            processed_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
            raw_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
            combined_path = os.path.join(processed_dir, "combined_marathon_data.csv")
            results_path = os.path.join(raw_dir, "marathon_results.csv")
            if os.path.exists(combined_path):
                results_df = pd.read_csv(combined_path)
                data_loaded = True
            elif os.path.exists(results_path):
                results_df = pd.read_csv(results_path)
                data_loaded = True
        except Exception:
            pass

    if data_loaded and results_df is not None:
        render_ml_page(results_df)
    else:
        st.warning(
            "Nenhum dado carregado. Certifique-se de que os conjuntos de dados de maratona estao "
            "disponiveis no diretorio de dados ou carregue-os atraves do ponto de entrada principal "
            "do aplicativo para que sejam armazenados no estado da sessao."
        )
