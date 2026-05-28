import os

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Marathon Majors - Painel Analitico",
    page_icon=":chart:",
    layout="wide",
)


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    try:
        with open(css_path, "r") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


def rename_nav_item():
    st.markdown("""
    <style>
    /* Hide the first nav item text (streamlit_app / Streamlit app) */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > ul > li:first-child a {
        font-size: 0 !important;
        line-height: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > ul > li:first-child a * {
        font-size: 0 !important;
        visibility: hidden !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > ul > li:first-child a::after {
        content: 'Painel';
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        visibility: visible !important;
        color: #0F172A;
        font-family: 'Inter', sans-serif;
        line-height: 1.5 !important;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, "..", "data", "raw")
    processed_dir = os.path.join(base_dir, "..", "data", "processed")

    datasets = {
        "marathon_results": os.path.join(raw_dir, "marathon_results.csv"),
        "winners": os.path.join(raw_dir, "winners_data.csv"),
        "race_metadata": os.path.join(raw_dir, "race_metadata.csv"),
        "brazilian_runners": os.path.join(processed_dir, "brazilian_runners_analysis.csv"),
        "pace_splits": os.path.join(processed_dir, "pace_splits_analysis.csv"),
    }

    data = {}
    for key, filepath in datasets.items():
        try:
            data[key] = pd.read_csv(filepath)
        except FileNotFoundError:
            data[key] = None

    return data


def get_unique_values(df, column):
    if df is None or df.empty or column not in df.columns:
        return []
    return sorted(df[column].dropna().unique().tolist())


def get_age_groups(df):
    if df is None or df.empty or "age" not in df.columns:
        return []
    age_min = int(df["age"].min())
    age_max = int(df["age"].max())
    bins = list(range(age_min - (age_min % 10), age_max + 10, 10))
    groups = []
    for i in range(len(bins) - 1):
        groups.append(f"{bins[i]}-{bins[i + 1] - 1}")
    return groups


def get_performance_categories():
    return ["Elite", "Competitivo", "Recreativo", "Fun Run"]


def format_seconds_to_hms(seconds):
    if pd.isna(seconds):
        return "--:--:--"
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_sidebar_filters(df):
    with st.sidebar:
        st.markdown("## Marathon Majors")
        st.markdown("---")

        if df is None or df.empty:
            st.markdown(
                "<p style='color: #64748B; font-size: 0.9rem;'>"
                "Nenhum dado disponivel para filtros."
                "</p>",
                unsafe_allow_html=True,
            )
            return {
                "years": [],
                "races": [],
                "genders": [],
                "age_groups": [],
                "performance_categories": [],
                "brazilian_only": False,
                "finish_time_range": (0, 21000),
                "pace_range": (150.0, 550.0),
            }

        years = get_unique_values(df, "year")
        races = get_unique_values(df, "marathon")
        genders = get_unique_values(df, "gender")
        age_groups = get_age_groups(df)
        perf_categories = get_performance_categories()

        with st.expander("Filtros", expanded=False):
            selected_years = st.multiselect(
                "Ano",
                options=years,
                default=years,
            )

            selected_races = st.multiselect(
                "Corrida",
                options=races,
                default=races,
            )

            selected_genders = st.multiselect(
                "Genero",
                options=genders,
                default=genders,
            )

            selected_age_groups = st.multiselect(
                "Faixa Etaria",
                options=age_groups,
                default=age_groups,
            )

            selected_perf = st.multiselect(
                "Categoria de Desempenho",
                options=perf_categories,
                default=perf_categories,
            )

            brazilian_only = st.checkbox(
                "Apenas Corredores Brasileiros",
                value=False,
            )

            st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

            if "finish_time_sec" in df.columns:
                finish_min = int(df["finish_time_sec"].min())
                finish_max = int(df["finish_time_sec"].max())
            else:
                finish_min = 0
                finish_max = 21000

            finish_range = st.slider(
                "Faixa de Tempo de Chegada (segundos)",
                min_value=finish_min,
                max_value=finish_max,
                value=(finish_min, finish_max),
            )

            if "finish_time_sec" in df.columns:
                pace_values = df["finish_time_sec"].dropna() / 42.195
                pace_min_val = float(pace_values.min())
                pace_max_val = float(pace_values.max())
            else:
                pace_min_val = 150.0
                pace_max_val = 550.0

            pace_range = st.slider(
                "Faixa de Pace (seg/km)",
                min_value=round(pace_min_val, 1),
                max_value=round(pace_max_val, 1),
                value=(round(pace_min_val, 1), round(pace_max_val, 1)),
            )

        st.markdown("---")
        st.markdown(
            "<p style='color: #94A3B8; font-size: 0.75rem; text-align: center;'>"
            "World Marathon Majors Analytics"
            "</p>",
            unsafe_allow_html=True,
        )

    return {
        "years": selected_years,
        "races": selected_races,
        "genders": selected_genders,
        "age_groups": selected_age_groups,
        "performance_categories": selected_perf,
        "brazilian_only": brazilian_only,
        "finish_time_range": finish_range,
        "pace_range": pace_range,
    }


def apply_filters(df, filters):
    if df is None or df.empty:
        return df

    filtered = df.copy()

    if "year" in filtered.columns and filters.get("years"):
        filtered = filtered[filtered["year"].isin(filters["years"])]

    if "marathon" in filtered.columns and filters.get("races"):
        filtered = filtered[filtered["marathon"].isin(filters["races"])]

    if "gender" in filtered.columns and filters.get("genders"):
        filtered = filtered[filtered["gender"].isin(filters["genders"])]

    if "age" in filtered.columns and filters.get("age_groups"):
        age_groups = filters["age_groups"]
        if age_groups:
            masks = []
            for group in age_groups:
                try:
                    parts = group.split("-")
                    low = int(parts[0])
                    high = int(parts[1])
                    masks.append((filtered["age"] >= low) & (filtered["age"] <= high))
                except (ValueError, IndexError):
                    continue
            if masks:
                combined_mask = masks[0]
                for m in masks[1:]:
                    combined_mask = combined_mask | m
                filtered = filtered[combined_mask]

    if filters.get("brazilian_only"):
        country_col = None
        for col in ["country", "country_x"]:
            if col in filtered.columns:
                country_col = col
                break
        if country_col:
            filtered = filtered[filtered[country_col] == "BRA"]

    if "finish_time_sec" in filtered.columns and filters.get("finish_time_range"):
        low, high = filters["finish_time_range"]
        filtered = filtered[
            (filtered["finish_time_sec"] >= low) & (filtered["finish_time_sec"] <= high)
        ]

    if "finish_time_sec" in filtered.columns and filters.get("pace_range"):
        pace_low, pace_high = filters["pace_range"]
        pace_per_km = filtered["finish_time_sec"] / 42.195
        filtered = filtered[(pace_per_km >= pace_low) & (pace_per_km <= pace_high)]

    if "finish_time_sec" in filtered.columns and filters.get("performance_categories"):
        perf = filters["performance_categories"]
        if perf:
            finish_times = filtered["finish_time_sec"].dropna()
            if not finish_times.empty:
                q25 = finish_times.quantile(0.25)
                q50 = finish_times.quantile(0.50)
                q75 = finish_times.quantile(0.75)
                perf_masks = []
                for category in perf:
                    if category == "Elite":
                        perf_masks.append(filtered["finish_time_sec"] <= q25)
                    elif category == "Competitivo":
                        perf_masks.append(
                            (filtered["finish_time_sec"] > q25)
                            & (filtered["finish_time_sec"] <= q50)
                        )
                    elif category == "Recreativo":
                        perf_masks.append(
                            (filtered["finish_time_sec"] > q50)
                            & (filtered["finish_time_sec"] <= q75)
                        )
                    elif category == "Fun Run":
                        perf_masks.append(filtered["finish_time_sec"] > q75)
                if perf_masks:
                    combined_mask = perf_masks[0]
                    for m in perf_masks[1:]:
                        combined_mask = combined_mask | m
                    filtered = filtered[combined_mask]

    return filtered


def render_kpi_metrics(df, winners_df):
    total_races = 0
    if df is not None and "marathon" in df.columns:
        total_races = df["marathon"].nunique()

    years_analyzed = 0
    if df is not None and "year" in df.columns:
        years_analyzed = df["year"].nunique()

    total_finishers = 0
    if df is not None:
        total_finishers = len(df)

    total_countries = 0
    if df is not None:
        country_col = None
        for col in ["country", "country_x"]:
            if col in df.columns:
                country_col = col
                break
        if country_col:
            total_countries = df[country_col].nunique()

    best_male_time = "--:--:--"
    best_female_time = "--:--:--"

    if winners_df is not None and not winners_df.empty:
        male_winners = winners_df[winners_df["gender"] == "M"]
        if not male_winners.empty and "winning_time_sec" in male_winners.columns:
            best_male_sec = male_winners["winning_time_sec"].min()
            best_male_time = format_seconds_to_hms(best_male_sec)

        female_winners = winners_df[winners_df["gender"] == "F"]
        if not female_winners.empty and "winning_time_sec" in female_winners.columns:
            best_female_sec = female_winners["winning_time_sec"].min()
            best_female_time = format_seconds_to_hms(best_female_sec)

    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row1_col1:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-value'>{total_races}</div>"
            f"<div class='kpi-label'>Total de Corridas</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row1_col2:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-value'>{years_analyzed}</div>"
            f"<div class='kpi-label'>Anos Analisados</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row1_col3:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-value'>{total_finishers:,}</div>"
            f"<div class='kpi-label'>Total de Finalizadores</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row2_col1:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-value'>{total_countries}</div>"
            f"<div class='kpi-label'>Países</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row2_col2:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-value'>{best_male_time}</div>"
            f"<div class='kpi-label'>Melhor Tempo Masculino</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row2_col3:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-value'>{best_female_time}</div>"
            f"<div class='kpi-label'>Melhor Tempo Feminino</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def main():
    load_css()
    rename_nav_item()

    if "data_loaded" not in st.session_state:
        try:
            data = load_data()
            st.session_state["data"] = data
            st.session_state["data_loaded"] = True
        except Exception:
            st.session_state["data"] = {}
            st.session_state["data_loaded"] = False

    if "filters" not in st.session_state:
        st.session_state["filters"] = {}

    data_dict = st.session_state.get("data", {})
    main_df = data_dict.get("marathon_results")

    filters = render_sidebar_filters(main_df)
    st.session_state["filters"] = filters

    if main_df is not None:
        st.session_state["filtered_data"] = apply_filters(main_df, filters)
    else:
        st.session_state["filtered_data"] = None

    st.markdown(
        """
        <div class="main-title">
            <h1>World Marathon Majors - Painel Analitico</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h3 style='color: #475569; font-weight: 400; margin-top: -0.5rem;'>"
        "Painel Interativo de Análise das Grandes Maratonas Mundiais"
        "</h3>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "Bem-vindo ao painel de analise das **World Marathon Majors**, "
        "a série de elite das seis maiores maratonas do mundo: "
        "Berlim, Boston, Chicago, Londres, Nova York e Toquio. "
        "Este dashboard oferece uma visão abrangente sobre o desempenho "
        "dos corredores ao longo dos anos, com ferramentas interativas "
        "para filtrar, comparar e explorar os dados de forma detalhada."
    )

    st.markdown(
        "Utilize os filtros na barra lateral para refinar sua análise por ano, "
        "corrida, gênero, faixa etária e categoria de desempenho. "
        "Navegue pelas paginas do menu lateral para acessar visualizacoes "
        "especificas e insights aprofundados sobre cada aspecto das competicoes."
    )

    st.markdown("---")

    with st.expander("Como Navegar"):
        st.markdown(
            "- Utilize os **filtros** na barra lateral para aplicar filtros globais "
            "em todas as páginas do dashboard.\n"
            "- Selecione uma **pagina** no menu de navegacao para acessar analises "
            "especificas e visualizacoes detalhadas.\n"
            "- Marque a opcao **Apenas Corredores Brasileiros** para isolar os dados "
            "de atletas do Brasil.\n"
            "- Ajuste os controles deslizantes de **Faixa de Tempo de Chegada** e "
            "**Faixa de Pace** para filtrar com base em tempos.\n"
            "- Os filtros de **Faixa Etaria** e **Categoria de Desempenho** permitem "
            "segmentar os corredores em grupos especificos.\n"
            "- Todas as paginas do dashboard compartilham os mesmos filtros aplicados "
            "na barra lateral."
        )

    st.markdown("---")

    winners_df = data_dict.get("winners")
    filtered_df = st.session_state.get("filtered_data", main_df)
    render_kpi_metrics(filtered_df, winners_df)

    st.markdown("---")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.8rem; text-align: center;'>"
        "World Marathon Majors | Painel Analitico | "
        "Painel de Analise de Dados de Corridas"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
