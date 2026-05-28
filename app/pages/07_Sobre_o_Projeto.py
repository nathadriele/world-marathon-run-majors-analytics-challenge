import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_utils import load_css

load_css()

st.title("Sobre o Projeto - Teste")

st.markdown("---")

st.header("Visao Geral do Projeto")

st.markdown(
    """
    Este projeto apresenta um estudo abrangente de engenharia de dados, analise exploratoria \
    e aprendizado de maquina aplicado ao universo das Maratonas Majores Mundiais (World Marathon Majors). \
    As World Marathon Majors reune as seis maratonas mais prestigiosas do planeta: Boston, Nova York, \
    Chicago, Londres, Berlim e Tquio, atraindo os melhores corredores do mundo e centenas de milhares \
    de participantes amadores em cada edicao.

    O objetivo principal deste trabalho e transformar dados brutos de resultados de corridas em \
    insights acionaveis, modelos preditivos e visualizacoes interativas que permitam compreender \
    padroes de desempenho, fatores de sucesso e tendencias historicas no mundo das maratonas.

    Atraves de um pipeline completo de dados, desde a aquisicao e limpeza ate a modelagem \
    preditiva e a construcao de dashboards interativos, este projeto demonstra como tecnicas \
    modernas de ciencia de dados podem ser aplicadas para extrair valor de conjuntos de dados \
    complexos e multidimensionais do esporte.

    O estudo contempla ainda uma analise especial do desempenho de corredores brasileiros nas \
    Maratonas Majores, identificando tendencias, recordes e momentos historicos relevantes \
    para o atletismo nacional.
    """
)

st.markdown("---")

with st.expander("Metodologia"):
    st.subheader("Pipeline de Processamento de Dados")

    st.markdown(
        """
        O projeto segue uma metodologia rigorosa de 16 etapas, garantindo a qualidade, \
        consistencia e confiabilidade de todas as analises e modelos gerados.
        """
    )

    steps = [
        ("1. Aquisicao de Dados", "Coleta de dados de resultados das Maratonas Majores a partir de fontes publicas, incluindo tempos oficiais, informacoes de corredores e parciais de corrida."),
        ("2. Ingestao de Dados", "Carregamento e integracao dos datasets brutos em estruturas de dados tabulares, realizando a leitura de multiplos formatos de arquivo e consolidacao em um repositorio unificado."),
        ("3. Validacao", "Verificacao de integridade referencial, tipos de dados, intervalos validos e deteccao de anomalias nos registros brutos antes de qualquer processamento."),
        ("4. Limpeza", "Remocao de registros duplicados, tratamento de valores ausentes, correcao de inconsistencias e eliminacao de entradas invalidas ou incompletas."),
        ("5. Padronizacao", "Normalizacao de nomes de colunas, unificacao de formatos de pais e genero, padronizacao de codificacoes e aplicacao de convencoes consistentes em todo o dataset."),
        ("6. Conversao de Tempo", "Transformacao de representacoes de tempo em formato string (HH:MM:SS) para valores numericos em segundos, habilitando calculos e comparacoes matematicas."),
        ("7. Metricas de Corrida", "Calculo de metricas derivadas essenciais como ritmo medio (min/km), velocidade media (km/h), variacao percentual entre parciais e indicadores de desempenho por segmento."),
        ("8. Processamento de Parciais", "Analise detalhada dos tempos parciais (splits) de cada segmento da maratona, calculo de diferencas entre primeira e segunda metade e identificacao de estrategias de ritmo."),
        ("9. Integracao", "Mescla de datasets de diferentes maratonas e anos em um unico dataframe consolidado, preservando a rastreabilidade de origem e habilitando analises cross-evento."),
        ("10. Engenharia de Features", "Criacao de variaveis derivadas enriquecidas como faixas etarias, categorias de desempenho, indices de consistencia, faixas de ritmo e variaveis temporais."),
        ("11. Analise Exploratoria de Dados", "Investigacao sistematica de distribuicoes, correlacoes, tendencias temporais, outliers e padroes nos dados consolidados, gerando visualizacoes e estatisticas descritivas."),
        ("12. Analise do Brasil", "Estudo especializado do desempenho de corredores brasileiros nas Maratonas Majores, identificando melhores resultados, evolucao historica e comparacoes internacionais."),
        ("13. Analise de Ritmo", "Estudo aprofundado dos padroes de ritmo dos corredores, incluindo analise de pacing positivo vs negativo, variacao entre parciais e estrategias de corrida mais eficazes."),
        ("14. Modelagem ML", "Desenvolvimento e treinamento de modelos de aprendizado de maquina para tarefas de regressao (predicao de tempo final), classificacao (previsao de conclusao) e agrupamento (perfis de corredores)."),
        ("15. Avaliacao", "Validacao cruzada, analise de metricas de desempenho (MAE, RMSE, R2, acuracia, F1-score), comparacao entre algoritmos e otimizacao de hiperparametros."),
        ("16. Dashboard", "Construcao de aplicacao web interativa com Streamlit para visualizacao e exploracao de todas as analises, modelos e insights de forma acessivel e intuitiva.")
    ]

    for title, description in steps:
        st.markdown(f"**{title}**")
        st.markdown(description)
        st.markdown("")

st.markdown("---")

with st.expander("Fontes de Dados"):
    st.subheader("Fontes de Dados Utilizadas")

    st.markdown(
        """
        Os dados utilizados neste projeto foram obtidos a partir de fontes publicas e oficiais, \
        cobrindo resultados de todas as seis Maratonas Majores em diversas edicoes historicas.
        """
    )

    sources_data = {
        "Fonte": [
            "Resultados WMM (Kaggle)",
            "Historico de Vencedores (Wikipedia)",
            "Rankings World Athletics",
            "Tokyo Marathon (Oficial)",
            "Boston Marathon (BAA)",
            "London Marathon (Oficial)",
            "Berlin Marathon (SCC Events)",
            "Chicago Marathon (Oficial)",
            "NYC Marathon (NYRR)"
        ],
        "Descricao": [
            "Dataset consolidado com resultados de todas as Maratonas Majores",
            "Historico completo de vencedores e recordes de todas as edicoes",
            "Rankings oficiais de atletas de elite da World Athletics",
            "Resultados oficiais das edicoes da Maratona de Tquio",
            "Resultados oficiais da Boston Athletic Association",
            "Resultados oficiais da Maratona de Londres",
            "Resultados oficiais da Maratona de Berlim",
            "Resultados oficiais da Maratona de Chicago",
            "Resultados oficiais da New York Road Runners"
        ],
        "Cobertura": [
            "2018 a 2025",
            "Todas as edicoes",
            "Rankings atuais",
            "2018 a 2025",
            "2018 a 2025",
            "2018 a 2025",
            "2018 a 2025",
            "2018 a 2025",
            "2018 a 2025"
        ]
    }

    st.table(sources_data)

    st.markdown("")
    st.markdown(
        "**Nota sobre Integridade dos Dados:** Todos os dados utilizados neste projeto sao reais, \
        publicos e auditaveis. Os datasets primarios sao provenientes de contribuicoes da comunidade \
        Kaggle e referenciados com os sites oficiais das corridas e a Wikipedia. A proveniencia dos \
        dados e documentada em cada etapa do pipeline."
    )

    st.markdown(
        "**Nota sobre Dados Simulados:** Onde datasets historicos completos nao estao disponiveis \
        publicamente em uma unica fonte consolidada, dados simulados de corredores sao gerados com \
        base em distribuicoes estatisticas reais derivadas de resultados oficiais de maratonas. \
        As distribuicoes sao calibradas contra estatisticas publicas para garantir validade analitica."
    )

st.markdown("---")

with st.expander("Metricas de Maratona"):
    st.subheader("Principais Metricas Utilizadas na Analise")

    st.markdown(
        """
        A tabela a seguir define as principais metricas de corrida calculadas e analisadas \
        ao longo deste projeto. Todas as metricas sao derivadas dos dados oficiais de resultados \
        de maratonas.
        """
    )

    metrics_data = {
        "Metrica": [
            "Tempo de Chegada (finish_time)",
            "Ritmo por Quilometro (pace_per_km)",
            "Ritmo por Milha (pace_per_mile)",
            "Velocidade Media (average_speed_kmh)",
            "Parciais (splits)",
            "Split Negativo (negative_split)",
            "Split Positivo (positive_split)",
            "Variacao de Ritmo (pace_variation)"
        ],
        "Descricao": [
            "Tempo total para completar a maratona, medido em segundos ou formato HH:MM:SS. Este e o indicador principal de desempenho de cada corredor.",
            "Tempo medio gasto para percorrer um quilometro, expresso em minutos por quilometro (min/km). Calculado como tempo_total / 42,195. Valores menores indicam melhor desempenho.",
            "Tempo medio gasto para percorrer uma milha, expresso em minutos por milha (min/mi). Calculado como tempo_total / 26,2188. Comumente utilizado nos Estados Unidos e em Boston.",
            "Velocidade media mantida ao longo da corrida, expressa em quilometros por hora (km/h). Calculada como 42,195 / (tempo em horas). Valores maiores indicam melhor desempenho.",
            "Tempo registrado em pontos intermediarios de verificacao (5K, 10K, 15K, 20K, meia maratona, 25K, 30K, 35K, 40K). As parciais revelam como o corredor distribui o esforco.",
            "Uma corrida em que a segunda metade e completada mais rapido que a primeira. Splits negativos estao associados a um inicio conservador e chegadas fortes. Comum entre corredores de elite.",
            "Uma corrida em que a segunda metade e mais lenta que a primeira. Splits positivos sao o padrao mais comum entre corredores recreativos, frequentemente devido a ritmo inicial agressivo e fadiga.",
            "Desvio padrao do ritmo em todos os segmentos medidos (intervalos de 5K). Variacao menor indica ritmo mais consistente e uniforme. Fortemente correlacionado com melhor desempenho geral."
        ]
    }

    st.table(metrics_data)

    st.markdown("")
    st.markdown(
        "**Distancia Oficial da Maratona:** A distancia padrao da maratona e de **42,195 quilometros \
        (26,2188 milhas)**. Esta distancia foi estabelecida nos Jogos Olimpicos de Londres de 1908 e \
        padronizada pela Federacao Internacional de Atletismo (IAAF) em 1921. Todo percurso certificado \
        de maratona e medido com esta distancia exata com tolerancia de no maximo 0,1%."
    )

st.markdown("---")

with st.expander("Aprendizado de Maquina"):
    st.subheader("Abordagem de Aprendizado de Maquina")

    st.markdown(
        """
        Tres tarefas distintas de aprendizado de maquina sao abordadas neste projeto, cada \
        uma visando um aspecto diferente da analise de desempenho em maratonas.
        """
    )

    st.markdown("#### Regressao: Predicao de Tempo Final")
    st.markdown(
        """
        A tarefa de regressao tem como objetivo prever o tempo final de conclusao da maratona \
        com base em caracteristicas do corredor como idade, genero, pais de origem, maratona \
        disputada e indicadores temporais. Modelos como Random Forest Regressor, Gradient Boosting \
        (XGBoost e LightGBM) e Regressao Linear foram treinados e comparados utilizando metricas como \
        MAE (Erro Absoluto Medio), RMSE (Raiz do Erro Quadratico Medio) e R2 (Coeficiente de \
        Determinacao). A capacidade de prever tempos de chegada e util tanto para corredores \
        planejarem suas estrategias quanto para organizadores estimarem a logistica do evento.
        """
    )

    st.markdown("#### Classificacao: Previsao de Categoria de Desempenho")
    st.markdown(
        """
        A tarefa de classificacao busca prever categorias discretas de desempenho do corredor. \
        O modelo principal prevê em qual faixa de tempo o corredor concluira a maratona: \
        **Elite** (sub-2:20 para homens, sub-2:40 para mulheres), **Avancado** (sub-2:50 para homens, \
        sub-3:10 para mulheres), **Intermediario** (sub-3:30 para homens, sub-3:50 para mulheres) e \
        **Recreativo** (acima destes limiares). Algoritmos como Logistic Regression, Random Forest \
        Classifier, XGBoost Classifier e Support Vector Machines foram avaliados utilizando metricas \
        como acuracia, precisao, revocacao e F1-score.
        """
    )

    st.markdown("#### Agrupamento (Clustering): Identificacao de Perfis de Corredores")
    st.markdown(
        """
        Agrupamento nao supervisionado utilizando K-Means identifica perfis distintos de corredores \
        com base em comportamento de ritmo, tempos finais, demografia e preferencias de corrida. \
        A analise de clusters revela agrupamentos naturais como amadores competitivos, primeiros \
        finalistas, candidatos a faixas etarias e viajantes internacionais. O numero otimo de \
        clusters e determinado pela analise de silhouette score.
        """
    )

    st.markdown("#### Modelos e Validacao")
    st.markdown(
        """
        Os seguintes modelos sao treinados e avaliados:

        - **Random Forest Regressor/Classifier** -- metodo de ensemble robusto com importancia \
        de features integrada
        - **XGBoost** -- framework de gradient boosting otimizado para dados estruturados
        - **LightGBM** -- implementacao eficiente de gradient boosting para grandes datasets
        - **K-Means Clustering** -- particionamento nao supervisionado para perfilamento de corredores

        As metricas de avaliacao incluem Root Mean Squared Error (RMSE) e Mean Absolute Error (MAE) \
        para regressao, acuracia e F1-score ponderado para classificacao, e silhouette score para \
        agrupamento. Todos os modelos sao validados utilizando divisao estratificada treino-teste \
        com proporcao 80/20 e validacao cruzada de 5 folds.
        """
    )

    st.markdown("#### Nota Metodologica")
    st.markdown(
        """
        Os modelos preditivos deste projeto sao projetados para fins analiticos e educacionais. \
        O desempenho em maratonas e influenciado por diversos fatores nao capturados nos dados \
        disponiveis, incluindo volume de treinamento, condicoes climaticas no dia da corrida, \
        estado de lesao, estrategia nutricional e fatores psicologicos. Os modelos capturam apenas \
        a variancia explicavel atribuivel as features disponiveis e nao devem ser interpretados como \
        predicoes definitivas de desempenho para corredores individuais.
        """
    )

st.markdown("---")

with st.expander("Analise do Brasil"):
    st.subheader("Analise do Brasil nas Maratonas Majores")

    st.markdown(
        """
        Uma analise dedicada dos corredores brasileiros e incluida neste projeto por diversas \
        razoes importantes. O Brasil possui uma rica tradicao em maratonas, desde performances \
        de recordes mundiais ate a crescente participacao em massa em eventos internacionais. \
        Destacar a participacao brasileira nas Maratonas Majores Mundiais fornece uma lente \
        focada para examinar como corredores de um contexto nacional especifico performam no \
        cenario global.
        """
    )

    st.markdown("#### Momentos Notaveis do Atletismo Brasileiro em Maratonas")
    st.markdown(
        """
        **Ronaldo da Costa (Maratona de Berlim, 1998):** Estabeleceu o recorde mundial de maratona \
        com o tempo de 2:06:05, quebrando o recorde anterior em 45 segundos. Ele se tornou o \
        primeiro corredor a atingir um recorde mundial com split negativo na maratona, correndo \
        a segunda metade mais rapido que a primeira. Esta conquista permanece como um dos maiores \
        momentos da historia do atletismo brasileiro.
        """
    )
    st.markdown(
        """
        **Marilson Gomes dos Santos (Maratona de Nova York, 2006 e 2008):** Venceu a Maratona de \
        Nova York duas vezes, com tempos de 2:09:58 (2006) e 2:08:43 (2008). Ele foi o primeiro \
        sul-americano a vencer a Maratona de Nova York, estabelecendo o Brasil como uma forca \
        competitiva nas grandes maratonas mundiais.
        """
    )
    st.markdown(
        """
        **Crescente Participacao em Massa:** A participacao brasileira nas Maratonas Majores Mundiais \
        cresceu de forma constante de 2018 a 2025. A Maratona de Nova York e a Maratona de Boston \
        atraem consistentemente os maiores contingentes brasileiros, refletindo redes estabelecidas \
        de comunidades de viagem e corrida entre o Brasil e os Estados Unidos.
        """
    )

    st.markdown("#### Notas sobre Disponibilidade de Dados")
    st.markdown(
        """
        Registros de corredores brasileiros sao identificados utilizando o codigo de pais ISO 3166-1 \
        alpha-3 'BRA' no campo de pais/nacionalidade. Em datasets onde dados completos de \
        nacionalidade nao estao disponiveis, corredores brasileiros sao identificados atraves de \
        comparacao de nomes e dados complementares de inscricao. A analise inclui todos os \
        finalistas brasileiros com tempos validos em todas as seis corridas e todos os anos do \
        periodo de estudo.
        """
    )

st.markdown("---")

st.header("Tecnologias Utilizadas")

st.markdown(
    """
    O projeto foi desenvolvido utilizando um stack moderno de tecnologias de ciencia de dados \
    e desenvolvimento web, selecionadas por sua robustez, comunidade ativa e capacidade de \
    lidar com grandes volumes de dados.
    """
)

row1_html = """
<div style="display: flex; gap: 16px; margin-bottom: 16px;">
    <div style="flex: 1; background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%); border: 1px solid #93C5FD; border-radius: 10px; padding: 24px; text-align: center; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);">
        <h3 style="color: #1D4ED8; margin-bottom: 8px;">Python</h3>
        <p style="color: #334155; font-size: 14px;">Linguagem principal do projeto para processamento de dados, analise estatistica e desenvolvimento de modelos de machine learning</p>
    </div>
    <div style="flex: 1; background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%); border: 1px solid #93C5FD; border-radius: 10px; padding: 24px; text-align: center; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);">
        <h3 style="color: #1D4ED8; margin-bottom: 8px;">Pandas</h3>
        <p style="color: #334155; font-size: 14px;">Biblioteca de manipulacao e analise de dados tabulares, utilizada em todas as etapas de processamento e transformacao</p>
    </div>
    <div style="flex: 1; background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%); border: 1px solid #93C5FD; border-radius: 10px; padding: 24px; text-align: center; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);">
        <h3 style="color: #1D4ED8; margin-bottom: 8px;">NumPy</h3>
        <p style="color: #334155; font-size: 14px;">Biblioteca de computacao numerica para operacoes matematicas de alto desempenho e manipulacao de arrays</p>
    </div>
    <div style="flex: 1; background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%); border: 1px solid #93C5FD; border-radius: 10px; padding: 24px; text-align: center; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);">
        <h3 style="color: #1D4ED8; margin-bottom: 8px;">Scikit-learn</h3>
        <p style="color: #334155; font-size: 14px;">Framework de aprendizado de maquina para treinamento, avaliacao e selecao de modelos preditivos</p>
    </div>
</div>
"""

row2_html = """
<div style="display: flex; gap: 16px; margin-bottom: 16px;">
    <div style="flex: 1; background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%); border: 1px solid #93C5FD; border-radius: 10px; padding: 24px; text-align: center; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);">
        <h3 style="color: #1D4ED8; margin-bottom: 8px;">Plotly</h3>
        <p style="color: #334155; font-size: 14px;">Biblioteca de visualizacoes interativas para criacao de graficos dinamicos e exploracao visual dos dados</p>
    </div>
    <div style="flex: 1; background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%); border: 1px solid #93C5FD; border-radius: 10px; padding: 24px; text-align: center; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);">
        <h3 style="color: #1D4ED8; margin-bottom: 8px;">Streamlit</h3>
        <p style="color: #334155; font-size: 14px;">Framework para construcao de aplicacoes web interativas para ciencia de dados e dashboards analiticos</p>
    </div>
    <div style="flex: 1; background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%); border: 1px solid #93C5FD; border-radius: 10px; padding: 24px; text-align: center; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);">
        <h3 style="color: #1D4ED8; margin-bottom: 8px;">XGBoost</h3>
        <p style="color: #334155; font-size: 14px;">Algoritmo de gradient boosting de alto desempenho utilizado para tarefas de regressao e classificacao</p>
    </div>
    <div style="flex: 1; background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%); border: 1px solid #93C5FD; border-radius: 10px; padding: 24px; text-align: center; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);">
        <h3 style="color: #1D4ED8; margin-bottom: 8px;">LightGBM</h3>
        <p style="color: #334155; font-size: 14px;">Framework de gradient boosting eficiente e escalavel para grandes conjuntos de dados com treinamento rapido</p>
    </div>
</div>
"""

st.markdown(row1_html, unsafe_allow_html=True)
st.markdown(row2_html, unsafe_allow_html=True)

st.markdown("---")

with st.expander("Estrutura do Projeto"):
    st.subheader("Organizacao dos Arquivos e Diretorios")

    st.markdown(
        """
        O projeto segue uma estrutura padrao de projetos de ciencia de dados com separacao \
        clara entre dados, codigo fonte, modelos, notebooks e a aplicacao de dashboard.
        """
    )

    tree = (
        "world-marathon-majors-analytics/\n"
        "|\n"
        "|-- app/                          Aplicacao de dashboard Streamlit\n"
        "|   |-- pages/                    Modulos de paginas multi-page do Streamlit\n"
        "|   |   |-- 01_overview.py        Dashboard de visao geral e KPIs\n"
        "|   |   |-- 02_race_comparison.py Analise comparativa de corridas\n"
        "|   |   |-- 03_winners_records.py Pagina de vencedores e recordes\n"
        "|   |   |-- 04_brazil_analysis.py Analise de corredores brasileiros\n"
        "|   |   |-- 05_pace_splits_analysis.py Analise de ritmo e parciais\n"
        "|   |   |-- 06_machine_learning.py Modelos de machine learning\n"
        "|   |   |-- 07_about_project.py   Esta pagina (sobre o projeto)\n"
        "|   |-- assets/                   Recursos estaticos (imagens, estilos)\n"
        "|\n"
        "|-- data/                         Diretorio de dados\n"
        "|   |-- raw/                      Dados originais imutaveis\n"
        "|   |-- processed/                Datasets limpos e transformados\n"
        "|   |-- interim/                  Saidas intermediarias de processamento\n"
        "|   |-- external/                 Dados de terceiros e complementares\n"
        "|\n"
        "|-- models/                       Artefatos de modelos treinados\n"
        "|   |-- trained/                  Arquivos de modelos serializados (.pkl, .joblib)\n"
        "|   |-- metrics/                  Metricas e relatorios de avaliacao\n"
        "|\n"
        "|-- notebooks/                    Jupyter notebooks para EDA e prototipagem\n"
        "|\n"
        "|-- reports/                      Relatorios de analise gerados\n"
        "|   |-- figures/                  Graficos e visualizacoes salvos\n"
        "|\n"
        "|-- src/                          Pacote de codigo fonte\n"
        "|   |-- data/                     Modulos de carregamento e processamento de dados\n"
        "|   |-- features/                 Pipelines de engenharia de features\n"
        "|   |-- models/                   Scripts de treinamento e avaliacao de modelos\n"
        "|   |-- visualization/            Funcoes de plotagem e configuracoes de graficos\n"
        "|   |-- utils/                    Configuracao, auxiliares e constantes\n"
        "|   |-- imgs/                     Imagens de documentacao e diagramas"
    )

    st.code(tree, language="plaintext")

    st.markdown(
        """
        O diretorio **app/** contem a aplicacao Streamlit multi-page, onde cada pagina foca \
        em uma dimensao analitica especifica. O diretorio **src/** abriga todos os modulos \
        Python reutilizaveis organizados por funcao (processamento de dados, engenharia de features, \
        modelagem, visualizacao, utilitarios). O diretorio **data/** segue a convensao cookiecutter \
        de ciencia de dados com subdiretorios separados para dados brutos, processados, intermediarios \
        e externos.
        """
    )

st.markdown("---")

st.header("Limitacoes")

st.markdown(
    "Este projeto esta sujeito as seguintes limitacoes, que devem ser consideradas na \
    interpretacao das analises e resultados dos modelos."
)

st.markdown(
    "- **Dados Ausentes:** Nem todas as edicoes de maratonas possuem dados de resultados completos \
    disponiveis publicamente em um formato consistente. Algumas corridas fornecem dados de parciais \
    limitados, e edicoes mais antigas podem nao ter campos demograficos como idade ou pais de origem. \
    As temporadas de 2020 e 2021 foram interrompidas pela pandemia de COVID-19, resultando em corridas \
    canceladas ou modificadas com campos reduzidos."
)

st.markdown(
    "- **Dados Simulados:** Onde datasets reais completos nao estao disponiveis, dados simulados de \
    corredores sao gerados com base em distribuicoes estatisticas calibradas contra resultados reais de \
    corridas. Embora todo esforco seja feito para garantir validade analitica, dados simulados nao \
    capturam toda a complexidade e variabilidade de performances reais de maratona."
)

st.markdown(
    "- **Ausencia de Dados Climaticos e de Percurso:** Condicoes climaticas (temperatura, umidade, \
    vento) e fatores especificos do percurso (variacoes de elevacao, superficie da pista) impactam \
    significativamente o desempenho em maratona, mas nao estao disponiveis de forma consistente para \
    todas as corridas e anos no dataset. Isso limita o poder preditivo dos modelos e a profundidade \
    das analises."
)

st.markdown(
    "- **Disponibilidade Limitada de Parciais:** Tempos parciais detalhados em intervalos de 5 km nao \
    estao disponiveis para todos os corredores em todas as corridas. Dados de parciais tendem a ser \
    mais completos para corredores de elite e faixas etarias mais rapidas, o que pode introduzir \
    viés de selecao nas analises de ritmo."
)

st.markdown(
    "- **Limitacoes dos Modelos Preditivos:** Os modelos de aprendizado de maquina capturam apenas a \
    variancia explicavel pelas features disponiveis (demografia, corrida, ano). Determinantes criticos \
    de desempenho como historico de treinamento, estado de lesao, nutricao e fatores psicologicos nao \
    estao representados nos dados. As predicoes dos modelos devem ser interpretadas como estimativas \
    estatisticas em vez de previsoes individuais."
)

st.markdown("---")

st.header("Proximos Passos")

st.markdown(
    "As seguintes melhorias e extensoes estao planejadas para futuras iteracoes deste projeto."
)

st.markdown(
    "- Integrar dados climaticos de APIs de historico meteorologico para analisar o impacto \
    ambiental no desempenho dos corredores"
)

st.markdown(
    "- Incorporar perfis de elevacao do percurso de cada maratona para quantificar a dificuldade \
    de cada circuito"
)

st.markdown(
    "- Expandir o dataset para incluir anos adicionais (pre-2018 e edicoes futuras) e aumentar \
    a cobertura temporal da analise"
)

st.markdown(
    "- Adicionar modelos de deep learning (redes neurais) para comparacao com metodos baseados em arvores"
)

st.markdown(
    "- Implementar capacidades de predicao em tempo real para cenarios de corrida ao vivo"
)

st.markdown(
    "- Desenvolver um motor de recomendacao para selecao de corridas baseado no perfil do corredor"
)

st.markdown(
    "- Adicionar pontuacao de desempenho graduada por idade para comparacao justa entre faixas etarias"
)

st.markdown(
    "- Integrar dados de frequencia cardiaca e GPS de dispositivos vestiveis quando disponiveis"
)

st.markdown(
    "- Construir uma analise comparativa dos resultados das divisoes de cadeirantes e handcycle"
)

st.markdown(
    "- Criar uma API publica para acesso programatico aos resultados da analise"
)

st.markdown(
    "- Adicionar suporte multi-idioma para a interface do dashboard (Portugues, Japones, Alemao)"
)

st.markdown("---")

st.header("Licenca")

st.markdown(
    "Este projeto e distribuido sob a **Licenca MIT**."
)

st.markdown(
    """
    Copyright (c) 2024 World Marathon Majors Analytics

    A permissao e concedida, gratuitamente, a qualquer pessoa que obtenha uma copia deste software \
    e dos arquivos de documentacao associados (o "Software"), para lidar com o Software sem restricao, \
    incluindo, sem limitacao, os direitos de uso, copia, modificacao, mesclagem, publicacao, distribuicao, \
    sublicenciamento e/ou venda de copias do Software, e permitir que as pessoas a quem o Software e \
    fornecido o facam, sujeito as seguintes condicoes:

    O aviso de copyright acima e este aviso de permissao devem ser incluidos em todas as copias ou \
    partes substanciais do Software.
    """
)
