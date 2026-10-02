from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel


# =========================================================
# Configurações
# =========================================================

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1600)


# =========================================================
# Caminhos
# =========================================================

RAIZ = Path(__file__).resolve().parents[1]

CAMINHO_BASE = (
    RAIZ
    / "datasets"
    / "processed"
    / "base_integrada.csv"
)

PASTA_SAIDA = (
    RAIZ
    / "datasets"
    / "processed"
)

SAIDA_BASE_MODELAGEM = (
    PASTA_SAIDA
    / "base_modelagem.csv"
)

SAIDA_RESULTADOS_ORDINAL = (
    PASTA_SAIDA
    / "resultados_regressao_ordinal.csv"
)

SAIDA_SENSIBILIDADE = (
    PASTA_SAIDA
    / "sensibilidade_distancia.csv"
)

SAIDA_RESULTADOS_MULTINOMIAL = (
    PASTA_SAIDA
    / "resultados_regressao_multinomial.csv"
)

SAIDA_DIAGNOSTICO_AUSENCIA = (
    PASTA_SAIDA
    / "diagnostico_ausencia_precipitacao.csv"
)


# =========================================================
# Parâmetros metodológicos
# =========================================================

# Análise principal:
# - acidentes com precipitação válida;
# - estação meteorológica a até 30 km;
# - medição meteorológica a até 60 min.
#
# Os limites de 40 e 50 km são usados somente em análise
# de sensibilidade, não para escolher o resultado pelo p-valor.

LIMITE_DISTANCIA_KM = 30.0
LIMITE_TEMPO_MIN = 60.0

LIMITES_SENSIBILIDADE_KM = [
    30.0,
    40.0,
    50.0,
]

# Estações com poucos acidentes são agrupadas apenas no
# diagnóstico do mecanismo de ausência para reduzir instabilidade
# por categorias extremamente raras.
MIN_REGISTROS_ESTACAO_MODELO_AUSENCIA = 20


# =========================================================
# Funções auxiliares
# =========================================================

def formatar_significancia(p_valor):
    if pd.isna(p_valor):
        return ""

    if p_valor < 0.001:
        return "***"

    if p_valor < 0.01:
        return "**"

    if p_valor < 0.05:
        return "*"

    if p_valor < 0.1:
        return "."

    return "ns"


def agrupar_tracado_via(valor):
    if pd.isna(valor):
        return np.nan

    texto = str(valor).strip()

    if "Curva" in texto:
        return "Curva"

    if "Reta" in texto:
        return "Reta"

    if any(
        chave in texto
        for chave in [
            "Interseção",
            "Rotatória",
            "Cruzamento",
            "Retorno",
        ]
    ):
        return "Interseção"

    return "Outros"


def remover_colunas_sem_variacao(matriz):
    colunas_validas = [
        coluna
        for coluna in matriz.columns
        if matriz[coluna].nunique(dropna=False) > 1
    ]

    return matriz[colunas_validas].copy()


def extrair_tabela_modelo_ordinal(
    resultado,
    nome_modelo
):
    parametros = resultado.params
    erros = resultado.bse
    p_valores = resultado.pvalues
    intervalos = resultado.conf_int()

    linhas = []

    for variavel in parametros.index:

        # Pontos de corte do modelo ordinal
        if "/" in str(variavel) or str(variavel).startswith("cut"):
            continue

        coef = parametros[variavel]
        ep = erros.get(
            variavel,
            np.nan
        )
        p_val = p_valores.get(
            variavel,
            np.nan
        )

        if variavel in intervalos.index:
            ic_inf = intervalos.loc[
                variavel,
                0
            ]
            ic_sup = intervalos.loc[
                variavel,
                1
            ]
        else:
            ic_inf = np.nan
            ic_sup = np.nan

        linhas.append(
            {
                "modelo": nome_modelo,
                "variavel": variavel,
                "coeficiente": coef,
                "erro_padrao": ep,
                "p_valor": p_val,
                "sig": formatar_significancia(
                    p_val
                ),
                "odds_ratio": np.exp(
                    coef
                ),
                "ic_95_inf": np.exp(
                    ic_inf
                ),
                "ic_95_sup": np.exp(
                    ic_sup
                ),
            }
        )

    return pd.DataFrame(
        linhas
    )


def calcular_metricas(
    resultado,
    nome_modelo,
    n_obs
):
    return {
        "modelo": nome_modelo,
        "n_obs": n_obs,
        "log_likelihood": resultado.llf,
        "aic": resultado.aic,
        "bic": getattr(
            resultado,
            "bic",
            np.nan
        ),
    }


def preparar_dados_modelagem(
    dados_entrada,
    limite_distancia
):
    dados_modelo = dados_entrada[
        (
            dados_entrada[
                "status_integracao"
            ]
            == "integrado"
        )
        & dados_entrada[
            "precipitacao_mm"
        ].notna()
        & (
            dados_entrada[
                "distancia_estacao_km"
            ]
            <= limite_distancia
        )
        & (
            dados_entrada[
                "diferenca_tempo_min"
            ]
            <= LIMITE_TEMPO_MIN
        )
    ].copy()

    covariaveis_necessarias = [
        "classificacao_acidente",
        "fase_dia",
        "tipo_pista",
        "tracado_via",
        "ano",
        "mes",
    ]

    dados_modelo = (
        dados_modelo
        .dropna(
            subset=covariaveis_necessarias
        )
        .copy()
    )

    mapa_gravidade = {
        "Sem Vítimas": 0,
        "Com Vítimas Feridas": 1,
        "Com Vítimas Fatais": 2,
    }

    dados_modelo[
        "gravidade_ordinal"
    ] = (
        dados_modelo[
            "classificacao_acidente"
        ]
        .map(
            mapa_gravidade
        )
    )

    dados_modelo[
        "precipitacao_mm"
    ] = pd.to_numeric(
        dados_modelo[
            "precipitacao_mm"
        ],
        errors="coerce"
    )

    dados_modelo[
        "chuva_binaria"
    ] = (
        dados_modelo[
            "precipitacao_mm"
        ]
        > 0
    ).astype(
        int
    )

    # -------------------------
    # Padronização das covariáveis
    # -------------------------

    dados_modelo[
        "fase_dia"
    ] = (
        dados_modelo[
            "fase_dia"
        ]
        .astype(
            "string"
        )
        .str.strip()
        .replace(
            {
                "Plena noite":
                    "Plena Noite",
                "plena noite":
                    "Plena Noite",
                "pleno dia":
                    "Pleno dia",
            }
        )
    )

    dados_modelo[
        "tipo_pista"
    ] = (
        dados_modelo[
            "tipo_pista"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    dados_modelo[
        "tracado_via"
    ] = (
        dados_modelo[
            "tracado_via"
        ]
        .apply(
            agrupar_tracado_via
        )
    )

    # -------------------------
    # Categorias de referência
    # -------------------------

    dados_modelo[
        "fase_dia"
    ] = pd.Categorical(
        dados_modelo[
            "fase_dia"
        ],
        categories=[
            "Pleno dia",
            "Plena Noite",
            "Anoitecer",
            "Amanhecer",
        ],
        ordered=False
    )

    dados_modelo[
        "tipo_pista"
    ] = pd.Categorical(
        dados_modelo[
            "tipo_pista"
        ],
        categories=[
            "Dupla",
            "Simples",
            "Múltipla",
        ],
        ordered=False
    )

    dados_modelo[
        "tracado_via"
    ] = pd.Categorical(
        dados_modelo[
            "tracado_via"
        ],
        categories=[
            "Reta",
            "Curva",
            "Interseção",
            "Outros",
        ],
        ordered=False
    )

    # Ano e mês agora são explicitamente categóricos.
    # Isso evita tratá-los como variáveis numéricas lineares.
    anos = sorted(
        dados_modelo[
            "ano"
        ]
        .dropna()
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    meses = list(
        range(
            1,
            13
        )
    )

    dados_modelo[
        "ano"
    ] = pd.Categorical(
        dados_modelo[
            "ano"
        ].astype(
            int
        ),
        categories=anos,
        ordered=False
    )

    dados_modelo[
        "mes"
    ] = pd.Categorical(
        dados_modelo[
            "mes"
        ].astype(
            int
        ),
        categories=meses,
        ordered=False
    )

    dados_modelo = (
        dados_modelo
        .dropna(
            subset=[
                "gravidade_ordinal",
                "precipitacao_mm",
                "fase_dia",
                "tipo_pista",
                "tracado_via",
                "ano",
                "mes",
            ]
        )
        .copy()
    )

    dados_modelo[
        "gravidade_ordinal"
    ] = pd.Categorical(
        dados_modelo[
            "gravidade_ordinal"
        ].astype(
            int
        ),
        categories=[
            0,
            1,
            2,
        ],
        ordered=True
    )

    return dados_modelo


def criar_matriz_ajustada(
    dados_modelo,
    variavel_chuva
):
    categoricas = [
        "fase_dia",
        "tipo_pista",
        "tracado_via",
        "ano",
        "mes",
    ]

    dummies = pd.get_dummies(
        dados_modelo[
            categoricas
        ],
        columns=categoricas,
        drop_first=True,
        dtype=float
    )

    matriz = pd.concat(
        [
            dados_modelo[
                [
                    variavel_chuva
                ]
            ].astype(
                float
            ),
            dummies,
        ],
        axis=1
    )

    matriz = remover_colunas_sem_variacao(
        matriz
    )

    return matriz


def ajustar_modelo_ordinal(
    y,
    matriz_x,
    nome_modelo
):
    modelo = OrderedModel(
        y,
        matriz_x,
        distr="logit"
    )

    resultado = modelo.fit(
        method="bfgs",
        disp=False,
        maxiter=1500
    )

    tabela = extrair_tabela_modelo_ordinal(
        resultado,
        nome_modelo
    )

    return (
        resultado,
        tabela
    )


def extrair_resultado_multinomial(
    resultado,
    variavel_exposicao,
    nome_modelo
):
    """
    No MNLogit, como a gravidade é codificada:
      0 = Sem Vítimas       -> referência
      1 = Com Vítimas Feridas
      2 = Com Vítimas Fatais

    As duas colunas de parâmetros correspondem, respectivamente,
    às categorias 1 e 2 em comparação com a referência 0.
    """

    resultados = []

    comparacoes = [
        (
            0,
            "Com Vítimas Feridas vs Sem Vítimas"
        ),
        (
            1,
            "Com Vítimas Fatais vs Sem Vítimas"
        ),
    ]

    for coluna_parametro, comparacao in comparacoes:
        coef = resultado.params.loc[
            variavel_exposicao,
            coluna_parametro
        ]

        ep = resultado.bse.loc[
            variavel_exposicao,
            coluna_parametro
        ]

        p_valor = resultado.pvalues.loc[
            variavel_exposicao,
            coluna_parametro
        ]

        ic_inf = coef - 1.96 * ep
        ic_sup = coef + 1.96 * ep

        resultados.append(
            {
                "modelo": nome_modelo,
                "comparacao": comparacao,
                "variavel":
                    variavel_exposicao,
                "coeficiente":
                    coef,
                "erro_padrao":
                    ep,
                "p_valor":
                    p_valor,
                "sig":
                    formatar_significancia(
                        p_valor
                    ),
                "odds_ratio":
                    np.exp(
                        coef
                    ),
                "ic_95_inf":
                    np.exp(
                        ic_inf
                    ),
                "ic_95_sup":
                    np.exp(
                        ic_sup
                    ),
            }
        )

    return pd.DataFrame(
        resultados
    )


# =========================================================
# Carregar base
# =========================================================

if not CAMINHO_BASE.exists():
    raise FileNotFoundError(
        f"Base integrada não encontrada em: "
        f"{CAMINHO_BASE}\n"
        "Execute as etapas anteriores antes da modelagem."
    )

dados = pd.read_csv(
    CAMINHO_BASE
)

print()
print(
    "="
    * 72
)
print(
    "ANÁLISE DA ASSOCIAÇÃO ENTRE PRECIPITAÇÃO E GRAVIDADE"
)
print(
    "="
    * 72
)

print(
    f"\nQuantidade inicial de acidentes: "
    f"{len(dados):,}"
)


# =========================================================
# 1. Diagnóstico dos dados ausentes de precipitação
# =========================================================

print()
print(
    "="
    * 72
)
print(
    "DIAGNÓSTICO DOS DADOS AUSENTES DE PRECIPITAÇÃO"
)
print(
    "="
    * 72
)

dados[
    "precipitacao_ausente"
] = (
    dados[
        "precipitacao_mm"
    ]
    .isna()
    .astype(
        int
    )
)

total_ausente = int(
    dados[
        "precipitacao_ausente"
    ]
    .sum()
)

print(
    f"\nPrecipitação ausente: "
    f"{total_ausente:,} "
    f"("
    f"{total_ausente / len(dados) * 100:.2f}%"
    f")"
)


def imprimir_ausencia_por(
    coluna
):
    if coluna not in dados.columns:
        return

    tabela = (
        dados
        .groupby(
            coluna,
            dropna=False
        )[
            "precipitacao_ausente"
        ]
        .agg(
            total="size",
            ausentes="sum"
        )
        .reset_index()
    )

    tabela[
        "percentual_ausente"
    ] = (
        tabela[
            "ausentes"
        ]
        / tabela[
            "total"
        ]
        * 100
    )

    print()
    print(
        f"Ausência por {coluna}:"
    )

    print(
        tabela.to_string(
            index=False,
            formatters={
                "percentual_ausente":
                    lambda x:
                    f"{x:.2f}%"
            }
        )
    )


for coluna in [
    "ano",
    "mes",
    "classificacao_acidente",
    "condicao_metereologica",
]:
    imprimir_ausencia_por(
        coluna
    )


# -------------------------
# Faixas de distância
# -------------------------

dados[
    "faixa_distancia"
] = pd.cut(
    dados[
        "distancia_estacao_km"
    ],
    bins=[
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        np.inf,
    ],
    labels=[
        "0-10",
        "10-20",
        "20-30",
        "30-40",
        "40-50",
        "50-60",
        ">60",
    ],
    include_lowest=True
)

imprimir_ausencia_por(
    "faixa_distancia"
)


# -------------------------
# Ausência por estação
# -------------------------

if "estacao_inmet" in dados.columns:
    tabela_estacoes = (
        dados
        .groupby(
            "estacao_inmet",
            dropna=False
        )[
            "precipitacao_ausente"
        ]
        .agg(
            total="size",
            ausentes="sum"
        )
        .reset_index()
    )

    tabela_estacoes[
        "percentual_ausente"
    ] = (
        tabela_estacoes[
            "ausentes"
        ]
        / tabela_estacoes[
            "total"
        ]
        * 100
    )

    tabela_estacoes = (
        tabela_estacoes
        .sort_values(
            [
                "percentual_ausente",
                "ausentes",
            ],
            ascending=[
                False,
                False,
            ]
        )
    )

    print()
    print(
        "15 estações com maior percentual de ausência:"
    )

    print(
        tabela_estacoes
        .head(
            15
        )
        .to_string(
            index=False,
            formatters={
                "percentual_ausente":
                    lambda x:
                    f"{x:.2f}%"
            }
        )
    )


# =========================================================
# 1.1 Modelo da probabilidade de precipitação ausente
#     incluindo efeito da estação
# =========================================================

print()
print(
    "-"
    * 72
)
print(
    "MODELO DA PROBABILIDADE DE PRECIPITAÇÃO AUSENTE"
)
print(
    "-"
    * 72
)

dados_ausencia = (
    dados
    .copy()
)

dados_ausencia[
    "condicao_metereologica"
] = (
    dados_ausencia[
        "condicao_metereologica"
    ]
    .fillna(
        "Ausente"
    )
)

if "estacao_inmet" in dados_ausencia.columns:
    dados_ausencia[
        "estacao_inmet"
    ] = (
        dados_ausencia[
            "estacao_inmet"
        ]
        .fillna(
            "SEM_ESTACAO"
        )
        .astype(
            str
        )
    )

    contagem_estacoes = (
        dados_ausencia[
            "estacao_inmet"
        ]
        .value_counts()
    )

    estacoes_raras = (
        contagem_estacoes[
            contagem_estacoes
            < MIN_REGISTROS_ESTACAO_MODELO_AUSENCIA
        ]
        .index
    )

    dados_ausencia[
        "estacao_modelo_ausencia"
    ] = (
        dados_ausencia[
            "estacao_inmet"
        ]
        .where(
            ~dados_ausencia[
                "estacao_inmet"
            ]
            .isin(
                estacoes_raras
            ),
            "OUTRAS_ESTACOES"
        )
    )
else:
    dados_ausencia[
        "estacao_modelo_ausencia"
    ] = "SEM_ESTACAO"

formula_ausencia = (
    "precipitacao_ausente ~ "
    "C(classificacao_acidente) + "
    "C(ano) + "
    "C(mes) + "
    "distancia_estacao_km + "
    "C(condicao_metereologica) + "
    "C(estacao_modelo_ausencia)"
)

print(
    "\nModelo ajustado por gravidade, ano, mês, "
    "distância, condição meteorológica da PRF e estação."
)

try:
    modelo_ausencia = (
        smf.glm(
            formula=formula_ausencia,
            data=dados_ausencia,
            family=sm.families.Binomial()
        )
        .fit(
            maxiter=500
        )
    )

    resultado_ausencia = pd.DataFrame(
        {
            "coeficiente":
                modelo_ausencia.params,
            "erro_padrao":
                modelo_ausencia.bse,
            "odds_ratio":
                np.exp(
                    modelo_ausencia.params
                ),
            "p_valor":
                modelo_ausencia.pvalues,
        }
    )

    ic_ausencia = (
        modelo_ausencia
        .conf_int()
    )

    resultado_ausencia[
        "ic_95_inf"
    ] = np.exp(
        ic_ausencia[
            0
        ]
    )

    resultado_ausencia[
        "ic_95_sup"
    ] = np.exp(
        ic_ausencia[
            1
        ]
    )

    resultado_ausencia[
        "sig"
    ] = (
        resultado_ausencia[
            "p_valor"
        ]
        .apply(
            formatar_significancia
        )
    )

    linhas_gravidade = [
        indice
        for indice
        in resultado_ausencia.index
        if "classificacao_acidente"
        in indice
    ]

    print(
        "\nTermos de gravidade no modelo de ausência:"
    )

    print(
        resultado_ausencia.loc[
            linhas_gravidade,
            [
                "odds_ratio",
                "ic_95_inf",
                "ic_95_sup",
                "p_valor",
                "sig",
            ]
        ]
        .round(
            4
        )
    )

    gravidade_associada_ausencia = any(
        resultado_ausencia.loc[
            linhas_gravidade,
            "p_valor"
        ]
        < 0.05
    )

    print()

    if gravidade_associada_ausencia:
        print(
            "RESULTADO: a gravidade continua associada à "
            "ausência de precipitação mesmo após o ajuste "
            "incluindo a estação."
        )
        print(
            "Casos completos podem ser usados na análise principal, "
            "mas o possível viés por dados ausentes deve ser tratado "
            "como limitação e acompanhado de análises de sensibilidade."
        )
    else:
        print(
            "RESULTADO: após incluir a estação e os demais ajustes, "
            "não foi detectada associação estatisticamente "
            "significativa entre gravidade e ausência de precipitação."
        )
        print(
            "Isso torna a análise por casos completos mais defensável, "
            "embora não prove que os dados sejam MCAR."
        )

except Exception as erro:
    print(
        "\nNão foi possível ajustar o modelo de ausência:"
    )
    print(
        erro
    )

    resultado_ausencia = pd.DataFrame()


# =========================================================
# 2. Fluxo de seleção da análise principal
# =========================================================

print()
print(
    "="
    * 72
)
print(
    "FLUXO DE SELEÇÃO DA ANÁLISE PRINCIPAL"
)
print(
    "="
    * 72
)

total_inicial = len(
    dados
)

mascara_precipitacao = (
    (
        dados[
            "status_integracao"
        ]
        == "integrado"
    )
    & dados[
        "precipitacao_mm"
    ]
    .notna()
)

dados_etapa1 = (
    dados[
        mascara_precipitacao
    ]
    .copy()
)

perda_precipitacao = (
    total_inicial
    - len(
        dados_etapa1
    )
)

dados_etapa2 = (
    dados_etapa1[
        dados_etapa1[
            "distancia_estacao_km"
        ]
        <= LIMITE_DISTANCIA_KM
    ]
    .copy()
)

perda_distancia = (
    len(
        dados_etapa1
    )
    - len(
        dados_etapa2
    )
)

dados_etapa3 = (
    dados_etapa2[
        dados_etapa2[
            "diferenca_tempo_min"
        ]
        <= LIMITE_TEMPO_MIN
    ]
    .copy()
)

perda_tempo = (
    len(
        dados_etapa2
    )
    - len(
        dados_etapa3
    )
)

covariaveis_necessarias = [
    "classificacao_acidente",
    "fase_dia",
    "tipo_pista",
    "tracado_via",
    "ano",
    "mes",
]

dados_etapa4 = (
    dados_etapa3
    .dropna(
        subset=covariaveis_necessarias
    )
    .copy()
)

perda_covariaveis = (
    len(
        dados_etapa3
    )
    - len(
        dados_etapa4
    )
)

fluxo_exclusao = pd.DataFrame(
    [
        {
            "etapa":
                "Base integrada inicial",
            "registros_restantes":
                total_inicial,
            "excluidos_etapa":
                0,
        },
        {
            "etapa":
                "Sem medição de precipitação",
            "registros_restantes":
                len(
                    dados_etapa1
                ),
            "excluidos_etapa":
                perda_precipitacao,
        },
        {
            "etapa":
                f"Distância > "
                f"{LIMITE_DISTANCIA_KM:.0f} km",
            "registros_restantes":
                len(
                    dados_etapa2
                ),
            "excluidos_etapa":
                perda_distancia,
        },
        {
            "etapa":
                f"Diferença temporal > "
                f"{LIMITE_TEMPO_MIN:.0f} min",
            "registros_restantes":
                len(
                    dados_etapa3
                ),
            "excluidos_etapa":
                perda_tempo,
        },
        {
            "etapa":
                "Ausência nas covariáveis",
            "registros_restantes":
                len(
                    dados_etapa4
                ),
            "excluidos_etapa":
                perda_covariaveis,
        },
    ]
)

fluxo_exclusao[
    "percentual_amostra_inicial"
] = (
    fluxo_exclusao[
        "registros_restantes"
    ]
    / total_inicial
    * 100
).round(
    2
)

print()
print(
    fluxo_exclusao
    .to_string(
        index=False
    )
)


# =========================================================
# 3. Preparar amostra principal
# =========================================================

dados_modelo = (
    preparar_dados_modelagem(
        dados,
        LIMITE_DISTANCIA_KM
    )
)

print()
print(
    "="
    * 72
)
print(
    "AMOSTRA FINAL DE MODELAGEM"
)
print(
    "="
    * 72
)

print(
    f"\nTamanho final da amostra: "
    f"{len(dados_modelo):,}"
)

print(
    "\nDistribuição da gravidade:"
)

resumo_gravidade = (
    dados_modelo[
        "classificacao_acidente"
    ]
    .value_counts()
    .rename(
        "acidentes"
    )
    .to_frame()
)

resumo_gravidade[
    "percentual"
] = (
    resumo_gravidade[
        "acidentes"
    ]
    / len(
        dados_modelo
    )
    * 100
).round(
    2
)

print(
    resumo_gravidade
)

print(
    "\nDistribuição por ano e gravidade:"
)

print(
    pd.crosstab(
        dados_modelo[
            "ano"
        ],
        dados_modelo[
            "classificacao_acidente"
        ],
        margins=True,
        margins_name="Total"
    )
)

print(
    "\nOcorrência de chuva (> 0 mm):"
)

resumo_chuva = (
    dados_modelo[
        "chuva_binaria"
    ]
    .value_counts()
    .rename(
        {
            0:
                "Sem Chuva",
            1:
                "Com Chuva",
        }
    )
    .to_frame(
        "acidentes"
    )
)

resumo_chuva[
    "percentual"
] = (
    resumo_chuva[
        "acidentes"
    ]
    / len(
        dados_modelo
    )
    * 100
).round(
    2
)

print(
    resumo_chuva
)


# =========================================================
# 4. Matrizes dos modelos
# =========================================================

matriz_x_m1a = (
    dados_modelo[
        [
            "precipitacao_mm"
        ]
    ]
    .astype(
        float
    )
)

matriz_x_m1b = (
    dados_modelo[
        [
            "chuva_binaria"
        ]
    ]
    .astype(
        float
    )
)

matriz_x_m2a = (
    criar_matriz_ajustada(
        dados_modelo,
        "precipitacao_mm"
    )
)

matriz_x_m2b = (
    criar_matriz_ajustada(
        dados_modelo,
        "chuva_binaria"
    )
)

y_ordinal = (
    dados_modelo[
        "gravidade_ordinal"
    ]
)


# =========================================================
# 5. Regressão logística ordinal
# =========================================================

print()
print(
    "="
    * 72
)
print(
    "REGRESSÃO LOGÍSTICA ORDINAL"
)
print(
    "="
    * 72
)

resultados_ordinais = []
metricas_ordinais = []


def executar_e_imprimir_ordinal(
    nome_modelo,
    matriz_x
):
    print()
    print(
        "-"
        * 72
    )
    print(
        nome_modelo.upper()
    )
    print(
        "-"
        * 72
    )

    resultado, tabela = (
        ajustar_modelo_ordinal(
            y_ordinal,
            matriz_x,
            nome_modelo
        )
    )

    print(
        resultado.summary()
    )

    print(
        "\nOdds Ratios:"
    )

    print(
        tabela[
            [
                "variavel",
                "coeficiente",
                "erro_padrao",
                "p_valor",
                "sig",
                "odds_ratio",
                "ic_95_inf",
                "ic_95_sup",
            ]
        ]
        .to_string(
            index=False
        )
    )

    resultados_ordinais.append(
        tabela
    )

    metricas_ordinais.append(
        calcular_metricas(
            resultado,
            nome_modelo,
            len(
                dados_modelo
            )
        )
    )

    return resultado


resultado_m1a = (
    executar_e_imprimir_ordinal(
        "Modelo 1A - Bruto - Precipitação mm",
        matriz_x_m1a
    )
)

resultado_m1b = (
    executar_e_imprimir_ordinal(
        "Modelo 1B - Bruto - Chuva binária",
        matriz_x_m1b
    )
)

resultado_m2a = (
    executar_e_imprimir_ordinal(
        "Modelo 2A - Ajustado - Precipitação mm",
        matriz_x_m2a
    )
)

resultado_m2b = (
    executar_e_imprimir_ordinal(
        "Modelo 2B - Ajustado - Chuva binária",
        matriz_x_m2b
    )
)


# =========================================================
# 6. Diagnóstico aproximado de chances proporcionais
# =========================================================

print()
print(
    "="
    * 72
)
print(
    "DIAGNÓSTICO DA SUPOSIÇÃO DE CHANCES PROPORCIONAIS"
)
print(
    "="
    * 72
)

print(
    "\nDiagnóstico exploratório: comparação dos coeficientes "
    "em dois cortes cumulativos. Não substitui um teste formal "
    "de proportional odds."
)

gravidade_num = (
    dados_modelo[
        "gravidade_ordinal"
    ]
    .astype(
        int
    )
)

corte_1 = (
    gravidade_num
    >= 1
).astype(
    int
)

corte_2 = (
    gravidade_num
    >= 2
).astype(
    int
)

matriz_x_const = (
    sm.add_constant(
        matriz_x_m2a,
        has_constant="add"
    )
)

logit_corte1 = (
    sm.Logit(
        corte_1,
        matriz_x_const
    )
    .fit(
        disp=False,
        maxiter=1000
    )
)

logit_corte2 = (
    sm.Logit(
        corte_2,
        matriz_x_const
    )
    .fit(
        disp=False,
        maxiter=1000
    )
)

comparacao_cortes = []

for variavel in [
    v
    for v
    in logit_corte1.params.index
    if v != "const"
]:
    b1 = (
        logit_corte1
        .params
        .get(
            variavel,
            np.nan
        )
    )

    b2 = (
        logit_corte2
        .params
        .get(
            variavel,
            np.nan
        )
    )

    bord = (
        resultado_m2a
        .params
        .get(
            variavel,
            np.nan
        )
    )

    comparacao_cortes.append(
        {
            "variavel":
                variavel,
            "beta_corte1":
                b1,
            "or_corte1":
                np.exp(
                    b1
                ),
            "p_corte1":
                logit_corte1
                .pvalues
                .get(
                    variavel,
                    np.nan
                ),
            "beta_corte2":
                b2,
            "or_corte2":
                np.exp(
                    b2
                ),
            "p_corte2":
                logit_corte2
                .pvalues
                .get(
                    variavel,
                    np.nan
                ),
            "beta_ordinal":
                bord,
            "or_ordinal":
                (
                    np.exp(
                        bord
                    )
                    if not np.isnan(
                        bord
                    )
                    else np.nan
                ),
        }
    )

df_proporcional = pd.DataFrame(
    comparacao_cortes
)

print(
    df_proporcional
    .round(
        4
    )
    .to_string(
        index=False
    )
)


# =========================================================
# 7. Análise multinomial de sensibilidade
# =========================================================

print()
print(
    "="
    * 72
)
print(
    "REGRESSÃO LOGÍSTICA MULTINOMIAL DE SENSIBILIDADE"
)
print(
    "="
    * 72
)

print(
    "\nReferência do desfecho: Sem Vítimas."
)
print(
    "A multinomial é usada como verificação porque não exige "
    "a hipótese de chances proporcionais da regressão ordinal."
)

y_multinomial = (
    dados_modelo[
        "gravidade_ordinal"
    ]
    .astype(
        int
    )
)

resultados_multinomiais = []

for (
    nome_modelo,
    variavel_exposicao,
    matriz_ajustada
) in [
    (
        "Multinomial ajustado - Precipitação mm",
        "precipitacao_mm",
        matriz_x_m2a
    ),
    (
        "Multinomial ajustado - Chuva binária",
        "chuva_binaria",
        matriz_x_m2b
    ),
]:
    print()
    print(
        "-"
        * 72
    )
    print(
        nome_modelo.upper()
    )
    print(
        "-"
        * 72
    )

    x_multinomial = (
        sm.add_constant(
            matriz_ajustada,
            has_constant="add"
        )
    )

    modelo_multinomial = (
        sm.MNLogit(
            y_multinomial,
            x_multinomial
        )
    )

    resultado_multinomial = (
        modelo_multinomial
        .fit(
            method="newton",
            disp=False,
            maxiter=1000
        )
    )

    print(
        resultado_multinomial.summary()
    )

    tabela_multinomial = (
        extrair_resultado_multinomial(
            resultado_multinomial,
            variavel_exposicao,
            nome_modelo
        )
    )

    print(
        "\nEfeito da precipitação:"
    )

    print(
        tabela_multinomial[
            [
                "comparacao",
                "variavel",
                "coeficiente",
                "erro_padrao",
                "p_valor",
                "sig",
                "odds_ratio",
                "ic_95_inf",
                "ic_95_sup",
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    resultados_multinomiais.append(
        tabela_multinomial
    )


# =========================================================
# 8. Sensibilidade ao limite espacial
# =========================================================

print()
print(
    "="
    * 72
)
print(
    "ANÁLISE DE SENSIBILIDADE DO LIMITE ESPACIAL"
)
print(
    "="
    * 72
)

sensibilidade = []

for limite in LIMITES_SENSIBILIDADE_KM:
    dados_sensibilidade = (
        preparar_dados_modelagem(
            dados,
            limite
        )
    )

    if dados_sensibilidade.empty:
        continue

    y_sensibilidade = (
        dados_sensibilidade[
            "gravidade_ordinal"
        ]
    )

    x_mm = (
        criar_matriz_ajustada(
            dados_sensibilidade,
            "precipitacao_mm"
        )
    )

    x_bin = (
        criar_matriz_ajustada(
            dados_sensibilidade,
            "chuva_binaria"
        )
    )

    resultado_mm, _ = (
        ajustar_modelo_ordinal(
            y_sensibilidade,
            x_mm,
            (
                "Sensibilidade "
                f"{limite:.0f} km - mm"
            )
        )
    )

    resultado_bin, _ = (
        ajustar_modelo_ordinal(
            y_sensibilidade,
            x_bin,
            (
                "Sensibilidade "
                f"{limite:.0f} km - binária"
            )
        )
    )

    for (
        nome_variavel,
        modelo
    ) in [
        (
            "precipitacao_mm",
            resultado_mm
        ),
        (
            "chuva_binaria",
            resultado_bin
        ),
    ]:
        coef = (
            modelo.params[
                nome_variavel
            ]
        )

        p_val = (
            modelo.pvalues[
                nome_variavel
            ]
        )

        ic = (
            modelo
            .conf_int()
            .loc[
                nome_variavel
            ]
        )

        sensibilidade.append(
            {
                "limite_km":
                    limite,
                "n":
                    len(
                        dados_sensibilidade
                    ),
                "exposicao":
                    nome_variavel,
                "odds_ratio":
                    np.exp(
                        coef
                    ),
                "ic_95_inf":
                    np.exp(
                        ic.iloc[
                            0
                        ]
                    ),
                "ic_95_sup":
                    np.exp(
                        ic.iloc[
                            1
                        ]
                    ),
                "p_valor":
                    p_val,
                "sig":
                    formatar_significancia(
                        p_val
                    ),
            }
        )

df_sensibilidade = pd.DataFrame(
    sensibilidade
)

print(
    df_sensibilidade
    .round(
        4
    )
    .to_string(
        index=False
    )
)


# =========================================================
# 9. Comparação de ajuste dos modelos ordinais
# =========================================================

print()
print(
    "="
    * 72
)
print(
    "COMPARAÇÃO DE AJUSTE DOS MODELOS ORDINAIS"
)
print(
    "="
    * 72
)

tabela_metricas = pd.DataFrame(
    metricas_ordinais
)

print(
    tabela_metricas
    .to_string(
        index=False
    )
)


# =========================================================
# 10. Salvar resultados
# =========================================================

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)

todas_tabelas_ordinais = pd.concat(
    resultados_ordinais,
    ignore_index=True
)

if resultados_multinomiais:
    todas_tabelas_multinomiais = pd.concat(
        resultados_multinomiais,
        ignore_index=True
    )
else:
    todas_tabelas_multinomiais = pd.DataFrame()

dados_modelo.to_csv(
    SAIDA_BASE_MODELAGEM,
    index=False,
    encoding="utf-8-sig"
)

todas_tabelas_ordinais.to_csv(
    SAIDA_RESULTADOS_ORDINAL,
    index=False,
    encoding="utf-8-sig"
)

df_sensibilidade.to_csv(
    SAIDA_SENSIBILIDADE,
    index=False,
    encoding="utf-8-sig"
)

todas_tabelas_multinomiais.to_csv(
    SAIDA_RESULTADOS_MULTINOMIAL,
    index=False,
    encoding="utf-8-sig"
)

if not resultado_ausencia.empty:
    resultado_ausencia.to_csv(
        SAIDA_DIAGNOSTICO_AUSENCIA,
        index=True,
        encoding="utf-8-sig"
    )


# =========================================================
# Finalização
# =========================================================

print()
print(
    "="
    * 72
)
print(
    "FINALIZAÇÃO"
)
print(
    "="
    * 72
)

print(
    f"\nBase de modelagem:\n"
    f"{SAIDA_BASE_MODELAGEM}"
)

print(
    f"\nResultados ordinais:\n"
    f"{SAIDA_RESULTADOS_ORDINAL}"
)

print(
    f"\nResultados multinomiais:\n"
    f"{SAIDA_RESULTADOS_MULTINOMIAL}"
)

print(
    f"\nSensibilidade espacial:\n"
    f"{SAIDA_SENSIBILIDADE}"
)

print(
    f"\nDiagnóstico de ausência:\n"
    f"{SAIDA_DIAGNOSTICO_AUSENCIA}"
)

print(
    "\nExecução concluída."
)
