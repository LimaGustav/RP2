from pathlib import Path

import pandas as pd


# =========================================================
# Caminho
# =========================================================

RAIZ = Path(__file__).resolve().parents[1]

CAMINHO = (
    RAIZ
    / "datasets"
    / "processed"
    / "base_integrada.csv"
)


# =========================================================
# Carregar base integrada
# =========================================================

dados = pd.read_csv(CAMINHO)


# =========================================================
# Resumo geral
# =========================================================

print()
print("=" * 60)
print("ANÁLISE DESCRITIVA")
print("=" * 60)

print("\nQuantidade total de acidentes:")
print(len(dados))


# =========================================================
# Acidentes por ano
# =========================================================

if "ano" in dados.columns:
    print("\nAcidentes por ano:")
    print(
        dados["ano"]
        .value_counts()
        .sort_index()
    )


# =========================================================
# Gravidade geral
# =========================================================

print("\nGravidade:")
print(
    dados["classificacao_acidente"]
    .value_counts(dropna=False)
)


# =========================================================
# Gravidade por ano
# =========================================================

if "ano" in dados.columns:
    print("\nGravidade por ano:")
    print(
        pd.crosstab(
            dados["ano"],
            dados["classificacao_acidente"]
        )
    )


# =========================================================
# Status da integração
# =========================================================

if "status_integracao" in dados.columns:
    print("\nStatus da integração:")
    print(
        dados["status_integracao"]
        .value_counts(dropna=False)
    )


# =========================================================
# Status da integração por ano
# =========================================================

if (
    "ano" in dados.columns
    and "status_integracao" in dados.columns
):
    print("\nStatus da integração por ano:")
    print(
        pd.crosstab(
            dados["ano"],
            dados["status_integracao"]
        )
    )


# =========================================================
# Precipitação
# =========================================================

print("\nPrecipitação:")
print(
    dados["precipitacao_mm"]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95
        ]
    )
)


# =========================================================
# Precipitação ausente
# =========================================================

quantidade_ausente = (
    dados["precipitacao_mm"]
    .isna()
    .sum()
)

percentual_ausente = (
    quantidade_ausente
    / len(dados)
    * 100
)

print("\nPrecipitação ausente:")
print(
    f"{quantidade_ausente} "
    f"({percentual_ausente:.2f}%)"
)


# =========================================================
# Precipitação ausente por ano
# =========================================================

if "ano" in dados.columns:
    print("\nPrecipitação ausente por ano:")

    resumo_ausentes = (
        dados.groupby("ano")["precipitacao_mm"]
        .agg(
            total="size",
            ausentes=lambda x: x.isna().sum()
        )
    )

    resumo_ausentes["percentual_ausente"] = (
        resumo_ausentes["ausentes"]
        / resumo_ausentes["total"]
        * 100
    )

    print(resumo_ausentes)


# =========================================================
# Distância até estação
# =========================================================

print("\nDistância até estação:")
print(
    dados["distancia_estacao_km"]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95
        ]
    )
)


# =========================================================
# Distância por ano
# =========================================================

if "ano" in dados.columns:
    print("\nDistância até estação por ano:")
    print(
        dados.groupby("ano")["distancia_estacao_km"]
        .describe()
    )


# =========================================================
# Diferença temporal
# =========================================================

print("\nDiferença temporal:")
print(
    dados["diferenca_tempo_min"]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95
        ]
    )
)


# =========================================================
# Diferença temporal por ano
# =========================================================

if "ano" in dados.columns:
    print("\nDiferença temporal por ano:")
    print(
        dados.groupby("ano")["diferenca_tempo_min"]
        .describe()
    )


# =========================================================
# Precipitação por gravidade
# =========================================================

print("\nPrecipitação por gravidade:")
print(
    dados.groupby("classificacao_acidente")["precipitacao_mm"]
    .describe()
)


# =========================================================
# Precipitação por ano
# =========================================================

if "ano" in dados.columns:
    print("\nPrecipitação por ano:")
    print(
        dados.groupby("ano")["precipitacao_mm"]
        .describe()
    )


# =========================================================
# Precipitação por ano e gravidade
# =========================================================

if "ano" in dados.columns:
    print("\nPrecipitação por ano e gravidade:")
    print(
        dados.groupby(
            [
                "ano",
                "classificacao_acidente"
            ]
        )["precipitacao_mm"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max"
            ]
        )
    )


# =========================================================
# Base com precipitação válida
# =========================================================

dados_validos = dados[
    dados["precipitacao_mm"].notna()
].copy()

dados_validos["chuva_inmet"] = (
    dados_validos["precipitacao_mm"] > 0
)


# =========================================================
# Acidentes com chuva registrada
# =========================================================

quantidade_com_chuva = (
    dados_validos["chuva_inmet"]
    .sum()
)

percentual_com_chuva = (
    quantidade_com_chuva
    / len(dados_validos)
    * 100
    if len(dados_validos) > 0
    else 0
)

print("\nAcidentes com precipitação maior que 0 mm:")
print(
    f"{quantidade_com_chuva} "
    f"({percentual_com_chuva:.2f}% "
    "dos acidentes com precipitação válida)"
)


# =========================================================
# Chuva registrada por ano
# =========================================================

if (
    "ano" in dados_validos.columns
    and not dados_validos.empty
):
    print("\nAcidentes com precipitação > 0 mm por ano:")

    resumo_chuva = (
        dados_validos
        .groupby("ano")["chuva_inmet"]
        .agg(
            [
                "count",
                "sum"
            ]
        )
    )

    resumo_chuva["percentual"] = (
        resumo_chuva["sum"]
        / resumo_chuva["count"]
        * 100
    )

    resumo_chuva = resumo_chuva.rename(
        columns={
            "count": "precipitacao_valida",
            "sum": "com_chuva"
        }
    )

    print(resumo_chuva)


# =========================================================
# Comparação PRF x INMET
# =========================================================

if "condicao_metereologica" in dados.columns:

    print()
    print("=" * 60)
    print("COMPARAÇÃO PRF x INMET")
    print("=" * 60)

    print("\nCondições meteorológicas registradas pela PRF:")
    print(
        dados["condicao_metereologica"]
        .value_counts(dropna=False)
    )

    dados_validos["chuva_inmet_texto"] = (
        dados_validos["chuva_inmet"]
        .map(
            {
                True: "Com chuva",
                False: "Sem chuva"
            }
        )
    )

    print(
        "\nCondição meteorológica da PRF x "
        "chuva registrada pelo INMET:"
    )

    tabela_prf_inmet = pd.crosstab(
        dados_validos["condicao_metereologica"]
        .fillna("Ausente"),
        dados_validos["chuva_inmet_texto"]
    )

    print(tabela_prf_inmet)

    print(
        "\nPercentual de registros do INMET "
        "dentro de cada condição da PRF:"
    )

    percentual_prf_inmet = (
        pd.crosstab(
            dados_validos["condicao_metereologica"]
            .fillna("Ausente"),
            dados_validos["chuva_inmet_texto"],
            normalize="index"
        )
        .mul(100)
        .round(2)
    )

    print(percentual_prf_inmet)

    print(
        "\nPrecipitação por condição "
        "meteorológica registrada pela PRF:"
    )

    print(
        dados_validos.groupby(
            "condicao_metereologica"
        )["precipitacao_mm"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "max"
            ]
        )
        .sort_values(
            "mean",
            ascending=False
        )
    )


# =========================================================
# Avaliação de limites de distância
# =========================================================

print()
print("=" * 60)
print("AVALIAÇÃO DA DISTÂNCIA ATÉ A ESTAÇÃO")
print("=" * 60)

limites_distancia = [
    10,
    20,
    30,
    40,
    50,
    60
]

resumo_distancias = []

for limite in limites_distancia:

    dentro_limite = (
        dados["distancia_estacao_km"]
        <= limite
    )

    quantidade = dentro_limite.sum()

    percentual = (
        quantidade
        / len(dados)
        * 100
    )

    resumo_distancias.append(
        {
            "limite_km": limite,
            "acidentes": quantidade,
            "percentual": percentual
        }
    )

resumo_distancias = pd.DataFrame(
    resumo_distancias
)

print(
    "\nAcidentes mantidos para "
    "diferentes limites de distância:"
)

print(
    resumo_distancias.to_string(
        index=False,
        formatters={
            "percentual":
            lambda x: f"{x:.2f}%"
        }
    )
)


# =========================================================
# Qualidade PRF x INMET por limite de distância
# =========================================================

if "condicao_metereologica" in dados_validos.columns:

    print()
    print("=" * 60)
    print("QUALIDADE PRF x INMET POR LIMITE DE DISTÂNCIA")
    print("=" * 60)

    # Para a PRF, consideramos presença de chuva apenas quando
    # a condição registrada é Chuva ou Garoa/Chuvisco.
    categorias_chuva_prf = [
        "Chuva",
        "Garoa/Chuvisco"
    ]

    # "Ignorado" não deve ser interpretado como ausência de chuva.
    categorias_sem_chuva_prf = [
        "Céu Claro",
        "Nublado",
        "Sol",
        "Nevoeiro/Neblina",
        "Vento"
    ]

    dados_validos["chuva_prf"] = (
        dados_validos["condicao_metereologica"]
        .isin(categorias_chuva_prf)
    )

    dados_validos["sem_chuva_prf"] = (
        dados_validos["condicao_metereologica"]
        .isin(categorias_sem_chuva_prf)
    )

    resultados_limites = []

    for limite in limites_distancia:

        recorte = dados_validos[
            dados_validos["distancia_estacao_km"]
            <= limite
        ].copy()

        total_valido = len(recorte)

        quantidade_prf_chuva = (
            recorte["chuva_prf"]
            .sum()
        )

        quantidade_prf_sem_chuva = (
            recorte["sem_chuva_prf"]
            .sum()
        )

        ambos_chuva = (
            recorte["chuva_prf"]
            & recorte["chuva_inmet"]
        ).sum()

        inmet_chuva_prf_sem_chuva = (
            recorte["sem_chuva_prf"]
            & recorte["chuva_inmet"]
        ).sum()

        percentual_inmet_quando_prf_chuva = (
            ambos_chuva
            / quantidade_prf_chuva
            * 100
            if quantidade_prf_chuva > 0
            else 0
        )

        percentual_inmet_quando_prf_sem_chuva = (
            inmet_chuva_prf_sem_chuva
            / quantidade_prf_sem_chuva
            * 100
            if quantidade_prf_sem_chuva > 0
            else 0
        )

        fatais = (
            recorte["classificacao_acidente"]
            == "Com Vítimas Fatais"
        )

        quantidade_fatais = fatais.sum()

        fatais_com_chuva = (
            fatais
            & recorte["chuva_inmet"]
        ).sum()

        resultados_limites.append(
            {
                "limite_km": limite,
                "total_precipitacao_valida": total_valido,
                "prf_chuva": quantidade_prf_chuva,
                "prf_chuva_e_inmet_chuva": ambos_chuva,
                "%_inmet_chuva_quando_prf_chuva":
                    percentual_inmet_quando_prf_chuva,
                "prf_sem_chuva": quantidade_prf_sem_chuva,
                "inmet_chuva_quando_prf_sem_chuva":
                    inmet_chuva_prf_sem_chuva,
                "%_inmet_chuva_quando_prf_sem_chuva":
                    percentual_inmet_quando_prf_sem_chuva,
                "fatais": quantidade_fatais,
                "fatais_com_chuva": fatais_com_chuva
            }
        )

    resultados_limites = pd.DataFrame(
        resultados_limites
    )

    print(
        "\nComparação da qualidade da integração "
        "para diferentes limites:"
    )

    print(
        resultados_limites.to_string(
            index=False,
            formatters={
                "%_inmet_chuva_quando_prf_chuva":
                    lambda x: f"{x:.2f}%",
                "%_inmet_chuva_quando_prf_sem_chuva":
                    lambda x: f"{x:.2f}%"
            }
        )
    )


# =========================================================
# Valores extremos de precipitação
# =========================================================

print()
print("=" * 60)
print("MAIORES VALORES DE PRECIPITAÇÃO")
print("=" * 60)

colunas_extremos = [
    coluna
    for coluna in [
        "id",
        "ano",
        "data_inversa",
        "horario",
        "municipio",
        "classificacao_acidente",
        "condicao_metereologica",
        "estacao_inmet",
        "data_hora_medicao",
        "precipitacao_mm",
        "distancia_estacao_km",
        "diferenca_tempo_min"
    ]
    if coluna in dados_validos.columns
]

print("\n10 maiores valores de precipitação:")

print(
    dados_validos
    .nlargest(
        10,
        "precipitacao_mm"
    )[
        colunas_extremos
    ]
    .to_string(
        index=False
    )
)


# =========================================================
# Colunas relacionadas à estação
# =========================================================

print()
print("=" * 60)
print("COLUNAS RELACIONADAS À ESTAÇÃO")
print("=" * 60)

colunas_estacao = [
    coluna
    for coluna in dados.columns
    if "estacao" in coluna.lower()
]

print(colunas_estacao)