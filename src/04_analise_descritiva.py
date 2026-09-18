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

dados = pd.read_csv(
    CAMINHO
)


# =========================================================
# Resumo geral
# =========================================================

print()
print("=" * 60)
print("ANÁLISE DESCRITIVA")
print("=" * 60)


print(
    "\nQuantidade total de acidentes:"
)

print(
    len(dados)
)


# =========================================================
# Acidentes por ano
# =========================================================

if "ano" in dados.columns:

    print(
        "\nAcidentes por ano:"
    )

    print(
        dados[
            "ano"
        ]
        .value_counts()
        .sort_index()
    )


# =========================================================
# Gravidade geral
# =========================================================

print(
    "\nGravidade:"
)

print(
    dados[
        "classificacao_acidente"
    ]
    .value_counts(
        dropna=False
    )
)


# =========================================================
# Gravidade por ano
# =========================================================

if "ano" in dados.columns:

    print(
        "\nGravidade por ano:"
    )

    print(
        pd.crosstab(
            dados["ano"],
            dados[
                "classificacao_acidente"
            ]
        )
    )


# =========================================================
# Status da integração
# =========================================================

if "status_integracao" in dados.columns:

    print(
        "\nStatus da integração:"
    )

    print(
        dados[
            "status_integracao"
        ]
        .value_counts(
            dropna=False
        )
    )


# =========================================================
# Status da integração por ano
# =========================================================

if (
    "ano" in dados.columns
    and "status_integracao" in dados.columns
):

    print(
        "\nStatus da integração por ano:"
    )

    print(
        pd.crosstab(
            dados["ano"],
            dados[
                "status_integracao"
            ]
        )
    )


# =========================================================
# Precipitação
# =========================================================

print(
    "\nPrecipitação:"
)

print(
    dados[
        "precipitacao_mm"
    ]
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
    dados[
        "precipitacao_mm"
    ]
    .isna()
    .sum()
)

percentual_ausente = (
    quantidade_ausente
    / len(dados)
    * 100
)


print(
    "\nPrecipitação ausente:"
)

print(
    f"{quantidade_ausente} "
    f"({percentual_ausente:.2f}%)"
)


# =========================================================
# Precipitação ausente por ano
# =========================================================

if "ano" in dados.columns:

    print(
        "\nPrecipitação ausente por ano:"
    )

    resumo_ausentes = (
        dados.groupby(
            "ano"
        )[
            "precipitacao_mm"
        ]
        .agg(
            total="size",
            ausentes=lambda x:
            x.isna().sum()
        )
    )

    resumo_ausentes[
        "percentual_ausente"
    ] = (
        resumo_ausentes[
            "ausentes"
        ]
        / resumo_ausentes[
            "total"
        ]
        * 100
    )

    print(
        resumo_ausentes
    )


# =========================================================
# Distância até estação
# =========================================================

print(
    "\nDistância até estação:"
)

print(
    dados[
        "distancia_estacao_km"
    ]
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

    print(
        "\nDistância até estação por ano:"
    )

    print(
        dados.groupby(
            "ano"
        )[
            "distancia_estacao_km"
        ]
        .describe()
    )


# =========================================================
# Diferença temporal
# =========================================================

print(
    "\nDiferença temporal:"
)

print(
    dados[
        "diferenca_tempo_min"
    ]
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

    print(
        "\nDiferença temporal por ano:"
    )

    print(
        dados.groupby(
            "ano"
        )[
            "diferenca_tempo_min"
        ]
        .describe()
    )


# =========================================================
# Precipitação por gravidade
# =========================================================

print(
    "\nPrecipitação por gravidade:"
)

print(
    dados.groupby(
        "classificacao_acidente"
    )[
        "precipitacao_mm"
    ]
    .describe()
)


# =========================================================
# Precipitação por ano
# =========================================================

if "ano" in dados.columns:

    print(
        "\nPrecipitação por ano:"
    )

    print(
        dados.groupby(
            "ano"
        )[
            "precipitacao_mm"
        ]
        .describe()
    )


# =========================================================
# Precipitação por ano e gravidade
# =========================================================

if "ano" in dados.columns:

    print(
        "\nPrecipitação por ano e gravidade:"
    )

    print(
        dados.groupby(
            [
                "ano",
                "classificacao_acidente"
            ]
        )[
            "precipitacao_mm"
        ]
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
# Acidentes com chuva registrada
# =========================================================

dados_validos = dados[
    dados[
        "precipitacao_mm"
    ].notna()
].copy()


quantidade_com_chuva = (
    dados_validos[
        "precipitacao_mm"
    ]
    .gt(0)
    .sum()
)


percentual_com_chuva = (
    quantidade_com_chuva
    / len(dados_validos)
    * 100
    if len(dados_validos) > 0
    else 0
)


print(
    "\nAcidentes com precipitação maior que 0 mm:"
)

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

    print(
        "\nAcidentes com precipitação > 0 mm por ano:"
    )

    resumo_chuva = (
        dados_validos
        .assign(
            chuva=(
                dados_validos[
                    "precipitacao_mm"
                ] > 0
            )
        )
        .groupby(
            "ano"
        )[
            "chuva"
        ]
        .agg(
            [
                "count",
                "sum"
            ]
        )
    )

    resumo_chuva[
        "percentual"
    ] = (
        resumo_chuva[
            "sum"
        ]
        / resumo_chuva[
            "count"
        ]
        * 100
    )

    resumo_chuva = (
        resumo_chuva.rename(
            columns={
                "count": "precipitacao_valida",
                "sum": "com_chuva"
            }
        )
    )

    print(
        resumo_chuva
    )