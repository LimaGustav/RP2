import pandas as pd


PERCENTIS = [
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
]


def imprimir_resumo_geral(dados):
    print()
    print("=" * 60)
    print("ANÁLISE DESCRITIVA")
    print("=" * 60)

    print("\nQuantidade total de acidentes:")
    print(len(dados))

    print("\nAcidentes por ano:")
    print(
        dados["ano"]
        .value_counts()
        .sort_index()
    )

    print("\nGravidade:")
    print(
        dados["classificacao_acidente"]
        .value_counts(dropna=False)
    )

    print("\nGravidade por ano:")
    print(
        pd.crosstab(
            dados["ano"],
            dados["classificacao_acidente"],
        )
    )

    print("\nStatus da integração:")
    print(
        dados["status_integracao"]
        .value_counts(dropna=False)
    )

    print("\nStatus da integração por ano:")
    print(
        pd.crosstab(
            dados["ano"],
            dados["status_integracao"],
        )
    )


def imprimir_precipitacao(dados):
    print("\nPrecipitação:")
    print(
        dados["precipitacao_mm"]
        .describe(percentiles=PERCENTIS)
    )

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

    print("\nPrecipitação ausente por ano:")

    resumo_ausentes = (
        dados.groupby("ano")["precipitacao_mm"]
        .agg(
            total="size",
            ausentes=lambda x: x.isna().sum(),
        )
    )

    resumo_ausentes["percentual_ausente"] = (
        resumo_ausentes["ausentes"]
        / resumo_ausentes["total"]
        * 100
    )

    print(resumo_ausentes)


def imprimir_correspondencia(dados):
    print("\nDistância até estação:")
    print(
        dados["distancia_estacao_km"]
        .describe(percentiles=PERCENTIS)
    )

    print("\nDistância até estação por ano:")
    print(
        dados.groupby("ano")[
            "distancia_estacao_km"
        ].describe()
    )

    print("\nDiferença temporal:")
    print(
        dados["diferenca_tempo_min"]
        .describe(percentiles=PERCENTIS)
    )

    print("\nDiferença temporal por ano:")
    print(
        dados.groupby("ano")[
            "diferenca_tempo_min"
        ].describe()
    )


def imprimir_precipitacao_por_grupos(dados):
    print("\nPrecipitação por gravidade:")
    print(
        dados.groupby(
            "classificacao_acidente"
        )["precipitacao_mm"]
        .describe()
    )

    print("\nPrecipitação por ano:")
    print(
        dados.groupby(
            "ano"
        )["precipitacao_mm"]
        .describe()
    )

    print("\nPrecipitação por ano e gravidade:")
    print(
        dados.groupby(
            [
                "ano",
                "classificacao_acidente",
            ]
        )["precipitacao_mm"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
    )


def preparar_dados_validos(dados):
    dados_validos = dados[
        dados["precipitacao_mm"].notna()
    ].copy()

    dados_validos["chuva_inmet"] = (
        dados_validos["precipitacao_mm"] > 0
    )

    return dados_validos


def imprimir_resumo_chuva(dados_validos):
    quantidade_com_chuva = (
        dados_validos["chuva_inmet"].sum()
    )

    percentual_com_chuva = (
        quantidade_com_chuva
        / len(dados_validos)
        * 100
        if len(dados_validos) > 0
        else 0
    )

    print(
        "\nAcidentes com precipitação "
        "maior que 0 mm:"
    )

    print(
        f"{quantidade_com_chuva} "
        f"({percentual_com_chuva:.2f}% "
        "dos acidentes com precipitação válida)"
    )

    print(
        "\nAcidentes com precipitação "
        "> 0 mm por ano:"
    )

    resumo = (
        dados_validos
        .groupby("ano")["chuva_inmet"]
        .agg(["count", "sum"])
        .rename(
            columns={
                "count": "precipitacao_valida",
                "sum": "com_chuva",
            }
        )
    )

    resumo["percentual"] = (
        resumo["com_chuva"]
        / resumo["precipitacao_valida"]
        * 100
    )

    print(resumo)


def executar_analise_descritiva(dados):
    imprimir_resumo_geral(dados)
    imprimir_precipitacao(dados)
    imprimir_correspondencia(dados)
    imprimir_precipitacao_por_grupos(dados)

    dados_validos = preparar_dados_validos(
        dados
    )

    imprimir_resumo_chuva(dados_validos)

    return dados_validos