import pandas as pd


LIMITES_DISTANCIA = [
    10,
    20,
    30,
    40,
    50,
    60,
]


def comparar_prf_inmet(
    dados,
    dados_validos,
):
    print()
    print("=" * 60)
    print("COMPARAÇÃO PRF x INMET")
    print("=" * 60)

    print(
        "\nCondições meteorológicas "
        "registradas pela PRF:"
    )

    print(
        dados["condicao_metereologica"]
        .value_counts(dropna=False)
    )

    dados_validos["chuva_inmet_texto"] = (
        dados_validos["chuva_inmet"]
        .map(
            {
                True: "Com chuva",
                False: "Sem chuva",
            }
        )
    )

    print(
        "\nCondição meteorológica da PRF x "
        "chuva registrada pelo INMET:"
    )

    tabela = pd.crosstab(
        dados_validos[
            "condicao_metereologica"
        ].fillna("Ausente"),
        dados_validos[
            "chuva_inmet_texto"
        ],
    )

    print(tabela)

    print(
        "\nPercentual de registros do INMET "
        "dentro de cada condição da PRF:"
    )

    percentual = (
        pd.crosstab(
            dados_validos[
                "condicao_metereologica"
            ].fillna("Ausente"),
            dados_validos[
                "chuva_inmet_texto"
            ],
            normalize="index",
        )
        .mul(100)
        .round(2)
    )

    print(percentual)

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
                "max",
            ]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
    )


def avaliar_distancias(dados):
    print()
    print("=" * 60)
    print("AVALIAÇÃO DA DISTÂNCIA ATÉ A ESTAÇÃO")
    print("=" * 60)

    resultados = []

    for limite in LIMITES_DISTANCIA:
        quantidade = (
            dados["distancia_estacao_km"]
            .le(limite)
            .sum()
        )

        resultados.append(
            {
                "limite_km": limite,
                "acidentes": quantidade,
                "percentual": (
                    quantidade
                    / len(dados)
                    * 100
                ),
            }
        )

    resumo = pd.DataFrame(resultados)

    print(
        "\nAcidentes mantidos para "
        "diferentes limites de distância:"
    )

    print(
        resumo.to_string(
            index=False,
            formatters={
                "percentual":
                    lambda x: f"{x:.2f}%"
            },
        )
    )


def avaliar_qualidade_por_distancia(
    dados_validos,
):
    print()
    print("=" * 60)
    print(
        "QUALIDADE PRF x INMET "
        "POR LIMITE DE DISTÂNCIA"
    )
    print("=" * 60)

    dados_validos = dados_validos.copy()

    categorias_chuva = [
        "Chuva",
        "Garoa/Chuvisco",
    ]

    categorias_sem_chuva = [
        "Céu Claro",
        "Nublado",
        "Sol",
        "Nevoeiro/Neblina",
        "Vento",
    ]

    dados_validos["chuva_prf"] = (
        dados_validos[
            "condicao_metereologica"
        ].isin(categorias_chuva)
    )

    dados_validos["sem_chuva_prf"] = (
        dados_validos[
            "condicao_metereologica"
        ].isin(categorias_sem_chuva)
    )

    resultados = []

    for limite in LIMITES_DISTANCIA:
        recorte = dados_validos[
            dados_validos[
                "distancia_estacao_km"
            ] <= limite
        ]

        prf_chuva = (
            recorte["chuva_prf"].sum()
        )

        prf_sem_chuva = (
            recorte["sem_chuva_prf"].sum()
        )

        ambos_chuva = (
            recorte["chuva_prf"]
            & recorte["chuva_inmet"]
        ).sum()

        inmet_chuva_sem_prf = (
            recorte["sem_chuva_prf"]
            & recorte["chuva_inmet"]
        ).sum()

        fatais = (
            recorte[
                "classificacao_acidente"
            ]
            == "Com Vítimas Fatais"
        )

        resultados.append(
            {
                "limite_km": limite,
                "total_precipitacao_valida":
                    len(recorte),

                "prf_chuva":
                    prf_chuva,

                "prf_chuva_e_inmet_chuva":
                    ambos_chuva,

                "%_inmet_chuva_quando_prf_chuva":
                    (
                        ambos_chuva
                        / prf_chuva
                        * 100
                        if prf_chuva
                        else 0
                    ),

                "prf_sem_chuva":
                    prf_sem_chuva,

                "inmet_chuva_quando_prf_sem_chuva":
                    inmet_chuva_sem_prf,

                "%_inmet_chuva_quando_prf_sem_chuva":
                    (
                        inmet_chuva_sem_prf
                        / prf_sem_chuva
                        * 100
                        if prf_sem_chuva
                        else 0
                    ),

                "fatais":
                    fatais.sum(),

                "fatais_com_chuva":
                    (
                        fatais
                        & recorte["chuva_inmet"]
                    ).sum(),
            }
        )

    resultado = pd.DataFrame(
        resultados
    )

    print(
        "\nComparação da qualidade da integração "
        "para diferentes limites:"
    )

    print(
        resultado.to_string(
            index=False,
            formatters={
                "%_inmet_chuva_quando_prf_chuva":
                    lambda x: f"{x:.2f}%",

                "%_inmet_chuva_quando_prf_sem_chuva":
                    lambda x: f"{x:.2f}%",
            },
        )
    )


def mostrar_extremos(dados_validos):
    print()
    print("=" * 60)
    print("MAIORES VALORES DE PRECIPITAÇÃO")
    print("=" * 60)

    colunas = [
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
            "diferenca_tempo_min",
        ]
        if coluna in dados_validos.columns
    ]

    print(
        "\n10 maiores valores "
        "de precipitação:"
    )

    print(
        dados_validos
        .nlargest(
            10,
            "precipitacao_mm",
        )[colunas]
        .to_string(index=False)
    )


def mostrar_colunas_estacao(dados):
    print()
    print("=" * 60)
    print("COLUNAS RELACIONADAS À ESTAÇÃO")
    print("=" * 60)

    colunas = [
        coluna
        for coluna in dados.columns
        if "estacao" in coluna.lower()
    ]

    print(colunas)


def executar_analise_qualidade(
    dados,
    dados_validos,
):
    comparar_prf_inmet(
        dados,
        dados_validos,
    )

    avaliar_distancias(dados)

    avaliar_qualidade_por_distancia(
        dados_validos
    )

    mostrar_extremos(
        dados_validos
    )

    mostrar_colunas_estacao(dados)