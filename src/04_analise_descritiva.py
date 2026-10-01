# =========================================================
# Condição meteorológica registrada pela PRF
# =========================================================

if "condicao_metereologica" in dados.columns:

    print()
    print("=" * 60)
    print("COMPARAÇÃO PRF x INMET")
    print("=" * 60)

    print(
        "\nCondições meteorológicas registradas pela PRF:"
    )

    print(
        dados[
            "condicao_metereologica"
        ]
        .value_counts(
            dropna=False
        )
    )


# =========================================================
# Presença de chuva no INMET
# =========================================================

if (
    "condicao_metereologica" in dados_validos.columns
    and not dados_validos.empty
):

    dados_validos[
        "chuva_inmet"
    ] = (
        dados_validos[
            "precipitacao_mm"
        ]
        .gt(0)
        .map(
            {
                True: "Com chuva",
                False: "Sem chuva"
            }
        )
    )


    # -----------------------------------------------------
    # PRF x INMET - quantidades
    # -----------------------------------------------------

    print(
        "\nCondição meteorológica da PRF x "
        "chuva registrada pelo INMET:"
    )

    tabela_prf_inmet = pd.crosstab(
        dados_validos[
            "condicao_metereologica"
        ].fillna(
            "Ausente"
        ),
        dados_validos[
            "chuva_inmet"
        ]
    )

    print(
        tabela_prf_inmet
    )


    # -----------------------------------------------------
    # PRF x INMET - percentuais por condição da PRF
    # -----------------------------------------------------

    print(
        "\nPercentual de registros do INMET "
        "dentro de cada condição da PRF:"
    )

    percentual_prf_inmet = (
        pd.crosstab(
            dados_validos[
                "condicao_metereologica"
            ].fillna(
                "Ausente"
            ),
            dados_validos[
                "chuva_inmet"
            ],
            normalize="index"
        )
        .mul(100)
        .round(2)
    )

    print(
        percentual_prf_inmet
    )


    # -----------------------------------------------------
    # Precipitação por condição meteorológica da PRF
    # -----------------------------------------------------

    print(
        "\nPrecipitação por condição "
        "meteorológica registrada pela PRF:"
    )

    print(
        dados_validos.groupby(
            "condicao_metereologica"
        )[
            "precipitacao_mm"
        ]
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
        dados[
            "distancia_estacao_km"
        ]
        <= limite
    )

    quantidade = (
        dentro_limite.sum()
    )

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
        "precipitacao_mm",
        "distancia_estacao_km",
        "diferenca_tempo_min"
    ]
    if coluna in dados_validos.columns
]


print(
    "\n10 maiores valores de precipitação:"
)

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