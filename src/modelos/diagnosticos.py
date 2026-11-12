import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from modelos.comum import formatar_significancia


def _imprimir_ausencia_por(
    dados,
    coluna,
):
    if coluna not in dados.columns:
        return

    tabela = (
        dados
        .groupby(
            coluna,
            dropna=False,
        )["precipitacao_ausente"]
        .agg(
            total="size",
            ausentes="sum",
        )
        .reset_index()
    )

    tabela["percentual_ausente"] = (
        tabela["ausentes"]
        / tabela["total"]
        * 100
    )

    print()
    print(f"Ausência por {coluna}:")
    print(
        tabela.to_string(
            index=False,
            formatters={
                "percentual_ausente":
                    lambda x: f"{x:.2f}%"
            },
        )
    )


def diagnosticar_ausencia_precipitacao(
    dados_entrada,
    min_registros_estacao=20,
):
    """Descreve e modela a probabilidade de precipitação ausente."""
    print()
    print("=" * 72)
    print("DIAGNÓSTICO DOS DADOS AUSENTES DE PRECIPITAÇÃO")
    print("=" * 72)

    dados = dados_entrada.copy()

    dados["precipitacao_ausente"] = (
        dados["precipitacao_mm"]
        .isna()
        .astype(int)
    )

    total_ausente = int(
        dados["precipitacao_ausente"].sum()
    )

    print(
        f"\nPrecipitação ausente: "
        f"{total_ausente:,} "
        f"({total_ausente / len(dados) * 100:.2f}%)"
    )

    for coluna in [
        "ano",
        "mes",
        "classificacao_acidente",
        "condicao_metereologica",
    ]:
        _imprimir_ausencia_por(
            dados,
            coluna,
        )

    dados["faixa_distancia"] = pd.cut(
        dados["distancia_estacao_km"],
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
        include_lowest=True,
    )

    _imprimir_ausencia_por(
        dados,
        "faixa_distancia",
    )

    if "estacao_inmet" in dados.columns:
        tabela_estacoes = (
            dados
            .groupby(
                "estacao_inmet",
                dropna=False,
            )["precipitacao_ausente"]
            .agg(
                total="size",
                ausentes="sum",
            )
            .reset_index()
        )

        tabela_estacoes["percentual_ausente"] = (
            tabela_estacoes["ausentes"]
            / tabela_estacoes["total"]
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
                ],
            )
        )

        print()
        print(
            "15 estações com maior percentual de ausência:"
        )
        print(
            tabela_estacoes
            .head(15)
            .to_string(
                index=False,
                formatters={
                    "percentual_ausente":
                        lambda x: f"{x:.2f}%"
                },
            )
        )

    print()
    print("-" * 72)
    print("MODELO DA PROBABILIDADE DE PRECIPITAÇÃO AUSENTE")
    print("-" * 72)

    dados_ausencia = dados.copy()

    dados_ausencia["condicao_metereologica"] = (
        dados_ausencia["condicao_metereologica"]
        .fillna("Ausente")
    )

    if "estacao_inmet" in dados_ausencia.columns:
        dados_ausencia["estacao_inmet"] = (
            dados_ausencia["estacao_inmet"]
            .fillna("SEM_ESTACAO")
            .astype(str)
        )

        contagem_estacoes = (
            dados_ausencia["estacao_inmet"]
            .value_counts()
        )

        estacoes_raras = (
            contagem_estacoes[
                contagem_estacoes
                < min_registros_estacao
            ]
            .index
        )

        dados_ausencia["estacao_modelo_ausencia"] = (
            dados_ausencia["estacao_inmet"]
            .where(
                ~dados_ausencia["estacao_inmet"]
                .isin(estacoes_raras),
                "OUTRAS_ESTACOES",
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
                family=sm.families.Binomial(),
            )
            .fit(maxiter=500)
        )

        resultado_ausencia = pd.DataFrame(
            {
                "coeficiente": modelo_ausencia.params,
                "erro_padrao": modelo_ausencia.bse,
                "odds_ratio": np.exp(
                    modelo_ausencia.params
                ),
                "p_valor": modelo_ausencia.pvalues,
            }
        )

        ic_ausencia = modelo_ausencia.conf_int()

        resultado_ausencia["ic_95_inf"] = np.exp(
            ic_ausencia[0]
        )
        resultado_ausencia["ic_95_sup"] = np.exp(
            ic_ausencia[1]
        )
        resultado_ausencia["sig"] = (
            resultado_ausencia["p_valor"]
            .apply(formatar_significancia)
        )

        linhas_gravidade = [
            indice
            for indice in resultado_ausencia.index
            if "classificacao_acidente" in indice
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
                ],
            ].round(4)
        )

        gravidade_associada_ausencia = any(
            resultado_ausencia.loc[
                linhas_gravidade,
                "p_valor",
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

        return resultado_ausencia

    except Exception as erro:
        print(
            "\nNão foi possível ajustar o modelo de ausência:"
        )
        print(erro)

        return pd.DataFrame()


def diagnosticar_chances_proporcionais(
    dados_modelo,
    matriz_x_m2a,
    resultado_m2a,
):
    """Compara coeficientes em dois cortes cumulativos de forma exploratória."""
    print()
    print("=" * 72)
    print("DIAGNÓSTICO DA SUPOSIÇÃO DE CHANCES PROPORCIONAIS")
    print("=" * 72)

    print(
        "\nDiagnóstico exploratório: comparação dos coeficientes "
        "em dois cortes cumulativos. Não substitui um teste formal "
        "de proportional odds."
    )

    gravidade_num = (
        dados_modelo["gravidade_ordinal"]
        .astype(int)
    )

    corte_1 = (gravidade_num >= 1).astype(int)
    corte_2 = (gravidade_num >= 2).astype(int)

    matriz_x_const = sm.add_constant(
        matriz_x_m2a,
        has_constant="add",
    )

    logit_corte1 = (
        sm.Logit(
            corte_1,
            matriz_x_const,
        )
        .fit(
            disp=False,
            maxiter=1000,
        )
    )

    logit_corte2 = (
        sm.Logit(
            corte_2,
            matriz_x_const,
        )
        .fit(
            disp=False,
            maxiter=1000,
        )
    )

    comparacao_cortes = []

    for variavel in [
        variavel
        for variavel in logit_corte1.params.index
        if variavel != "const"
    ]:
        b1 = logit_corte1.params.get(
            variavel,
            np.nan,
        )
        b2 = logit_corte2.params.get(
            variavel,
            np.nan,
        )
        bord = resultado_m2a.params.get(
            variavel,
            np.nan,
        )

        comparacao_cortes.append(
            {
                "variavel": variavel,
                "beta_corte1": b1,
                "or_corte1": np.exp(b1),
                "p_corte1": (
                    logit_corte1
                    .pvalues
                    .get(
                        variavel,
                        np.nan,
                    )
                ),
                "beta_corte2": b2,
                "or_corte2": np.exp(b2),
                "p_corte2": (
                    logit_corte2
                    .pvalues
                    .get(
                        variavel,
                        np.nan,
                    )
                ),
                "beta_ordinal": bord,
                "or_ordinal": (
                    np.exp(bord)
                    if not np.isnan(bord)
                    else np.nan
                ),
            }
        )

    tabela = pd.DataFrame(
        comparacao_cortes
    )

    print(
        tabela
        .round(4)
        .to_string(index=False)
    )

    return tabela
