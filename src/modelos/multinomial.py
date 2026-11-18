import numpy as np
import pandas as pd
import statsmodels.api as sm

from modelos.comum import formatar_significancia


def extrair_resultado_multinomial(
    resultado,
    variavel_exposicao,
    nome_modelo,
):
    """
    Extrai os efeitos da exposição no MNLogit.

    Codificação do desfecho:
      0 = Sem Vítimas
      1 = Com Vítimas Feridas
      2 = Com Vítimas Fatais

    A categoria 0 é a referência.
    """
    resultados = []

    comparacoes = [
        (
            0,
            "Com Vítimas Feridas vs Sem Vítimas",
        ),
        (
            1,
            "Com Vítimas Fatais vs Sem Vítimas",
        ),
    ]

    for coluna_parametro, comparacao in comparacoes:
        coef = resultado.params.loc[
            variavel_exposicao,
            coluna_parametro,
        ]

        ep = resultado.bse.loc[
            variavel_exposicao,
            coluna_parametro,
        ]

        p_valor = resultado.pvalues.loc[
            variavel_exposicao,
            coluna_parametro,
        ]

        ic_inf = coef - 1.96 * ep
        ic_sup = coef + 1.96 * ep

        resultados.append(
            {
                "modelo": nome_modelo,
                "comparacao": comparacao,
                "variavel": variavel_exposicao,
                "coeficiente": coef,
                "erro_padrao": ep,
                "p_valor": p_valor,
                "sig": formatar_significancia(p_valor),
                "odds_ratio": np.exp(coef),
                "ic_95_inf": np.exp(ic_inf),
                "ic_95_sup": np.exp(ic_sup),
            }
        )

    return pd.DataFrame(resultados)


def executar_modelos_multinomiais(
    dados_modelo,
    matriz_x_m2a,
    matriz_x_m2b,
):
    """Executa a regressão multinomial como análise complementar."""
    print()
    print("=" * 72)
    print("REGRESSÃO LOGÍSTICA MULTINOMIAL COMPLEMENTAR")
    print("=" * 72)

    print("\nReferência do desfecho: Sem Vítimas.")
    print(
        "A multinomial é usada como análise complementar porque "
        "não exige a hipótese de chances proporcionais da regressão ordinal."
    )

    y_multinomial = (
        dados_modelo["gravidade_ordinal"]
        .astype(int)
    )

    resultados_multinomiais = []

    especificacoes = [
        (
            "Multinomial ajustado - Precipitação mm",
            "precipitacao_mm",
            matriz_x_m2a,
        ),
        (
            "Multinomial ajustado - Chuva binária",
            "chuva_binaria",
            matriz_x_m2b,
        ),
    ]

    for (
        nome_modelo,
        variavel_exposicao,
        matriz_ajustada,
    ) in especificacoes:
        print()
        print("-" * 72)
        print(nome_modelo.upper())
        print("-" * 72)

        x_multinomial = sm.add_constant(
            matriz_ajustada,
            has_constant="add",
        )

        modelo_multinomial = sm.MNLogit(
            y_multinomial,
            x_multinomial,
        )

        resultado_multinomial = (
            modelo_multinomial
            .fit(
                method="newton",
                disp=False,
                maxiter=1000,
            )
        )

        print(resultado_multinomial.summary())

        tabela_multinomial = (
            extrair_resultado_multinomial(
                resultado_multinomial,
                variavel_exposicao,
                nome_modelo,
            )
        )

        print("\nEfeito da precipitação:")
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
            .round(4)
            .to_string(index=False)
        )

        resultados_multinomiais.append(
            tabela_multinomial
        )

    if not resultados_multinomiais:
        return pd.DataFrame()

    return pd.concat(
        resultados_multinomiais,
        ignore_index=True,
    )
