import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from modelos.comum import formatar_significancia
from modelos.preparacao import (
    criar_matriz_ajustada,
    preparar_dados_modelagem,
)


def extrair_tabela_modelo_ordinal(
    resultado,
    nome_modelo,
):
    """Extrai coeficientes, OR, IC95% e valor-p do OrderedModel."""
    parametros = resultado.params
    erros = resultado.bse
    p_valores = resultado.pvalues
    intervalos = resultado.conf_int()

    linhas = []

    for variavel in parametros.index:
        if "/" in str(variavel) or str(variavel).startswith("cut"):
            continue

        coef = parametros[variavel]
        ep = erros.get(variavel, np.nan)
        p_val = p_valores.get(variavel, np.nan)

        if variavel in intervalos.index:
            ic_inf = intervalos.loc[variavel, 0]
            ic_sup = intervalos.loc[variavel, 1]
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
                "sig": formatar_significancia(p_val),
                "odds_ratio": np.exp(coef),
                "ic_95_inf": np.exp(ic_inf),
                "ic_95_sup": np.exp(ic_sup),
            }
        )

    return pd.DataFrame(linhas)


def calcular_metricas(
    resultado,
    nome_modelo,
    n_obs,
):
    """Retorna métricas globais de ajuste do modelo."""
    return {
        "modelo": nome_modelo,
        "n_obs": n_obs,
        "log_likelihood": resultado.llf,
        "aic": resultado.aic,
        "bic": getattr(resultado, "bic", np.nan),
    }


def ajustar_modelo_ordinal(
    y,
    matriz_x,
    nome_modelo,
):
    """Ajusta uma regressão logística ordinal por máxima verossimilhança."""
    modelo = OrderedModel(
        y,
        matriz_x,
        distr="logit",
    )

    resultado = modelo.fit(
        method="bfgs",
        disp=False,
        maxiter=1500,
    )

    tabela = extrair_tabela_modelo_ordinal(
        resultado,
        nome_modelo,
    )

    return resultado, tabela


def _executar_e_imprimir_ordinal(
    y_ordinal,
    dados_modelo,
    nome_modelo,
    matriz_x,
):
    print()
    print("-" * 72)
    print(nome_modelo.upper())
    print("-" * 72)

    resultado, tabela = ajustar_modelo_ordinal(
        y_ordinal,
        matriz_x,
        nome_modelo,
    )

    print(resultado.summary())

    print("\nOdds Ratios:")
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
        ].to_string(index=False)
    )

    metricas = calcular_metricas(
        resultado,
        nome_modelo,
        len(dados_modelo),
    )

    return resultado, tabela, metricas


def executar_modelos_ordinais(dados_modelo):
    """Executa os quatro modelos ordinais principal/bruto/ajustado."""
    print()
    print("=" * 72)
    print("REGRESSÃO LOGÍSTICA ORDINAL")
    print("=" * 72)

    y_ordinal = dados_modelo["gravidade_ordinal"]

    matrizes = {
        "m1a": dados_modelo[["precipitacao_mm"]].astype(float),
        "m1b": dados_modelo[["chuva_binaria"]].astype(float),
        "m2a": criar_matriz_ajustada(
            dados_modelo,
            "precipitacao_mm",
        ),
        "m2b": criar_matriz_ajustada(
            dados_modelo,
            "chuva_binaria",
        ),
    }

    especificacoes = [
        (
            "m1a",
            "Modelo 1A - Bruto - Precipitação mm",
        ),
        (
            "m1b",
            "Modelo 1B - Bruto - Chuva binária",
        ),
        (
            "m2a",
            "Modelo 2A - Ajustado - Precipitação mm",
        ),
        (
            "m2b",
            "Modelo 2B - Ajustado - Chuva binária",
        ),
    ]

    resultados = {}
    tabelas = []
    metricas = []

    for chave, nome_modelo in especificacoes:
        resultado, tabela, metrica = (
            _executar_e_imprimir_ordinal(
                y_ordinal,
                dados_modelo,
                nome_modelo,
                matrizes[chave],
            )
        )

        resultados[chave] = resultado
        tabelas.append(tabela)
        metricas.append(metrica)

    return {
        "resultados": resultados,
        "tabela": pd.concat(
            tabelas,
            ignore_index=True,
        ),
        "metricas": pd.DataFrame(metricas),
        "matrizes": matrizes,
    }


def executar_sensibilidade_espacial(
    dados,
    limites_km,
    limite_tempo_min,
):
    """Repete os modelos ajustados para diferentes limites espaciais."""
    print()
    print("=" * 72)
    print("ANÁLISE DE SENSIBILIDADE DO LIMITE ESPACIAL")
    print("=" * 72)

    sensibilidade = []

    for limite in limites_km:
        dados_sensibilidade = preparar_dados_modelagem(
            dados,
            limite,
            limite_tempo_min,
        )

        if dados_sensibilidade.empty:
            continue

        y_sensibilidade = (
            dados_sensibilidade["gravidade_ordinal"]
        )

        x_mm = criar_matriz_ajustada(
            dados_sensibilidade,
            "precipitacao_mm",
        )

        x_bin = criar_matriz_ajustada(
            dados_sensibilidade,
            "chuva_binaria",
        )

        resultado_mm, _ = ajustar_modelo_ordinal(
            y_sensibilidade,
            x_mm,
            f"Sensibilidade {limite:.0f} km - mm",
        )

        resultado_bin, _ = ajustar_modelo_ordinal(
            y_sensibilidade,
            x_bin,
            f"Sensibilidade {limite:.0f} km - binária",
        )

        for nome_variavel, modelo in [
            ("precipitacao_mm", resultado_mm),
            ("chuva_binaria", resultado_bin),
        ]:
            coef = modelo.params[nome_variavel]
            p_val = modelo.pvalues[nome_variavel]
            ic = modelo.conf_int().loc[nome_variavel]

            sensibilidade.append(
                {
                    "limite_km": limite,
                    "n": len(dados_sensibilidade),
                    "exposicao": nome_variavel,
                    "odds_ratio": np.exp(coef),
                    "ic_95_inf": np.exp(ic.iloc[0]),
                    "ic_95_sup": np.exp(ic.iloc[1]),
                    "p_valor": p_val,
                    "sig": formatar_significancia(p_val),
                }
            )

    tabela = pd.DataFrame(sensibilidade)

    print(
        tabela
        .round(4)
        .to_string(index=False)
    )

    return tabela
