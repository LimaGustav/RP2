import pandas as pd

from utils.geografia import calcular_distancia


def encontrar_estacao_mais_proxima(
    acidente,
    estacoes
):
    melhor_estacao = None
    menor_distancia = None

    for _, estacao in estacoes.iterrows():

        distancia = calcular_distancia(
            acidente["latitude"],
            acidente["longitude"],
            estacao["latitude_estacao"],
            estacao["longitude_estacao"]
        )

        if (
            menor_distancia is None
            or distancia < menor_distancia
        ):
            menor_distancia = distancia
            melhor_estacao = estacao

    return melhor_estacao, menor_distancia


def encontrar_medicao_mais_proxima(
    acidente,
    medicoes
):
    if medicoes is None or medicoes.empty:
        return None, None

    medicoes = medicoes.dropna(
        subset=["data_hora_sp"]
    ).copy()

    if medicoes.empty:
        return None, None

    diferencas = (
        medicoes["data_hora_sp"]
        - acidente["data_hora"]
    ).abs()

    diferencas = diferencas.dropna()

    if diferencas.empty:
        return None, None

    indice = diferencas.idxmin()

    medicao = medicoes.loc[indice]

    diferenca_minutos = (
        diferencas.loc[indice]
        .total_seconds()
        / 60
    )

    return medicao, diferenca_minutos