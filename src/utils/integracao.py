import pandas as pd

from utils.geografia import calcular_distancia


def preparar_estacoes(
    inmet: pd.DataFrame,
) -> pd.DataFrame:
    """Obtém uma linha por estação meteorológica."""
    return (
        inmet[
            [
                "codigo_wmo",
                "estacao",
                "latitude",
                "longitude",
            ]
        ]
        .drop_duplicates(
            subset=["codigo_wmo"]
        )
        .dropna()
        .reset_index(drop=True)
    )


def agrupar_medicoes(
    inmet: pd.DataFrame,
) -> dict:
    """Agrupa as medições pelo código da estação."""
    return {
        codigo: grupo.copy()
        for codigo, grupo
        in inmet.groupby("codigo_wmo")
    }


def encontrar_estacao_mais_proxima(
    acidente: pd.Series,
    estacoes: pd.DataFrame,
):
    """Encontra a estação geograficamente mais próxima."""
    melhor_estacao = None
    menor_distancia = None

    for _, estacao in estacoes.iterrows():
        distancia = calcular_distancia(
            acidente["latitude"],
            acidente["longitude"],
            estacao["latitude"],
            estacao["longitude"],
        )

        if (
            menor_distancia is None
            or distancia < menor_distancia
        ):
            menor_distancia = distancia
            melhor_estacao = estacao

    return melhor_estacao, menor_distancia


def encontrar_medicao_mais_proxima(
    acidente: pd.Series,
    medicoes: pd.DataFrame,
):
    """Encontra a medição temporalmente mais próxima."""
    if medicoes is None or medicoes.empty:
        return None, None

    medicoes = medicoes.dropna(
        subset=["data_hora"]
    )

    if medicoes.empty:
        return None, None

    diferencas = (
        medicoes["data_hora"]
        - acidente["data_hora"]
    ).abs()

    diferencas = diferencas.dropna()

    if diferencas.empty:
        return None, None

    indice = diferencas.idxmin()

    medicao = medicoes.loc[indice]

    diferenca_minutos = (
        diferencas.loc[indice].total_seconds()
        / 60
    )

    return medicao, diferenca_minutos


def integrar_acidente(
    acidente: pd.Series,
    estacoes: pd.DataFrame,
    medicoes_por_estacao: dict,
) -> dict:
    """Integra um acidente à estação e medição mais próximas."""
    registro = acidente.to_dict()

    if (
        pd.isna(acidente["latitude"])
        or pd.isna(acidente["longitude"])
        or pd.isna(acidente["data_hora"])
    ):
        registro["status_integracao"] = (
            "dados_acidente_invalidos"
        )

        return registro

    estacao, distancia = (
        encontrar_estacao_mais_proxima(
            acidente,
            estacoes,
        )
    )

    if estacao is None:
        registro["status_integracao"] = (
            "estacao_nao_encontrada"
        )

        return registro

    codigo = estacao["codigo_wmo"]

    registro["codigo_wmo"] = codigo
    registro["estacao_inmet"] = estacao["estacao"]
    registro["distancia_estacao_km"] = distancia

    medicoes = medicoes_por_estacao.get(codigo)

    medicao, diferenca_minutos = (
        encontrar_medicao_mais_proxima(
            acidente,
            medicoes,
        )
    )

    if medicao is None:
        registro["status_integracao"] = (
            "sem_medicao_temporal"
        )

        return registro

    registro["data_hora_medicao"] = (
        medicao["data_hora"]
    )

    registro["diferenca_tempo_min"] = (
        diferenca_minutos
    )

    registro["precipitacao_mm"] = (
        medicao["precipitacao_mm"]
    )

    if pd.isna(medicao["precipitacao_mm"]):
        registro["status_integracao"] = (
            "precipitacao_ausente"
        )
    else:
        registro["status_integracao"] = (
            "integrado"
        )

    return registro


def integrar_bases(
    acidentes: pd.DataFrame,
    inmet: pd.DataFrame,
) -> pd.DataFrame:
    """Executa a integração espacial e temporal."""
    estacoes = preparar_estacoes(inmet)
    medicoes_por_estacao = agrupar_medicoes(inmet)

    resultados = []

    total = len(acidentes)

    for numero, (_, acidente) in enumerate(
        acidentes.iterrows(),
        start=1,
    ):
        registro = integrar_acidente(
            acidente,
            estacoes,
            medicoes_por_estacao,
        )

        resultados.append(registro)

        if numero % 100 == 0:
            print(
                f"{numero}/{total} "
                "acidentes processados"
            )

    return pd.DataFrame(resultados)