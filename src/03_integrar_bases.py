from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]

CAMINHO_PRF = (
    RAIZ
    / "datasets"
    / "processed"
    / "acidentes_sp.csv"
)

CAMINHO_INMET = (
    RAIZ
    / "datasets"
    / "processed"
    / "inmet_sp.csv"
)

SAIDA = (
    RAIZ
    / "datasets"
    / "processed"
    / "base_integrada.csv"
)


def calcular_distancia(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Distância Haversine em quilômetros.
    """

    raio_terra = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)

    lat2 = radians(lat2)
    lon2 = radians(lon2)

    diferenca_lat = lat2 - lat1
    diferenca_lon = lon2 - lon1

    a = (
        sin(diferenca_lat / 2) ** 2
        +
        cos(lat1)
        * cos(lat2)
        * sin(diferenca_lon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return raio_terra * c


# ========================================
# Carregar bases
# ========================================

acidentes = pd.read_csv(
    CAMINHO_PRF
)

inmet = pd.read_csv(
    CAMINHO_INMET
)

acidentes["data_hora"] = pd.to_datetime(
    acidentes["data_hora"]
)

inmet["data_hora_sp"] = pd.to_datetime(
    inmet["data_hora_sp"]
)


# ========================================
# Uma linha por estação
# ========================================

estacoes = (
    inmet[
        [
            "codigo_wmo",
            "estacao",
            "latitude_estacao",
            "longitude_estacao"
        ]
    ]
    .drop_duplicates()
    .dropna()
)


# ========================================
# Encontrar estação mais próxima
# ========================================

resultados = []

for indice, acidente in acidentes.iterrows():

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

    registro = acidente.to_dict()

    if melhor_estacao is not None:

        codigo = melhor_estacao[
            "codigo_wmo"
        ]

        registro["codigo_wmo"] = codigo
        registro["estacao_inmet"] = (
            melhor_estacao["estacao"]
        )

        registro[
            "distancia_estacao_km"
        ] = menor_distancia

        # ==================================
        # Dados horários dessa estação
        # ==================================

        medicoes = inmet[
            inmet["codigo_wmo"] == codigo
        ].copy()

        # Usar somente medições com chuva válida
        medicoes = medicoes[
            medicoes["precipitacao_mm"]
            .notna()
        ].copy()

        if not medicoes.empty:

            medicoes[
                "diferenca_tempo"
            ] = (
                medicoes["data_hora_sp"]
                - acidente["data_hora"]
            ).abs()

            medicao = medicoes.loc[
                medicoes[
                    "diferenca_tempo"
                ].idxmin()
            ]

            registro[
                "data_hora_medicao"
            ] = medicao["data_hora_sp"]

            registro[
                "precipitacao_mm"
            ] = medicao["precipitacao_mm"]

            registro[
                "diferenca_tempo_min"
            ] = (
                medicao["diferenca_tempo"]
                .total_seconds()
                / 60
            )

    resultados.append(registro)

    if (indice + 1) % 100 == 0:
        print(
            f"{indice + 1} acidentes processados"
        )


base_integrada = pd.DataFrame(
    resultados
)

base_integrada.to_csv(
    SAIDA,
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    f"Base integrada criada: {SAIDA}"
)

print(
    "Registros:",
    len(base_integrada)
)

print(
    "\nDistância até estação:"
)

print(
    base_integrada[
        "distancia_estacao_km"
    ].describe(
        percentiles=[
            .25,
            .5,
            .75,
            .9,
            .95
        ]
    )
)

print(
    "\nDiferença temporal:"
)

print(
    base_integrada[
        "diferenca_tempo_min"
    ].describe(
        percentiles=[
            .25,
            .5,
            .75,
            .9,
            .95
        ]
    )
)