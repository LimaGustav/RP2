from pathlib import Path

import pandas as pd

from utils.integracao import (
    encontrar_estacao_mais_proxima,
    encontrar_medicao_mais_proxima
)

from utils.validacao import preparar_tipos


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


acidentes = pd.read_csv(CAMINHO_PRF)
inmet = pd.read_csv(CAMINHO_INMET)

acidentes, inmet = preparar_tipos(
    acidentes,
    inmet
)


estacoes = (
    inmet[
        [
            "codigo_wmo",
            "estacao",
            "latitude_estacao",
            "longitude_estacao"
        ]
    ]
    .drop_duplicates(
        subset=["codigo_wmo"]
    )
    .dropna()
)


medicoes_por_estacao = {
    codigo: grupo.copy()
    for codigo, grupo
    in inmet.groupby("codigo_wmo")
}


resultados = []


for numero, (_, acidente) in enumerate(
    acidentes.iterrows(),
    start=1
):

    registro = acidente.to_dict()

    if (
        pd.isna(acidente["latitude"])
        or pd.isna(acidente["longitude"])
        or pd.isna(acidente["data_hora"])
    ):
        registro["status_integracao"] = (
            "dados_acidente_invalidos"
        )

        resultados.append(registro)
        continue


    estacao, distancia = (
        encontrar_estacao_mais_proxima(
            acidente,
            estacoes
        )
    )

    if estacao is None:
        registro["status_integracao"] = (
            "estacao_nao_encontrada"
        )

        resultados.append(registro)
        continue


    codigo = estacao["codigo_wmo"]

    registro["codigo_wmo"] = codigo
    registro["estacao_inmet"] = (
        estacao["estacao"]
    )
    registro["distancia_estacao_km"] = (
        distancia
    )


    medicoes = medicoes_por_estacao.get(
        codigo
    )

    medicao, diferenca_minutos = (
        encontrar_medicao_mais_proxima(
            acidente,
            medicoes
        )
    )


    if medicao is None:
        registro["status_integracao"] = (
            "sem_medicao_temporal"
        )

        resultados.append(registro)
        continue


    registro["data_hora_medicao"] = (
        medicao["data_hora_sp"]
    )

    registro["diferenca_tempo_min"] = (
        diferenca_minutos
    )

    registro["precipitacao_mm"] = (
        medicao["precipitacao_mm"]
    )

    if pd.isna(
        medicao["precipitacao_mm"]
    ):
        registro["status_integracao"] = (
            "precipitacao_ausente"
        )
    else:
        registro["status_integracao"] = (
            "integrado"
        )


    resultados.append(registro)


    if numero % 100 == 0:
        print(
            f"{numero}/{len(acidentes)} "
            "acidentes processados"
        )


base_integrada = pd.DataFrame(
    resultados
)

base_integrada.to_csv(
    SAIDA,
    index=False,
    encoding="utf-8-sig"
)


print("\nStatus da integração:")

print(
    base_integrada[
        "status_integracao"
    ].value_counts(
        dropna=False
    )
)

print("\nDistância até estação:")

print(
    base_integrada[
        "distancia_estacao_km"
    ].describe()
)

print("\nDiferença temporal:")

print(
    base_integrada[
        "diferenca_tempo_min"
    ].describe()
)