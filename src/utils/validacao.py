import pandas as pd


def preparar_tipos(acidentes, inmet):

    acidentes["data_hora"] = pd.to_datetime(
        acidentes["data_hora"],
        errors="coerce"
    )

    inmet["data_hora_sp"] = pd.to_datetime(
        inmet["data_hora_sp"],
        errors="coerce"
    )

    inmet["precipitacao_mm"] = pd.to_numeric(
        inmet["precipitacao_mm"],
        errors="coerce"
    )

    for coluna in [
        "latitude",
        "longitude"
    ]:
        acidentes[coluna] = pd.to_numeric(
            acidentes[coluna],
            errors="coerce"
        )

    for coluna in [
        "latitude_estacao",
        "longitude_estacao"
    ]:
        inmet[coluna] = pd.to_numeric(
            inmet[coluna],
            errors="coerce"
        )

    return acidentes, inmet