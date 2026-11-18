import pandas as pd


def preparar_tipos(
    acidentes: pd.DataFrame,
    inmet: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Padroniza os tipos usados na integração."""
    acidentes = acidentes.copy()
    inmet = inmet.copy()

    acidentes["data_hora"] = pd.to_datetime(
        acidentes["data_hora"],
        errors="coerce",
    )

    inmet["data_hora"] = pd.to_datetime(
        inmet["data_hora"],
        errors="coerce",
    )

    for coluna in ["latitude", "longitude"]:
        acidentes[coluna] = pd.to_numeric(
            acidentes[coluna],
            errors="coerce",
        )

        inmet[coluna] = pd.to_numeric(
            inmet[coluna],
            errors="coerce",
        )

    inmet["precipitacao_mm"] = pd.to_numeric(
        inmet["precipitacao_mm"],
        errors="coerce",
    )

    return acidentes, inmet