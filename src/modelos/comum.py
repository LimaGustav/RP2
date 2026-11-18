import pandas as pd


def formatar_significancia(p_valor):
    """Converte um valor-p em uma marcação simples de significância."""
    if pd.isna(p_valor):
        return ""

    if p_valor < 0.001:
        return "***"
    if p_valor < 0.01:
        return "**"
    if p_valor < 0.05:
        return "*"
    if p_valor < 0.1:
        return "."

    return "ns"
