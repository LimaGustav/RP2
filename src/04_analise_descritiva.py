from pathlib import Path
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]

CAMINHO = (
    RAIZ
    / "datasets"
    / "processed"
    / "base_integrada.csv"
)

dados = pd.read_csv(
    CAMINHO
)


print("Quantidade de acidentes:")
print(len(dados))


print("\nGravidade:")
print(
    dados[
        "classificacao_acidente"
    ].value_counts()
)


print("\nPrecipitação:")
print(
    dados[
        "precipitacao_mm"
    ].describe()
)


print("\nDistância até estação:")
print(
    dados[
        "distancia_estacao_km"
    ].describe()
)


print("\nDiferença temporal:")
print(
    dados[
        "diferenca_tempo_min"
    ].describe()
)


print("\nPrecipitação por gravidade:")

print(
    dados.groupby(
        "classificacao_acidente"
    )["precipitacao_mm"]
    .describe()
)