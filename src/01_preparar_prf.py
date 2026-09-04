from pathlib import Path
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]

ENTRADA = RAIZ / "datasets" / "raw" / "prf" / "datatran2026.csv"
SAIDA = RAIZ / "datasets" / "processed" / "acidentes_sp.csv"


dados = pd.read_csv(
    ENTRADA,
    sep=";",
    encoding="latin1"
)

# Filtrar São Paulo
dados = dados[dados["uf"] == "SP"].copy()

colunas = [
    "id",
    "data_inversa",
    "horario",
    "uf",
    "br",
    "km",
    "municipio",
    "latitude",
    "longitude",
    "classificacao_acidente",
    "condicao_metereologica",
    "fase_dia",
    "tipo_pista",
    "tracado_via",
    "mortos",
    "feridos_graves",
    "feridos_leves"
]

dados = dados[colunas].copy()

# Data
acidentes["data_inversa"] = pd.to_datetime(
    acidentes["data_inversa"],
    format="%Y-%m-%d",
    errors="coerce"
)

# horario
acidentes["horario"] = (
    acidentes["horario"]
    .astype("string")
    .str.strip()
)

# Data + horário
acidentes["data_hora"] = pd.to_datetime(
    acidentes["data_inversa"].dt.strftime("%Y-%m-%d")
    + " "
    + acidentes["horario"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

# Coordenadas
for coluna in ["latitude", "longitude"]:
    dados[coluna] = (
        dados[coluna]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    dados[coluna] = pd.to_numeric(
        dados[coluna],
        errors="coerce"
    )

# Gravidade ordinal
mapa_gravidade = {
    "Sem Vítimas": 0,
    "Com Vítimas Feridas": 1,
    "Com Vítimas Fatais": 2
}

dados["gravidade"] = (
    dados["classificacao_acidente"]
    .map(mapa_gravidade)
)

SAIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)

dados.to_csv(
    SAIDA,
    index=False,
    encoding="utf-8-sig"
)

print(f"Acidentes em SP: {len(dados)}")

print("\nGravidade:")
print(dados["classificacao_acidente"].value_counts())

print("\nValidação das datas:")
print("Datas inválidas:", acidentes["data_inversa"].isna().sum())
print("Data/hora inválida:", acidentes["data_hora"].isna().sum())

print("\nPeríodo:")
print(acidentes["data_hora"].min())
print(acidentes["data_hora"].max())

print(f"\nArquivo criado: {SAIDA}")