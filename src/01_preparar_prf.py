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
dados["data_inversa"] = pd.to_datetime(
    dados["data_inversa"],
    errors="coerce"
)

# Data + horário
dados["data_hora"] = pd.to_datetime(
    dados["data_inversa"].dt.strftime("%Y-%m-%d")
    + " "
    + dados["horario"],
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

print(f"\nArquivo criado: {SAIDA}")