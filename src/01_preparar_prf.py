from pathlib import Path

import pandas as pd


# =========================================================
# Caminhos
# =========================================================

RAIZ = Path(__file__).resolve().parents[1]

ENTRADA = (
    RAIZ
    / "datasets"
    / "raw"
    / "prf"
    / "datatran2026.csv"
)

SAIDA = (
    RAIZ
    / "datasets"
    / "processed"
    / "acidentes_sp.csv"
)


# =========================================================
# Carregar base da PRF
# =========================================================

dados = pd.read_csv(
    ENTRADA,
    sep=";",
    encoding="latin1"
)


# =========================================================
# Filtrar acidentes de São Paulo
# =========================================================

dados = dados[
    dados["uf"] == "SP"
].copy()


# =========================================================
# Selecionar colunas utilizadas no projeto
# =========================================================

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

dados = dados[
    colunas
].copy()


# =========================================================
# Tratar data
#
# A PRF utiliza o formato:
# YYYY-MM-DD
# =========================================================

dados["data_inversa"] = pd.to_datetime(
    dados["data_inversa"],
    format="%Y-%m-%d",
    errors="coerce"
)


# =========================================================
# Tratar horário
# =========================================================

dados["horario"] = (
    dados["horario"]
    .astype("string")
    .str.strip()
)

horario_delta = pd.to_timedelta(
    dados["horario"],
    errors="coerce"
)


# =========================================================
# Criar data + horário do acidente
# =========================================================

dados["data_hora"] = (
    dados["data_inversa"]
    + horario_delta
)


# =========================================================
# Tratar coordenadas
# =========================================================

for coluna in [
    "latitude",
    "longitude"
]:

    dados[coluna] = (
        dados[coluna]
        .astype("string")
        .str.replace(
            ",",
            ".",
            regex=False
        )
    )

    dados[coluna] = pd.to_numeric(
        dados[coluna],
        errors="coerce"
    )


# =========================================================
# Criar variável ordinal de gravidade
#
# 0 = sem vítimas
# 1 = vítimas feridas
# 2 = vítimas fatais
# =========================================================

mapa_gravidade = {
    "Sem Vítimas": 0,
    "Com Vítimas Feridas": 1,
    "Com Vítimas Fatais": 2
}

dados["gravidade"] = (
    dados[
        "classificacao_acidente"
    ]
    .map(mapa_gravidade)
)


# =========================================================
# Validações
# =========================================================

print(
    f"Acidentes em SP: {len(dados)}"
)

print(
    "\nGravidade:"
)

print(
    dados[
        "classificacao_acidente"
    ].value_counts(
        dropna=False
    )
)


print(
    "\nValidação das datas:"
)

print(
    "Datas inválidas:",
    dados[
        "data_inversa"
    ].isna().sum()
)

print(
    "Horários inválidos:",
    horario_delta.isna().sum()
)

print(
    "Data/hora inválida:",
    dados[
        "data_hora"
    ].isna().sum()
)


print(
    "\nValidação das coordenadas:"
)

print(
    "Latitudes inválidas:",
    dados[
        "latitude"
    ].isna().sum()
)

print(
    "Longitudes inválidas:",
    dados[
        "longitude"
    ].isna().sum()
)


print(
    "\nPeríodo:"
)

print(
    "Início:",
    dados[
        "data_hora"
    ].min()
)

print(
    "Fim:",
    dados[
        "data_hora"
    ].max()
)


# =========================================================
# Salvar base processada
# =========================================================

SAIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)

dados.to_csv(
    SAIDA,
    index=False,
    encoding="utf-8-sig"
)


print(
    f"\nArquivo criado: {SAIDA}"
)