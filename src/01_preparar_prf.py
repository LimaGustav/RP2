from pathlib import Path

import pandas as pd


# =========================================================
# Caminhos
# =========================================================

RAIZ = Path(__file__).resolve().parents[1]

PASTA_PRF = (
    RAIZ
    / "datasets"
    / "raw"
    / "prf"
)

SAIDA = (
    RAIZ
    / "datasets"
    / "processed"
    / "acidentes_sp.csv"
)


# =========================================================
# Localizar bases da PRF
# =========================================================

arquivos_prf = sorted(
    PASTA_PRF.glob("datatran*.csv")
)

if not arquivos_prf:
    raise FileNotFoundError(
        f"Nenhum arquivo datatran*.csv encontrado em {PASTA_PRF}"
    )

print("Arquivos da PRF encontrados:")

for arquivo in arquivos_prf:
    print(f"- {arquivo.name}")


# =========================================================
# Carregar e unir bases
# =========================================================

bases = []

for arquivo in arquivos_prf:

    print(
        f"\nCarregando {arquivo.name}..."
    )

    dados_ano = pd.read_csv(
        arquivo,
        sep=";",
        encoding="latin1"
    )

    dados_ano["arquivo_origem"] = (
        arquivo.name
    )

    bases.append(
        dados_ano
    )


dados = pd.concat(
    bases,
    ignore_index=True
)


print(
    f"\nRegistros antes dos filtros: {len(dados)}"
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
    "feridos_leves",
    "arquivo_origem"
]

dados = dados[
    colunas
].copy()


# =========================================================
# Tratar data
#
# Formato utilizado pela PRF:
# DD/MM/YYYY
# =========================================================

dados["data_inversa"] = pd.to_datetime(
    dados["data_inversa"],
    format="%d/%m/%Y",
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

dados["horario_delta"] = pd.to_timedelta(
    dados["horario"],
    errors="coerce"
)


# =========================================================
# Criar data + horário do acidente
# =========================================================

dados["data_hora"] = (
    dados["data_inversa"]
    + dados["horario_delta"]
)


# =========================================================
# Criar ano e mês
# =========================================================

dados["ano"] = (
    dados["data_inversa"]
    .dt.year
)

dados["mes"] = (
    dados["data_inversa"]
    .dt.month
)


# =========================================================
# Manter período comparável entre 2025 e 2026
#
# Como a base atual de 2026 termina em maio,
# utilizamos janeiro a maio dos dois anos.
# =========================================================

dados = dados[
    dados["ano"].isin(
        [2025, 2026]
    )
    &
    dados["mes"].between(
        1,
        5
    )
].copy()


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
# Ordenar base
# =========================================================

dados = dados.sort_values(
    [
        "data_hora",
        "id"
    ]
).reset_index(
    drop=True
)


# =========================================================
# Validações
# =========================================================

print()
print("=" * 60)
print("RESUMO DA BASE PRF")
print("=" * 60)


print(
    f"\nAcidentes em SP no período selecionado: {len(dados)}"
)


# ---------------------------------------------------------
# Quantidade por ano
# ---------------------------------------------------------

print(
    "\nAcidentes por ano:"
)

print(
    dados[
        "ano"
    ]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# Gravidade geral
# ---------------------------------------------------------

print(
    "\nGravidade:"
)

print(
    dados[
        "classificacao_acidente"
    ]
    .value_counts(
        dropna=False
    )
)


# ---------------------------------------------------------
# Gravidade por ano
# ---------------------------------------------------------

print(
    "\nGravidade por ano:"
)

print(
    pd.crosstab(
        dados["ano"],
        dados[
            "classificacao_acidente"
        ]
    )
)


# ---------------------------------------------------------
# Datas
# ---------------------------------------------------------

print(
    "\nValidação das datas:"
)

print(
    "Datas inválidas:",
    dados["data_inversa"]
    .isna()
    .sum()
)

print(
    "Horários inválidos:",
    dados["horario_delta"]
    .isna()
    .sum()
)

print(
    "Data/hora inválida:",
    dados["data_hora"]
    .isna()
    .sum()
)

# ---------------------------------------------------------
# Coordenadas
# ---------------------------------------------------------

print(
    "\nValidação das coordenadas:"
)

print(
    "Latitudes inválidas:",
    dados[
        "latitude"
    ]
    .isna()
    .sum()
)

print(
    "Longitudes inválidas:",
    dados[
        "longitude"
    ]
    .isna()
    .sum()
)


# ---------------------------------------------------------
# Período
# ---------------------------------------------------------

print(
    "\nPeríodo:"
)

print(
    "Início:",
    dados[
        "data_hora"
    ]
    .min()
)

print(
    "Fim:",
    dados[
        "data_hora"
    ]
    .max()
)

# =========================================================
# Remover colunas auxiliares
# =========================================================

dados = dados.drop(
    columns=["horario_delta"]
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