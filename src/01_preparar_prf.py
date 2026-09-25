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
# Localizar arquivos da PRF
# =========================================================

arquivos_prf = sorted(
    PASTA_PRF.glob("datatran*.csv")
)

if not arquivos_prf:
    raise FileNotFoundError(
        f"Nenhum arquivo datatran encontrado em: {PASTA_PRF}"
    )


print("=" * 60)
print("PREPARAÇÃO DOS DADOS DA PRF")
print("=" * 60)

print("\nArquivos encontrados:")

for arquivo in arquivos_prf:
    print("-", arquivo.name)


# =========================================================
# Carregar arquivos
# =========================================================

bases = []

for arquivo in arquivos_prf:

    print(
        f"\nCarregando {arquivo.name}..."
    )

    base = pd.read_csv(
        arquivo,
        sep=";",
        encoding="latin1",
        low_memory=False
    )

    base["arquivo_origem"] = (
        arquivo.name
    )

    print(
        f"Registros carregados: {len(base):,}"
    )

    bases.append(base)


dados = pd.concat(
    bases,
    ignore_index=True
)


print(
    f"\nTotal antes dos filtros: {len(dados):,}"
)


# =========================================================
# Validar colunas necessárias
# =========================================================

colunas_necessarias = [
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
    "arquivo_origem",
]


colunas_ausentes = [
    coluna
    for coluna in colunas_necessarias
    if coluna not in dados.columns
]

if colunas_ausentes:
    raise ValueError(
        "Colunas ausentes na base da PRF:\n- "
        + "\n- ".join(colunas_ausentes)
    )


# =========================================================
# Filtrar São Paulo
# =========================================================

dados["uf"] = (
    dados["uf"]
    .astype("string")
    .str.strip()
    .str.upper()
)


dados = dados[
    dados["uf"] == "SP"
].copy()


print(
    f"\nRegistros de SP antes do filtro temporal: {len(dados):,}"
)


print(
    "\nRegistros de SP por arquivo:"
)

print(
    dados[
        "arquivo_origem"
    ]
    .value_counts()
)


# =========================================================
# Selecionar colunas
# =========================================================

dados = dados[
    colunas_necessarias
].copy()


# =========================================================
# Tratar datas
#
# Tenta diferentes formatos porque os arquivos de anos
# distintos podem utilizar representações diferentes.
# =========================================================

data_original = (
    dados["data_inversa"]
    .astype("string")
    .str.strip()
)


datas_convertidas = pd.Series(
    pd.NaT,
    index=dados.index,
    dtype="datetime64[ns]"
)


formatos_data = [
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%d-%m-%Y",
]


for formato in formatos_data:

    faltantes = (
        datas_convertidas.isna()
        & data_original.notna()
    )

    datas_convertidas.loc[faltantes] = (
        pd.to_datetime(
            data_original.loc[faltantes],
            format=formato,
            errors="coerce"
        )
    )


dados["data_inversa"] = (
    datas_convertidas
)


# =========================================================
# Validar datas ANTES do filtro temporal
# =========================================================

print(
    "\nValidação das datas por arquivo:"
)


validacao_datas = (
    dados.groupby(
        "arquivo_origem"
    )["data_inversa"]
    .agg(
        total="size",
        validas=lambda x: x.notna().sum(),
        invalidas=lambda x: x.isna().sum()
    )
)


print(
    validacao_datas
)


print(
    "\nExemplos das datas originais:"
)

for arquivo in dados[
    "arquivo_origem"
].unique():

    mascara = (
        dados["arquivo_origem"]
        == arquivo
    )

    exemplos = (
        data_original.loc[mascara]
        .dropna()
        .head(5)
        .tolist()
    )

    print(
        arquivo,
        "->",
        exemplos
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


print(
    "\nAno identificado antes do filtro:"
)

print(
    dados["ano"]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


# =========================================================
# Filtrar período comparável
#
# Janeiro a maio de 2025
# Janeiro a maio de 2026
# =========================================================

dados = dados[
    dados["ano"].isin(
        [2025, 2026]
    )
    & dados["mes"].between(
        1,
        5
    )
].copy()


print(
    "\nQuantidade de acidentes após filtro temporal:"
)

print(
    len(dados)
)


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


print(
    "\nAcidentes por arquivo após filtro:"
)

print(
    dados[
        "arquivo_origem"
    ]
    .value_counts()
)


# =========================================================
# Tratar horário
# =========================================================

dados["horario"] = (
    dados["horario"]
    .astype("string")
    .str.strip()
)


dados["horario_delta"] = (
    pd.to_timedelta(
        dados["horario"],
        errors="coerce"
    )
)


dados["data_hora"] = (
    dados["data_inversa"]
    + dados["horario_delta"]
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
        .str.strip()
        .str.replace(
            ",",
            ".",
            regex=False
        )
    )

    dados[coluna] = (
        pd.to_numeric(
            dados[coluna],
            errors="coerce"
        )
    )


# =========================================================
# Tratar BR e KM
# =========================================================

dados["br"] = pd.to_numeric(
    dados["br"],
    errors="coerce"
)


dados["km"] = (
    dados["km"]
    .astype("string")
    .str.replace(
        ",",
        ".",
        regex=False
    )
)


dados["km"] = pd.to_numeric(
    dados["km"],
    errors="coerce"
)


# =========================================================
# Variáveis numéricas complementares
# =========================================================

for coluna in [
    "mortos",
    "feridos_graves",
    "feridos_leves"
]:

    dados[coluna] = pd.to_numeric(
        dados[coluna],
        errors="coerce"
    )


# =========================================================
# Padronizar classificação do acidente
# =========================================================

dados["classificacao_acidente"] = (
    dados["classificacao_acidente"]
    .astype("string")
    .str.strip()
)


# =========================================================
# Criar variável ordinal de gravidade
#
# 0 = Sem vítimas
# 1 = Com vítimas feridas
# 2 = Com vítimas fatais
# =========================================================

mapa_gravidade = {
    "Sem Vítimas": 0,
    "Com Vítimas Feridas": 1,
    "Com Vítimas Fatais": 2,
}


dados["gravidade_ordinal"] = (
    dados[
        "classificacao_acidente"
    ]
    .map(
        mapa_gravidade
    )
)


# =========================================================
# Validações
# =========================================================

print(
    "\n" + "=" * 60
)

print(
    "VALIDAÇÃO DA BASE PREPARADA"
)

print(
    "=" * 60
)


print(
    "\nQuantidade total:"
)

print(
    len(dados)
)


print(
    "\nQuantidade por ano:"
)

print(
    dados[
        "ano"
    ]
    .value_counts()
    .sort_index()
)


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


print(
    "\nDatas inválidas:"
)

print(
    dados[
        "data_inversa"
    ]
    .isna()
    .sum()
)


print(
    "\nHorários inválidos:"
)

print(
    dados[
        "horario_delta"
    ]
    .isna()
    .sum()
)


print(
    "\nData/hora inválida:"
)

print(
    dados[
        "data_hora"
    ]
    .isna()
    .sum()
)


print(
    "\nLatitudes inválidas:"
)

print(
    dados[
        "latitude"
    ]
    .isna()
    .sum()
)


print(
    "\nLongitudes inválidas:"
)

print(
    dados[
        "longitude"
    ]
    .isna()
    .sum()
)


print(
    "\nGravidade ordinal inválida:"
)

print(
    dados[
        "gravidade_ordinal"
    ]
    .isna()
    .sum()
)


# =========================================================
# Verificar IDs duplicados
# =========================================================

ids_duplicados = (
    dados[
        "id"
    ]
    .duplicated()
    .sum()
)


print(
    "\nIDs duplicados:"
)

print(
    ids_duplicados
)


# =========================================================
# Período
# =========================================================

print(
    "\nPeríodo da base:"
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
# Ordenar
# =========================================================

dados = dados.sort_values(
    by=[
        "data_hora",
        "id"
    ]
).reset_index(
    drop=True
)


# =========================================================
# Remover coluna auxiliar
# =========================================================

dados = dados.drop(
    columns=[
        "horario_delta"
    ]
)


# =========================================================
# Salvar
# =========================================================

SAIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)


dados.to_csv(
    SAIDA,
    index=False
)


print(
    f"\nBase salva em: {SAIDA}"
)

print(
    "Preparação da PRF concluída."
)