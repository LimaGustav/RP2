from pathlib import Path

import pandas as pd


# =========================================================
# Caminhos
# =========================================================

RAIZ = Path(__file__).resolve().parents[1]

PASTA_INMET = (
    RAIZ
    / "datasets"
    / "raw"
    / "inmet"
)

SAIDA = (
    RAIZ
    / "datasets"
    / "processed"
    / "inmet_sp.csv"
)


# =========================================================
# Ler metadados da estação
# =========================================================

def ler_metadados(caminho):
    metadados = {}

    with open(
        caminho,
        "r",
        encoding="latin1"
    ) as arquivo:

        for _ in range(8):
            linha = arquivo.readline().strip()

            partes = linha.split(
                ";",
                1
            )

            if len(partes) == 2:

                chave = (
                    partes[0]
                    .replace(":", "")
                    .strip()
                )

                valor = (
                    partes[1]
                    .strip()
                )

                metadados[chave] = valor

    return metadados


# =========================================================
# Processar um arquivo do INMET
# =========================================================

def processar_arquivo(caminho):

    metadados = ler_metadados(
        caminho
    )

    # -----------------------------------------------------
    # Segurança extra:
    # processar somente estações de SP
    # -----------------------------------------------------

    if metadados.get("UF") != "SP":
        return None


    # -----------------------------------------------------
    # Carregar dados horários
    # -----------------------------------------------------

    dados = pd.read_csv(
        caminho,
        sep=";",
        encoding="latin1",
        skiprows=8,
        decimal=","
    )


    coluna_chuva = (
        "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)"
    )


    # -----------------------------------------------------
    # Verificar colunas necessárias
    # -----------------------------------------------------

    colunas_necessarias = [
        "Data",
        "Hora UTC",
        coluna_chuva
    ]

    colunas_ausentes = [
        coluna
        for coluna in colunas_necessarias
        if coluna not in dados.columns
    ]

    if colunas_ausentes:
        raise ValueError(
            "Colunas ausentes: "
            + ", ".join(colunas_ausentes)
        )


    dados = dados[
        colunas_necessarias
    ].copy()


    dados = dados.rename(
        columns={
            "Data": "data",
            "Hora UTC": "hora_utc",
            coluna_chuva: "precipitacao_mm"
        }
    )


    # =====================================================
    # Metadados da estação
    # =====================================================

    dados["codigo_wmo"] = (
        metadados.get(
            "CODIGO (WMO)"
        )
    )

    dados["estacao"] = (
        metadados.get(
            "ESTACAO"
        )
    )

    dados["uf_estacao"] = (
        metadados.get(
            "UF"
        )
    )


    dados["latitude_estacao"] = pd.to_numeric(
        str(
            metadados.get(
                "LATITUDE",
                ""
            )
        ).replace(
            ",",
            "."
        ),
        errors="coerce"
    )


    dados["longitude_estacao"] = pd.to_numeric(
        str(
            metadados.get(
                "LONGITUDE",
                ""
            )
        ).replace(
            ",",
            "."
        ),
        errors="coerce"
    )


    # =====================================================
    # Guardar origem do registro
    # =====================================================

    dados["arquivo_origem"] = (
        caminho.name
    )


    # =====================================================
    # Tratar horário UTC
    #
    # Exemplo:
    # 1200 UTC -> 1200
    # =====================================================

    dados["hora_utc"] = (
        dados["hora_utc"]
        .astype("string")
        .str.extract(
            r"(\d{4})"
        )[0]
    )


    # =====================================================
    # Criar data/hora UTC
    # =====================================================

    dados["data_hora_utc"] = pd.to_datetime(
        dados["data"].astype("string")
        + " "
        + dados["hora_utc"],
        format="%Y/%m/%d %H%M",
        errors="coerce",
        utc=True
    )


    # =====================================================
    # Converter UTC para horário de São Paulo
    # =====================================================

    dados["data_hora_sp"] = (
        dados["data_hora_utc"]
        .dt.tz_convert(
            "America/Sao_Paulo"
        )
        .dt.tz_localize(
            None
        )
    )


    # =====================================================
    # Criar ano e mês
    # =====================================================

    dados["ano"] = (
        dados["data_hora_sp"]
        .dt.year
    )

    dados["mes"] = (
        dados["data_hora_sp"]
        .dt.month
    )


    # =====================================================
    # Tratar precipitação
    # =====================================================

    dados["precipitacao_mm"] = pd.to_numeric(
        dados["precipitacao_mm"],
        errors="coerce"
    )


    return dados


# =========================================================
# Procurar arquivos do INMET recursivamente
#
# Estrutura esperada:
#
# datasets/raw/inmet/
# ├── 2025/
# └── 2026/
# =========================================================

arquivos = [
    arquivo
    for arquivo in PASTA_INMET.rglob("*")
    if arquivo.is_file()
    and arquivo.suffix.lower() == ".csv"
    and arquivo.name.upper().startswith(
        "INMET_"
    )
    and "_SP_" in arquivo.name.upper()
]


arquivos = sorted(
    arquivos
)


if not arquivos:
    raise FileNotFoundError(
        f"Nenhum arquivo do INMET de SP encontrado em {PASTA_INMET}"
    )


print(
    f"Arquivos de SP encontrados: {len(arquivos)}"
)


# =========================================================
# Mostrar quantidade encontrada por pasta/ano
# =========================================================

contagem_por_pasta = {}

for arquivo in arquivos:

    pasta = arquivo.parent.name

    contagem_por_pasta[pasta] = (
        contagem_por_pasta.get(
            pasta,
            0
        )
        + 1
    )


print(
    "\nArquivos encontrados por pasta:"
)

for pasta, quantidade in sorted(
    contagem_por_pasta.items()
):
    print(
        f"{pasta}: {quantidade}"
    )


# =========================================================
# Processar arquivos
# =========================================================

bases = []

erros = []


for caminho in arquivos:

    try:

        dados_estacao = (
            processar_arquivo(
                caminho
            )
        )

        if dados_estacao is not None:

            bases.append(
                dados_estacao
            )

            print(
                f"OK: "
                f"{caminho.parent.name}/"
                f"{caminho.name}"
            )

    except Exception as erro:

        erros.append(
            (
                caminho,
                str(erro)
            )
        )

        print(
            f"ERRO: "
            f"{caminho.parent.name}/"
            f"{caminho.name}"
        )

        print(
            erro
        )


# =========================================================
# Verificar se houve dados processados
# =========================================================

if not bases:
    raise ValueError(
        "Nenhuma estação de SP foi processada."
    )


# =========================================================
# Unir todas as estações e anos
# =========================================================

dados_inmet = pd.concat(
    bases,
    ignore_index=True
)


# =========================================================
# Ordenar base
# =========================================================

dados_inmet = dados_inmet.sort_values(
    [
        "data_hora_sp",
        "codigo_wmo"
    ]
).reset_index(
    drop=True
)


# =========================================================
# Validações
# =========================================================

print()
print("=" * 60)
print("RESUMO DA BASE INMET")
print("=" * 60)


print(
    "\nEstações únicas:",
    dados_inmet[
        "codigo_wmo"
    ].nunique()
)


print(
    "\nRegistros:",
    len(dados_inmet)
)


# ---------------------------------------------------------
# Registros por ano
# ---------------------------------------------------------

print(
    "\nRegistros por ano:"
)

print(
    dados_inmet[
        "ano"
    ]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# Estações por ano
# ---------------------------------------------------------

print(
    "\nEstações por ano:"
)

print(
    dados_inmet.groupby(
        "ano"
    )[
        "codigo_wmo"
    ].nunique()
)


# ---------------------------------------------------------
# Precipitação ausente
# ---------------------------------------------------------

print(
    "\nPrecipitação ausente por ano:"
)

print(
    dados_inmet.groupby(
        "ano"
    )[
        "precipitacao_mm"
    ]
    .apply(
        lambda coluna:
        coluna.isna().sum()
    )
)


# ---------------------------------------------------------
# Datas inválidas
# ---------------------------------------------------------

print(
    "\nData/hora inválida:",
    dados_inmet[
        "data_hora_sp"
    ]
    .isna()
    .sum()
)


# ---------------------------------------------------------
# Coordenadas inválidas
# ---------------------------------------------------------

print(
    "\nLatitude da estação inválida:",
    dados_inmet[
        "latitude_estacao"
    ]
    .isna()
    .sum()
)

print(
    "Longitude da estação inválida:",
    dados_inmet[
        "longitude_estacao"
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
    dados_inmet[
        "data_hora_sp"
    ]
    .min()
)

print(
    "Fim:",
    dados_inmet[
        "data_hora_sp"
    ]
    .max()
)


# ---------------------------------------------------------
# Erros
# ---------------------------------------------------------

print(
    "\nArquivos com erro:",
    len(erros)
)


# =========================================================
# Salvar base processada
# =========================================================

SAIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)

dados_inmet.to_csv(
    SAIDA,
    index=False,
    encoding="utf-8-sig"
)


print(
    f"\nArquivo criado: {SAIDA}"
)