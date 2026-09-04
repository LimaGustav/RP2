from pathlib import Path
import pandas as pd


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


def ler_metadados(caminho):
    metadados = {}

    with open(
        caminho,
        "r",
        encoding="latin1"
    ) as arquivo:

        for _ in range(8):
            linha = arquivo.readline().strip()

            partes = linha.split(";", 1)

            if len(partes) == 2:
                chave = (
                    partes[0]
                    .replace(":", "")
                    .strip()
                )

                valor = partes[1].strip()

                metadados[chave] = valor

    return metadados


def processar_arquivo(caminho):

    metadados = ler_metadados(caminho)

    # Segurança extra:
    # processar somente estações de SP
    if metadados.get("UF") != "SP":
        return None

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

    dados = dados[
        [
            "Data",
            "Hora UTC",
            coluna_chuva
        ]
    ].copy()

    dados = dados.rename(
        columns={
            "Data": "data",
            "Hora UTC": "hora_utc",
            coluna_chuva: "precipitacao_mm"
        }
    )

    # Metadados da estação
    dados["codigo_wmo"] = (
        metadados.get("CODIGO (WMO)")
    )

    dados["estacao"] = (
        metadados.get("ESTACAO")
    )

    dados["uf_estacao"] = (
        metadados.get("UF")
    )

    dados["latitude_estacao"] = pd.to_numeric(
        str(
            metadados.get("LATITUDE", "")
        ).replace(",", "."),
        errors="coerce"
    )

    dados["longitude_estacao"] = pd.to_numeric(
        str(
            metadados.get("LONGITUDE", "")
        ).replace(",", "."),
        errors="coerce"
    )

    # Exemplo: 1200 UTC -> 1200
    dados["hora_utc"] = (
        dados["hora_utc"]
        .astype(str)
        .str.extract(r"(\d{4})")[0]
    )

    # Criar horário UTC
    dados["data_hora_utc"] = pd.to_datetime(
        dados["data"].astype(str)
        + " "
        + dados["hora_utc"],
        format="%Y/%m/%d %H%M",
        errors="coerce",
        utc=True
    )

    # Converter para horário de São Paulo
    dados["data_hora_sp"] = (
        dados["data_hora_utc"]
        .dt.tz_convert("America/Sao_Paulo")
        .dt.tz_localize(None)
    )

    dados["precipitacao_mm"] = pd.to_numeric(
        dados["precipitacao_mm"],
        errors="coerce"
    )

    return dados


# Procurar arquivos de SP pelo nome
arquivos = list(
    PASTA_INMET.glob("INMET_*_SP_*.CSV")
)

print(
    f"Arquivos de SP encontrados: {len(arquivos)}"
)

bases = []

for caminho in arquivos:

    try:
        dados_estacao = processar_arquivo(caminho)

        if dados_estacao is not None:
            bases.append(dados_estacao)

            print(
                f"OK: {caminho.name}"
            )

    except Exception as erro:
        print(
            f"ERRO: {caminho.name}"
        )
        print(erro)


if not bases:
    raise ValueError(
        "Nenhuma estação de SP foi processada."
    )


dados_inmet = pd.concat(
    bases,
    ignore_index=True
)

SAIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)

dados_inmet.to_csv(
    SAIDA,
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    "Estações:",
    dados_inmet["codigo_wmo"].nunique()
)

print(
    "Registros:",
    len(dados_inmet)
)

print(
    "Período:",
    dados_inmet["data_hora_sp"].min(),
    "até",
    dados_inmet["data_hora_sp"].max()
)

print(
    f"\nArquivo criado: {SAIDA}"
)