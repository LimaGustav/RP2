import unicodedata

import pandas as pd

from config import ARQUIVO_INMET, PASTA_INMET


def normalizar_texto(valor: str) -> str:
    """Remove acentos e padroniza texto para comparação."""
    return (
        unicodedata.normalize("NFKD", str(valor))
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .upper()
    )


def converter_numero(valor):
    """Converte números com vírgula decimal para float."""
    if valor is None:
        return None

    texto = str(valor).strip().replace(",", ".")

    try:
        return float(texto)
    except ValueError:
        return None


def ler_metadados(arquivo) -> dict:
    """Lê os metadados presentes no início de um arquivo do INMET."""
    metadados = {}

    with open(arquivo, encoding="latin1") as arquivo_aberto:
        for _ in range(8):
            linha = arquivo_aberto.readline()

            chave, separador, valor = linha.partition(";")

            if not separador:
                continue

            chave = normalizar_texto(chave).rstrip(":")
            valor = valor.strip().strip(";")

            metadados[chave] = valor

    return {
        "estacao": metadados.get("ESTACAO"),
        "codigo_wmo": metadados.get("CODIGO (WMO)"),
        "latitude": converter_numero(metadados.get("LATITUDE")),
        "longitude": converter_numero(metadados.get("LONGITUDE")),
    }


def localizar_coluna(dados: pd.DataFrame, trecho: str) -> str:
    """Localiza uma coluna independentemente de acentos e caixa."""
    trecho = normalizar_texto(trecho)

    for coluna in dados.columns:
        if trecho in normalizar_texto(coluna):
            return coluna

    raise KeyError(f"Coluna não encontrada: {trecho}")


def criar_data_hora(dados: pd.DataFrame) -> pd.Series:
    """Combina data e hora UTC e converte para o horário de São Paulo."""
    coluna_data = localizar_coluna(dados, "DATA")
    coluna_hora = localizar_coluna(dados, "HORA UTC")

    datas = (
        dados[coluna_data]
        .astype("string")
        .str.strip()
    )

    horas = (
        dados[coluna_hora]
        .astype("string")
        .str.extract(r"(\d{1,4})", expand=False)
        .str.zfill(4)
    )

    data_hora_utc = pd.to_datetime(
        datas
        + " "
        + horas.str[:2]
        + ":"
        + horas.str[2:],
        errors="coerce",
        utc=True,
    )

    return (
        data_hora_utc
        .dt.tz_convert("America/Sao_Paulo")
        .dt.tz_localize(None)
    )


def processar_arquivo(arquivo) -> pd.DataFrame:
    """Processa um arquivo de uma estação meteorológica do INMET."""
    metadados = ler_metadados(arquivo)

    dados = pd.read_csv(
        arquivo,
        sep=";",
        skiprows=8,
        encoding="latin1",
        dtype="string",
    )

    coluna_precipitacao = localizar_coluna(
        dados,
        "PRECIPITACAO TOTAL, HORARIO",
    )

    precipitacao = pd.to_numeric(
        dados[coluna_precipitacao]
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )

    resultado = pd.DataFrame(
        {
            "codigo_wmo": metadados["codigo_wmo"],
            "estacao": metadados["estacao"],
            "latitude": metadados["latitude"],
            "longitude": metadados["longitude"],
            "data_hora": criar_data_hora(dados),
            "precipitacao_mm": precipitacao,
        }
    )

    resultado["ano"] = resultado["data_hora"].dt.year
    resultado["mes"] = resultado["data_hora"].dt.month

    return resultado


def localizar_arquivos() -> list:
    """Localiza os CSVs de São Paulo copiados pelo notebook."""
    if not PASTA_INMET.exists():
        raise FileNotFoundError(
            f"Pasta do INMET não encontrada: {PASTA_INMET}"
        )

    arquivos = sorted(
        arquivo
        for arquivo in PASTA_INMET.rglob("*")
        if arquivo.is_file()
        and arquivo.suffix.lower() == ".csv"
        and arquivo.name.upper().startswith("INMET_")
        and "_SP_" in arquivo.name.upper()
    )

    if not arquivos:
        raise FileNotFoundError(
            "Nenhum arquivo do INMET de São Paulo foi encontrado."
        )

    return arquivos


def preparar_inmet() -> pd.DataFrame:
    """Processa e combina todos os arquivos do INMET."""
    arquivos = localizar_arquivos()

    print(f"Arquivos encontrados: {len(arquivos)}")

    bases = []

    for arquivo in arquivos:
        try:
            bases.append(processar_arquivo(arquivo))
        except Exception as erro:
            raise RuntimeError(
                f"Erro ao processar {arquivo.name}"
            ) from erro

    dados = pd.concat(
        bases,
        ignore_index=True,
    )

    return dados.sort_values(
        ["codigo_wmo", "data_hora"]
    ).reset_index(drop=True)


def validar_dados(dados: pd.DataFrame) -> None:
    """Exibe verificações básicas da base meteorológica."""
    print(f"Estações: {dados['codigo_wmo'].nunique()}")
    print(f"Registros: {len(dados):,}")

    print("\nRegistros por ano:")
    print(
        dados["ano"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nValidação:")
    print(
        "Data/hora inválida:",
        dados["data_hora"].isna().sum(),
    )
    print(
        "Precipitação ausente:",
        dados["precipitacao_mm"].isna().sum(),
    )
    print(
        "Latitude ausente:",
        dados["latitude"].isna().sum(),
    )
    print(
        "Longitude ausente:",
        dados["longitude"].isna().sum(),
    )

    print("\nPeríodo:")
    print("Início:", dados["data_hora"].min())
    print("Fim:", dados["data_hora"].max())


def salvar_dados(dados: pd.DataFrame) -> None:
    """Salva a base consolidada do INMET."""
    ARQUIVO_INMET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dados.to_csv(
        ARQUIVO_INMET,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nArquivo criado: {ARQUIVO_INMET}")


def main() -> None:
    dados = preparar_inmet()

    validar_dados(dados)
    salvar_dados(dados)


if __name__ == "__main__":
    main()