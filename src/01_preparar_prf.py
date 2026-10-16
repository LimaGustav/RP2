import pandas as pd

from config import ARQUIVOS_PRF, ARQUIVO_ACIDENTES


COLUNAS_PRF = [
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
]

MAPA_GRAVIDADE = {
    "Sem Vítimas": 0,
    "Com Vítimas Feridas": 1,
    "Com Vítimas Fatais": 2,
}


def carregar_dados() -> pd.DataFrame:
    """Carrega e combina os arquivos anuais da PRF."""
    bases = []

    for arquivo in ARQUIVOS_PRF:
        if not arquivo.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {arquivo}")

        base = pd.read_csv(
            arquivo,
            sep=";",
            encoding="latin1",
        )

        bases.append(base)

    return pd.concat(bases, ignore_index=True)


def converter_datas(datas: pd.Series) -> pd.Series:
    """
    Converte os formatos utilizados nas bases da PRF.

    2025: YYYY-MM-DD
    2026: DD/MM/YYYY
    """
    texto = datas.astype("string").str.strip()

    datas_iso = pd.to_datetime(
        texto,
        format="%Y-%m-%d",
        errors="coerce",
    )

    datas_br = pd.to_datetime(
        texto,
        format="%d/%m/%Y",
        errors="coerce",
    )

    return datas_iso.fillna(datas_br)


def converter_coordenadas(dados: pd.DataFrame) -> None:
    """Converte latitude e longitude para valores numéricos."""
    for coluna in ["latitude", "longitude"]:
        dados[coluna] = pd.to_numeric(
            dados[coluna]
            .astype("string")
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )


def preparar_dados(dados: pd.DataFrame) -> pd.DataFrame:
    """Filtra e padroniza os registros utilizados no projeto."""
    dados = dados.loc[
        dados["uf"].eq("SP"),
        COLUNAS_PRF,
    ].copy()

    dados["data_inversa"] = converter_datas(
        dados["data_inversa"]
    )

    dados["ano"] = (
        dados["data_inversa"]
        .dt.year
    )

    dados["mes"] = (
        dados["data_inversa"]
        .dt.month
    )

    dados["horario"] = (
        dados["horario"]
        .astype("string")
        .str.strip()
    )

    horario_delta = pd.to_timedelta(
        dados["horario"],
        errors="coerce",
    )

    dados["data_hora"] = (
        dados["data_inversa"]
        + horario_delta
    )

    converter_coordenadas(dados)

    dados["gravidade"] = (
        dados["classificacao_acidente"]
        .map(MAPA_GRAVIDADE)
    )

    return dados

def validar_dados(dados: pd.DataFrame) -> None:
    """Exibe verificações básicas da base processada."""
    horario_invalido = pd.to_timedelta(
        dados["horario"],
        errors="coerce",
    ).isna().sum()

    print(f"Acidentes em SP: {len(dados):,}")

    print("\nAcidentes por ano:")
    print(
        dados["data_inversa"]
        .dt.year
        .value_counts()
        .sort_index()
    )

    print("\nGravidade:")
    print(
        dados["classificacao_acidente"]
        .value_counts(dropna=False)
    )

    print("\nValidação:")
    print("Datas inválidas:", dados["data_inversa"].isna().sum())
    print("Horários inválidos:", horario_invalido)
    print("Data/hora inválida:", dados["data_hora"].isna().sum())
    print("Latitudes inválidas:", dados["latitude"].isna().sum())
    print("Longitudes inválidas:", dados["longitude"].isna().sum())
    print("Gravidades inválidas:", dados["gravidade"].isna().sum())
    print("IDs duplicados:", dados["id"].duplicated().sum())

    print("\nPeríodo:")
    print("Início:", dados["data_hora"].min())
    print("Fim:", dados["data_hora"].max())


def salvar_dados(dados: pd.DataFrame) -> None:
    """Salva a base tratada da PRF."""
    ARQUIVO_ACIDENTES.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dados.to_csv(
        ARQUIVO_ACIDENTES,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nArquivo criado: {ARQUIVO_ACIDENTES}")


def main() -> None:
    dados = carregar_dados()
    dados = preparar_dados(dados)

    validar_dados(dados)
    salvar_dados(dados)


if __name__ == "__main__":
    main()