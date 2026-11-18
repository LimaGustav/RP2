import pandas as pd

from config import (
    ARQUIVO_ACIDENTES,
    ARQUIVO_INMET,
    ARQUIVO_BASE_INTEGRADA,
)
from utils.integracao import integrar_bases
from utils.validacao import preparar_tipos


def carregar_bases():
    """Carrega as bases preparadas da PRF e do INMET."""
    for arquivo in [
        ARQUIVO_ACIDENTES,
        ARQUIVO_INMET,
    ]:
        if not arquivo.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {arquivo}"
            )

    acidentes = pd.read_csv(
        ARQUIVO_ACIDENTES
    )

    inmet = pd.read_csv(
        ARQUIVO_INMET
    )

    return acidentes, inmet


def validar_integracao(
    dados: pd.DataFrame,
) -> None:
    """Exibe o resumo da integração."""
    print("\nStatus da integração:")
    print(
        dados["status_integracao"]
        .value_counts(dropna=False)
    )

    print("\nDistância até estação:")
    print(
        dados["distancia_estacao_km"]
        .describe()
    )

    print("\nDiferença temporal:")
    print(
        dados["diferenca_tempo_min"]
        .describe()
    )


def salvar_dados(
    dados: pd.DataFrame,
) -> None:
    """Salva a base integrada."""
    ARQUIVO_BASE_INTEGRADA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dados.to_csv(
        ARQUIVO_BASE_INTEGRADA,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nArquivo criado: "
        f"{ARQUIVO_BASE_INTEGRADA}"
    )


def main() -> None:
    acidentes, inmet = carregar_bases()

    acidentes, inmet = preparar_tipos(
        acidentes,
        inmet,
    )

    base_integrada = integrar_bases(
        acidentes,
        inmet,
    )

    validar_integracao(base_integrada)
    salvar_dados(base_integrada)


if __name__ == "__main__":
    main()