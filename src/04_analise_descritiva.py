import pandas as pd

from analises.descritiva import (
    executar_analise_descritiva,
)
from analises.qualidade_integracao import (
    executar_analise_qualidade,
)
from config import ARQUIVO_BASE_INTEGRADA


def carregar_dados():
    if not ARQUIVO_BASE_INTEGRADA.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: "
            f"{ARQUIVO_BASE_INTEGRADA}"
        )

    return pd.read_csv(
        ARQUIVO_BASE_INTEGRADA
    )


def main():
    dados = carregar_dados()

    dados_validos = (
        executar_analise_descritiva(
            dados
        )
    )

    executar_analise_qualidade(
        dados,
        dados_validos,
    )


if __name__ == "__main__":
    main()