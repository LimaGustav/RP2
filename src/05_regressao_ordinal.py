import pandas as pd

from config import (
    ARQUIVO_BASE_INTEGRADA,
    LIMITE_DISTANCIA_KM,
    LIMITE_TEMPO_MIN,
    PASTA_DADOS_PROCESSADOS,
)
from modelos.diagnosticos import (
    diagnosticar_ausencia_precipitacao,
    diagnosticar_chances_proporcionais,
    preparar_variaveis_ausencia,
)
from modelos.multinomial import (
    executar_modelos_multinomiais,
)
from modelos.ordinal import (
    executar_modelos_ordinais,
    executar_sensibilidade_espacial,
)
from modelos.preparacao import (
    calcular_fluxo_selecao,
    imprimir_resumo_amostra,
    preparar_dados_modelagem,
)


LIMITES_SENSIBILIDADE_KM = [
    30.0,
    40.0,
    50.0,
]

MIN_REGISTROS_ESTACAO_MODELO_AUSENCIA = 20

SAIDA_BASE_MODELAGEM = (
    PASTA_DADOS_PROCESSADOS
    / "base_modelagem.csv"
)

SAIDA_RESULTADOS_ORDINAL = (
    PASTA_DADOS_PROCESSADOS
    / "resultados_regressao_ordinal.csv"
)

SAIDA_SENSIBILIDADE = (
    PASTA_DADOS_PROCESSADOS
    / "sensibilidade_distancia.csv"
)

SAIDA_RESULTADOS_MULTINOMIAL = (
    PASTA_DADOS_PROCESSADOS
    / "resultados_regressao_multinomial.csv"
)

SAIDA_DIAGNOSTICO_AUSENCIA = (
    PASTA_DADOS_PROCESSADOS
    / "diagnostico_ausencia_precipitacao.csv"
)


def carregar_dados():
    if not ARQUIVO_BASE_INTEGRADA.exists():
        raise FileNotFoundError(
            f"Base integrada não encontrada em: "
            f"{ARQUIVO_BASE_INTEGRADA}\n"
            "Execute as etapas anteriores antes da modelagem."
        )

    return pd.read_csv(
        ARQUIVO_BASE_INTEGRADA
    )


def salvar_resultados(
    dados_modelo,
    tabela_ordinais,
    tabela_multinomial,
    tabela_sensibilidade,
    resultado_ausencia,
):
    PASTA_DADOS_PROCESSADOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    dados_modelo.to_csv(
        SAIDA_BASE_MODELAGEM,
        index=False,
        encoding="utf-8-sig",
    )

    tabela_ordinais.to_csv(
        SAIDA_RESULTADOS_ORDINAL,
        index=False,
        encoding="utf-8-sig",
    )

    tabela_sensibilidade.to_csv(
        SAIDA_SENSIBILIDADE,
        index=False,
        encoding="utf-8-sig",
    )

    tabela_multinomial.to_csv(
        SAIDA_RESULTADOS_MULTINOMIAL,
        index=False,
        encoding="utf-8-sig",
    )

    if not resultado_ausencia.empty:
        resultado_ausencia.to_csv(
            SAIDA_DIAGNOSTICO_AUSENCIA,
            index=True,
            encoding="utf-8-sig",
        )


def imprimir_finalizacao():
    print()
    print("=" * 72)
    print("FINALIZAÇÃO")
    print("=" * 72)

    print(
        f"\nBase de modelagem:\n"
        f"{SAIDA_BASE_MODELAGEM}"
    )

    print(
        f"\nResultados ordinais:\n"
        f"{SAIDA_RESULTADOS_ORDINAL}"
    )

    print(
        f"\nResultados multinomiais:\n"
        f"{SAIDA_RESULTADOS_MULTINOMIAL}"
    )

    print(
        f"\nSensibilidade espacial:\n"
        f"{SAIDA_SENSIBILIDADE}"
    )

    print(
        f"\nDiagnóstico de ausência:\n"
        f"{SAIDA_DIAGNOSTICO_AUSENCIA}"
    )

    print("\nExecução concluída.")


def main():
    dados = carregar_dados()

    print()
    print("=" * 72)
    print(
        "ANÁLISE DA ASSOCIAÇÃO ENTRE PRECIPITAÇÃO E GRAVIDADE"
    )
    print("=" * 72)

    print(
        f"\nQuantidade inicial de acidentes: "
        f"{len(dados):,}"
    )

    dados = preparar_variaveis_ausencia(dados)

    resultado_ausencia = (
        diagnosticar_ausencia_precipitacao(
            dados,
            min_registros_estacao=(
                MIN_REGISTROS_ESTACAO_MODELO_AUSENCIA
            ),
        )
    )

    print()
    print("=" * 72)
    print("FLUXO DE SELEÇÃO DA ANÁLISE PRINCIPAL")
    print("=" * 72)

    fluxo_exclusao = calcular_fluxo_selecao(
        dados,
        LIMITE_DISTANCIA_KM,
        LIMITE_TEMPO_MIN,
    )

    print()
    print(
        fluxo_exclusao.to_string(
            index=False
        )
    )

    dados_modelo = preparar_dados_modelagem(
        dados,
        LIMITE_DISTANCIA_KM,
        LIMITE_TEMPO_MIN,
    )

    imprimir_resumo_amostra(
        dados_modelo
    )

    ordinais = executar_modelos_ordinais(
        dados_modelo
    )

    diagnosticar_chances_proporcionais(
        dados_modelo,
        ordinais["matrizes"]["m2a"],
        ordinais["resultados"]["m2a"],
    )

    tabela_multinomial = (
        executar_modelos_multinomiais(
            dados_modelo,
            ordinais["matrizes"]["m2a"],
            ordinais["matrizes"]["m2b"],
        )
    )

    tabela_sensibilidade = (
        executar_sensibilidade_espacial(
            dados,
            LIMITES_SENSIBILIDADE_KM,
            LIMITE_TEMPO_MIN,
        )
    )

    print()
    print("=" * 72)
    print("COMPARAÇÃO DE AJUSTE DOS MODELOS ORDINAIS")
    print("=" * 72)
    print(
        ordinais["metricas"]
        .to_string(index=False)
    )

    salvar_resultados(
        dados_modelo=dados_modelo,
        tabela_ordinais=ordinais["tabela"],
        tabela_multinomial=tabela_multinomial,
        tabela_sensibilidade=tabela_sensibilidade,
        resultado_ausencia=resultado_ausencia,
    )

    imprimir_finalizacao()


if __name__ == "__main__":
    with pd.option_context("display.max_columns", None, "display.width", 1600):
        main()
