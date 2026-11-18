import numpy as np
import pandas as pd


COVARIAVEIS_NECESSARIAS = [
    "classificacao_acidente",
    "fase_dia",
    "tipo_pista",
    "tracado_via",
    "ano",
    "mes",
]


def agrupar_tracado_via(valor):
    """Agrupa as categorias de traçado usadas nos modelos."""
    if pd.isna(valor):
        return np.nan

    texto = str(valor).strip()

    if "Curva" in texto:
        return "Curva"

    if "Reta" in texto:
        return "Reta"

    if any(
        chave in texto
        for chave in [
            "Interseção",
            "Rotatória",
            "Cruzamento",
            "Retorno",
        ]
    ):
        return "Interseção"

    return "Outros"


def remover_colunas_sem_variacao(matriz):
    """Remove preditores constantes, que não podem ser estimados."""
    colunas_validas = [
        coluna
        for coluna in matriz.columns
        if matriz[coluna].nunique(dropna=False) > 1
    ]

    return matriz[colunas_validas].copy()


def calcular_fluxo_selecao(
    dados,
    limite_distancia_km,
    limite_tempo_min,
):
    """Resume as perdas em cada etapa até a amostra principal."""
    total_inicial = len(dados)

    mascara_precipitacao = (
        (dados["status_integracao"] == "integrado")
        & dados["precipitacao_mm"].notna()
    )

    dados_etapa1 = dados[mascara_precipitacao].copy()
    perda_precipitacao = total_inicial - len(dados_etapa1)

    dados_etapa2 = dados_etapa1[
        dados_etapa1["distancia_estacao_km"] <= limite_distancia_km
    ].copy()
    perda_distancia = len(dados_etapa1) - len(dados_etapa2)

    dados_etapa3 = dados_etapa2[
        dados_etapa2["diferenca_tempo_min"] <= limite_tempo_min
    ].copy()
    perda_tempo = len(dados_etapa2) - len(dados_etapa3)

    dados_etapa4 = (
        dados_etapa3
        .dropna(subset=COVARIAVEIS_NECESSARIAS)
        .copy()
    )
    perda_covariaveis = len(dados_etapa3) - len(dados_etapa4)

    fluxo = pd.DataFrame(
        [
            {
                "etapa": "Base integrada inicial",
                "registros_restantes": total_inicial,
                "excluidos_etapa": 0,
            },
            {
                "etapa": "Sem medição de precipitação",
                "registros_restantes": len(dados_etapa1),
                "excluidos_etapa": perda_precipitacao,
            },
            {
                "etapa": f"Distância > {limite_distancia_km:.0f} km",
                "registros_restantes": len(dados_etapa2),
                "excluidos_etapa": perda_distancia,
            },
            {
                "etapa": f"Diferença temporal > {limite_tempo_min:.0f} min",
                "registros_restantes": len(dados_etapa3),
                "excluidos_etapa": perda_tempo,
            },
            {
                "etapa": "Ausência nas covariáveis",
                "registros_restantes": len(dados_etapa4),
                "excluidos_etapa": perda_covariaveis,
            },
        ]
    )

    fluxo["percentual_amostra_inicial"] = (
        fluxo["registros_restantes"]
        / total_inicial
        * 100
    ).round(2)

    return fluxo


def preparar_dados_modelagem(
    dados_entrada,
    limite_distancia_km,
    limite_tempo_min,
):
    """Cria a amostra e as variáveis usadas nos modelos estatísticos."""
    dados_modelo = dados_entrada[
        (dados_entrada["status_integracao"] == "integrado")
        & dados_entrada["precipitacao_mm"].notna()
        & (
            dados_entrada["distancia_estacao_km"]
            <= limite_distancia_km
        )
        & (
            dados_entrada["diferenca_tempo_min"]
            <= limite_tempo_min
        )
    ].copy()

    dados_modelo = (
        dados_modelo
        .dropna(subset=COVARIAVEIS_NECESSARIAS)
        .copy()
    )

    mapa_gravidade = {
        "Sem Vítimas": 0,
        "Com Vítimas Feridas": 1,
        "Com Vítimas Fatais": 2,
    }

    dados_modelo["gravidade_ordinal"] = (
        dados_modelo["classificacao_acidente"]
        .map(mapa_gravidade)
    )

    dados_modelo["precipitacao_mm"] = pd.to_numeric(
        dados_modelo["precipitacao_mm"],
        errors="coerce",
    )

    dados_modelo["chuva_binaria"] = (
        dados_modelo["precipitacao_mm"] > 0
    ).astype(int)

    dados_modelo["fase_dia"] = (
        dados_modelo["fase_dia"]
        .astype("string")
        .str.strip()
        .replace(
            {
                "Plena noite": "Plena Noite",
                "plena noite": "Plena Noite",
                "pleno dia": "Pleno dia",
            }
        )
    )

    dados_modelo["tipo_pista"] = (
        dados_modelo["tipo_pista"]
        .astype("string")
        .str.strip()
    )

    dados_modelo["tracado_via"] = (
        dados_modelo["tracado_via"]
        .apply(agrupar_tracado_via)
    )

    dados_modelo["fase_dia"] = pd.Categorical(
        dados_modelo["fase_dia"],
        categories=[
            "Pleno dia",
            "Plena Noite",
            "Anoitecer",
            "Amanhecer",
        ],
        ordered=False,
    )

    dados_modelo["tipo_pista"] = pd.Categorical(
        dados_modelo["tipo_pista"],
        categories=[
            "Dupla",
            "Simples",
            "Múltipla",
        ],
        ordered=False,
    )

    dados_modelo["tracado_via"] = pd.Categorical(
        dados_modelo["tracado_via"],
        categories=[
            "Reta",
            "Curva",
            "Interseção",
            "Outros",
        ],
        ordered=False,
    )

    anos = sorted(
        dados_modelo["ano"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    dados_modelo["ano"] = pd.Categorical(
        dados_modelo["ano"].astype(int),
        categories=anos,
        ordered=False,
    )

    dados_modelo["mes"] = pd.Categorical(
        dados_modelo["mes"].astype(int),
        categories=list(range(1, 13)),
        ordered=False,
    )

    dados_modelo = (
        dados_modelo
        .dropna(
            subset=[
                "gravidade_ordinal",
                "precipitacao_mm",
                "fase_dia",
                "tipo_pista",
                "tracado_via",
                "ano",
                "mes",
            ]
        )
        .copy()
    )

    dados_modelo["gravidade_ordinal"] = pd.Categorical(
        dados_modelo["gravidade_ordinal"].astype(int),
        categories=[0, 1, 2],
        ordered=True,
    )

    return dados_modelo


def criar_matriz_ajustada(
    dados_modelo,
    variavel_chuva,
):
    """Cria a matriz de preditores dos modelos ajustados."""
    categoricas = [
        "fase_dia",
        "tipo_pista",
        "tracado_via",
        "ano",
        "mes",
    ]

    dummies = pd.get_dummies(
        dados_modelo[categoricas],
        columns=categoricas,
        drop_first=True,
        dtype=float,
    )

    matriz = pd.concat(
        [
            dados_modelo[[variavel_chuva]].astype(float),
            dummies,
        ],
        axis=1,
    )

    return remover_colunas_sem_variacao(matriz)


def imprimir_resumo_amostra(dados_modelo):
    """Mostra a composição da amostra final usada na modelagem."""
    print()
    print("=" * 72)
    print("AMOSTRA FINAL DE MODELAGEM")
    print("=" * 72)

    print(
        f"\nTamanho final da amostra: "
        f"{len(dados_modelo):,}"
    )

    print("\nDistribuição da gravidade:")

    resumo_gravidade = (
        dados_modelo["classificacao_acidente"]
        .value_counts()
        .rename("acidentes")
        .to_frame()
    )

    resumo_gravidade["percentual"] = (
        resumo_gravidade["acidentes"]
        / len(dados_modelo)
        * 100
    ).round(2)

    print(resumo_gravidade)

    print("\nDistribuição por ano e gravidade:")
    print(
        pd.crosstab(
            dados_modelo["ano"],
            dados_modelo["classificacao_acidente"],
            margins=True,
            margins_name="Total",
        )
    )

    print("\nOcorrência de chuva (> 0 mm):")

    resumo_chuva = (
        dados_modelo["chuva_binaria"]
        .value_counts()
        .rename(
            {
                0: "Sem Chuva",
                1: "Com Chuva",
            }
        )
        .to_frame("acidentes")
    )

    resumo_chuva["percentual"] = (
        resumo_chuva["acidentes"]
        / len(dados_modelo)
        * 100
    ).round(2)

    print(resumo_chuva)
