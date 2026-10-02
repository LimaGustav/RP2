from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel


# =========================================================
# Configurações
# =========================================================

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1400)


# =========================================================
# Caminhos
# =========================================================

RAIZ = Path(__file__).resolve().parents[1]

CAMINHO_BASE = (
    RAIZ
    / "datasets"
    / "processed"
    / "base_integrada.csv"
)

SAIDA_BASE_MODELAGEM = (
    RAIZ
    / "datasets"
    / "processed"
    / "base_modelagem.csv"
)

SAIDA_RESULTADOS = (
    RAIZ
    / "datasets"
    / "processed"
    / "resultados_regressao_ordinal.csv"
)

SAIDA_SENSIBILIDADE = (
    RAIZ
    / "datasets"
    / "processed"
    / "sensibilidade_distancia.csv"
)


# =========================================================
# Parâmetros metodológicos
# =========================================================

# Análise principal:
# 30 km é usado como limite espacial principal.
# Limites de 40 e 50 km são avaliados depois como sensibilidade.
LIMITE_DISTANCIA_KM = 30.0

# A base atual já apresentou diferenças temporais de no máximo 30 min.
# Mantemos 60 min apenas como salvaguarda contra futuras integrações
# com lacunas horárias maiores.
LIMITE_TEMPO_MIN = 60.0


# =========================================================
# Funções auxiliares
# =========================================================

def formatar_significancia(p_valor):
    if pd.isna(p_valor):
        return ""

    if p_valor < 0.001:
        return "***"

    if p_valor < 0.01:
        return "**"

    if p_valor < 0.05:
        return "*"

    if p_valor < 0.1:
        return "."

    return "ns"


def extrair_tabela_modelo(resultado, nome_modelo):
    parametros = resultado.params
    erros = resultado.bse
    p_valores = resultado.pvalues
    intervalos = resultado.conf_int()

    linhas = []

    for variavel in parametros.index:

        # Ignora os pontos de corte do modelo ordinal
        if "/" in str(variavel) or str(variavel).startswith("cut"):
            continue

        coef = parametros[variavel]
        ep = erros.get(variavel, np.nan)
        p_val = p_valores.get(variavel, np.nan)

        ic_inf = (
            intervalos.loc[variavel, 0]
            if variavel in intervalos.index
            else np.nan
        )

        ic_sup = (
            intervalos.loc[variavel, 1]
            if variavel in intervalos.index
            else np.nan
        )

        linhas.append(
            {
                "modelo": nome_modelo,
                "variavel": variavel,
                "coeficiente": coef,
                "erro_padrao": ep,
                "p_valor": p_val,
                "sig": formatar_significancia(p_val),
                "odds_ratio": np.exp(coef),
                "ic_95_inf": np.exp(ic_inf),
                "ic_95_sup": np.exp(ic_sup),
            }
        )

    return pd.DataFrame(linhas)


def calcular_metricas(resultado, nome_modelo, n_obs):
    return {
        "modelo": nome_modelo,
        "n_obs": n_obs,
        "log_likelihood": resultado.llf,
        "aic": resultado.aic,
        "bic": getattr(resultado, "bic", np.nan),
    }


def agrupar_tracado_via(valor):
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


def preparar_dados_modelagem(dados_entrada, limite_distancia):
    dados_modelo = dados_entrada[
        (dados_entrada["status_integracao"] == "integrado")
        & dados_entrada["precipitacao_mm"].notna()
        & (dados_entrada["distancia_estacao_km"] <= limite_distancia)
        & (dados_entrada["diferenca_tempo_min"] <= LIMITE_TEMPO_MIN)
    ].copy()

    covariaveis_necessarias = [
        "classificacao_acidente",
        "fase_dia",
        "tipo_pista",
        "tracado_via",
        "ano",
        "mes",
    ]

    dados_modelo = dados_modelo.dropna(
        subset=covariaveis_necessarias
    ).copy()

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
        errors="coerce"
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

    # Referências explícitas
    dados_modelo["fase_dia"] = pd.Categorical(
        dados_modelo["fase_dia"],
        categories=[
            "Pleno dia",
            "Plena Noite",
            "Anoitecer",
            "Amanhecer",
        ],
        ordered=False
    )

    dados_modelo["tipo_pista"] = pd.Categorical(
        dados_modelo["tipo_pista"],
        categories=[
            "Dupla",
            "Simples",
            "Múltipla",
        ],
        ordered=False
    )

    dados_modelo["tracado_via"] = pd.Categorical(
        dados_modelo["tracado_via"],
        categories=[
            "Reta",
            "Curva",
            "Interseção",
            "Outros",
        ],
        ordered=False
    )

    # Remove qualquer registro que tenha virado NaN após padronização
    dados_modelo = dados_modelo.dropna(
        subset=[
            "gravidade_ordinal",
            "precipitacao_mm",
            "fase_dia",
            "tipo_pista",
            "tracado_via",
            "ano",
            "mes",
        ]
    ).copy()

    dados_modelo["gravidade_ordinal"] = pd.Categorical(
        dados_modelo["gravidade_ordinal"].astype(int),
        categories=[0, 1, 2],
        ordered=True
    )

    return dados_modelo


def criar_matriz_ajustada(dados_modelo, variavel_chuva):
    categoricas = [
        "fase_dia",
        "tipo_pista",
        "tracado_via",
        "ano",
        "mes",
    ]

    dummies = pd.get_dummies(
        dados_modelo[categoricas],
        drop_first=True,
        dtype=float
    )

    matriz = pd.concat(
        [
            dados_modelo[[variavel_chuva]].astype(float),
            dummies,
        ],
        axis=1
    )

    return matriz


# =========================================================
# Carregar base
# =========================================================

if not CAMINHO_BASE.exists():
    raise FileNotFoundError(
        f"Base integrada não encontrada em: {CAMINHO_BASE}\n"
        "Execute as etapas anteriores antes da modelagem."
    )

dados = pd.read_csv(CAMINHO_BASE)

print()
print("=" * 70)
print("REGRESSÃO LOGÍSTICA ORDINAL")
print("=" * 70)

print(f"\nQuantidade inicial de acidentes: {len(dados)}")


# =========================================================
# 1. Diagnóstico dos dados ausentes de precipitação
# =========================================================

print()
print("=" * 70)
print("DIAGNÓSTICO DOS DADOS AUSENTES DE PRECIPITAÇÃO")
print("=" * 70)

dados["precipitacao_ausente"] = (
    dados["precipitacao_mm"].isna()
).astype(int)

total_ausente = int(dados["precipitacao_ausente"].sum())

print(
    f"\nPrecipitação ausente: {total_ausente} "
    f"({total_ausente / len(dados) * 100:.2f}%)"
)


def imprimir_ausencia_por(coluna):
    if coluna not in dados.columns:
        return

    tabela = (
        dados.groupby(
            coluna,
            dropna=False
        )["precipitacao_ausente"]
        .agg(
            total="size",
            ausentes="sum"
        )
        .reset_index()
    )

    tabela["percentual_ausente"] = (
        tabela["ausentes"]
        / tabela["total"]
        * 100
    )

    print()
    print(f"Ausência por {coluna}:")
    print(
        tabela.to_string(
            index=False,
            formatters={
                "percentual_ausente":
                    lambda x: f"{x:.2f}%"
            }
        )
    )


for coluna in [
    "ano",
    "mes",
    "classificacao_acidente",
    "condicao_metereologica",
]:
    imprimir_ausencia_por(coluna)


# Faixas de distância
dados["faixa_distancia"] = pd.cut(
    dados["distancia_estacao_km"],
    bins=[
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        np.inf,
    ],
    labels=[
        "0-10",
        "10-20",
        "20-30",
        "30-40",
        "40-50",
        "50-60",
        ">60",
    ],
    include_lowest=True
)

imprimir_ausencia_por("faixa_distancia")


# Estações com maior ausência
if "estacao_inmet" in dados.columns:
    tabela_estacoes = (
        dados.groupby(
            "estacao_inmet",
            dropna=False
        )["precipitacao_ausente"]
        .agg(
            total="size",
            ausentes="sum"
        )
        .reset_index()
    )

    tabela_estacoes["percentual_ausente"] = (
        tabela_estacoes["ausentes"]
        / tabela_estacoes["total"]
        * 100
    )

    tabela_estacoes = tabela_estacoes.sort_values(
        [
            "percentual_ausente",
            "ausentes",
        ],
        ascending=[
            False,
            False,
        ]
    )

    print()
    print("15 estações com maior percentual de ausência:")
    print(
        tabela_estacoes.head(15).to_string(
            index=False,
            formatters={
                "percentual_ausente":
                    lambda x: f"{x:.2f}%"
            }
        )
    )


# Regressão logística para verificar se a gravidade
# permanece associada à ausência depois dos ajustes observados.
print()
print("-" * 70)
print("MODELO DA PROBABILIDADE DE PRECIPITAÇÃO AUSENTE")
print("-" * 70)

dados_ausencia = dados.copy()

dados_ausencia["condicao_metereologica"] = (
    dados_ausencia["condicao_metereologica"]
    .fillna("Ausente")
)

formula_ausencia = (
    "precipitacao_ausente ~ "
    "C(classificacao_acidente) + "
    "C(ano) + "
    "C(mes) + "
    "distancia_estacao_km + "
    "C(condicao_metereologica)"
)

try:
    modelo_ausencia = smf.logit(
        formula=formula_ausencia,
        data=dados_ausencia
    ).fit(
        disp=False,
        maxiter=500
    )

    resultado_ausencia = pd.DataFrame(
        {
            "odds_ratio":
                np.exp(modelo_ausencia.params),
            "p_valor":
                modelo_ausencia.pvalues,
        }
    )

    ic_ausencia = modelo_ausencia.conf_int()

    resultado_ausencia["ic_95_inf"] = (
        np.exp(ic_ausencia[0])
    )

    resultado_ausencia["ic_95_sup"] = (
        np.exp(ic_ausencia[1])
    )

    linhas_gravidade = [
        indice
        for indice in resultado_ausencia.index
        if "classificacao_acidente" in indice
    ]

    print(
        "\nTermos de gravidade no modelo de ausência:"
    )

    print(
        resultado_ausencia.loc[
            linhas_gravidade,
            [
                "odds_ratio",
                "ic_95_inf",
                "ic_95_sup",
                "p_valor",
            ]
        ].round(4)
    )

    gravidade_associada_ausencia = any(
        resultado_ausencia.loc[
            linhas_gravidade,
            "p_valor"
        ] < 0.05
    )

    print()
    if gravidade_associada_ausencia:
        print(
            "ATENÇÃO: há evidência de associação entre "
            "gravidade e ausência de precipitação após os ajustes."
        )
        print(
            "A análise por casos completos deve ser tratada "
            "com cautela e acompanhada de análise de sensibilidade."
        )
    else:
        print(
            "Não foi detectada associação estatisticamente "
            "significativa entre gravidade e ausência de "
            "precipitação após os ajustes utilizados."
        )
        print(
            "Isso favorece o uso pragmático de casos completos, "
            "mas não prova que os dados sejam MCAR/MAR."
        )

except Exception as erro:
    print(
        "\nNão foi possível ajustar o modelo de ausência:"
    )
    print(erro)


# =========================================================
# 2. Fluxo de seleção da análise principal
# =========================================================

print()
print("=" * 70)
print("FLUXO DE SELEÇÃO DA ANÁLISE PRINCIPAL")
print("=" * 70)

total_inicial = len(dados)

mascara_precipitacao = (
    (dados["status_integracao"] == "integrado")
    & dados["precipitacao_mm"].notna()
)

dados_etapa1 = dados[
    mascara_precipitacao
].copy()

perda_precipitacao = (
    total_inicial
    - len(dados_etapa1)
)

dados_etapa2 = dados_etapa1[
    dados_etapa1["distancia_estacao_km"]
    <= LIMITE_DISTANCIA_KM
].copy()

perda_distancia = (
    len(dados_etapa1)
    - len(dados_etapa2)
)

dados_etapa3 = dados_etapa2[
    dados_etapa2["diferenca_tempo_min"]
    <= LIMITE_TEMPO_MIN
].copy()

perda_tempo = (
    len(dados_etapa2)
    - len(dados_etapa3)
)

covariaveis_necessarias = [
    "classificacao_acidente",
    "fase_dia",
    "tipo_pista",
    "tracado_via",
    "ano",
    "mes",
]

dados_etapa4 = dados_etapa3.dropna(
    subset=covariaveis_necessarias
).copy()

perda_covariaveis = (
    len(dados_etapa3)
    - len(dados_etapa4)
)

fluxo_exclusao = pd.DataFrame(
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
            "etapa":
                f"Distância > {LIMITE_DISTANCIA_KM:.0f} km",
            "registros_restantes": len(dados_etapa2),
            "excluidos_etapa": perda_distancia,
        },
        {
            "etapa":
                f"Diferença temporal > {LIMITE_TEMPO_MIN:.0f} min",
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

fluxo_exclusao["percentual_amostra_inicial"] = (
    fluxo_exclusao["registros_restantes"]
    / total_inicial
    * 100
).round(2)

print()
print(
    fluxo_exclusao.to_string(
        index=False
    )
)


# =========================================================
# 3. Preparar amostra principal
# =========================================================

dados_modelo = preparar_dados_modelagem(
    dados,
    LIMITE_DISTANCIA_KM
)

print()
print("=" * 70)
print("AMOSTRA FINAL DE MODELAGEM")
print("=" * 70)

print(
    f"\nTamanho final da amostra: "
    f"{len(dados_modelo):,}"
)

print(
    "\nDistribuição da gravidade:"
)

resumo_gravidade = (
    dados_modelo[
        "classificacao_acidente"
    ]
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

print(
    "\nDistribuição por ano e gravidade:"
)

print(
    pd.crosstab(
        dados_modelo["ano"],
        dados_modelo[
            "classificacao_acidente"
        ],
        margins=True,
        margins_name="Total"
    )
)

print(
    "\nOcorrência de chuva (> 0 mm):"
)

resumo_chuva = (
    dados_modelo[
        "chuva_binaria"
    ]
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


# =========================================================
# 4. Matrizes dos modelos
# =========================================================

matriz_x_m1a = (
    dados_modelo[
        ["precipitacao_mm"]
    ]
    .astype(float)
)

matriz_x_m1b = (
    dados_modelo[
        ["chuva_binaria"]
    ]
    .astype(float)
)

matriz_x_m2a = criar_matriz_ajustada(
    dados_modelo,
    "precipitacao_mm"
)

matriz_x_m2b = criar_matriz_ajustada(
    dados_modelo,
    "chuva_binaria"
)

y_ordinal = (
    dados_modelo[
        "gravidade_ordinal"
    ]
)


# =========================================================
# 5. Modelos ordinais
# =========================================================

resultados_tabelas = []
metricas = []


def ajustar_modelo_ordinal(
    nome_modelo,
    matriz_x
):
    print()
    print("=" * 70)
    print(nome_modelo.upper())
    print("=" * 70)

    modelo = OrderedModel(
        y_ordinal,
        matriz_x,
        distr="logit"
    )

    resultado = modelo.fit(
        method="bfgs",
        disp=False,
        maxiter=1000
    )

    print(
        resultado.summary()
    )

    tabela = extrair_tabela_modelo(
        resultado,
        nome_modelo
    )

    print(
        "\nOdds Ratios:"
    )

    print(
        tabela[
            [
                "variavel",
                "coeficiente",
                "erro_padrao",
                "p_valor",
                "sig",
                "odds_ratio",
                "ic_95_inf",
                "ic_95_sup",
            ]
        ].to_string(
            index=False
        )
    )

    resultados_tabelas.append(
        tabela
    )

    metricas.append(
        calcular_metricas(
            resultado,
            nome_modelo,
            len(dados_modelo)
        )
    )

    return resultado


resultado_m1a = ajustar_modelo_ordinal(
    "Modelo 1A - Bruto - Precipitação mm",
    matriz_x_m1a
)

resultado_m1b = ajustar_modelo_ordinal(
    "Modelo 1B - Bruto - Chuva binária",
    matriz_x_m1b
)

resultado_m2a = ajustar_modelo_ordinal(
    "Modelo 2A - Ajustado - Precipitação mm",
    matriz_x_m2a
)

resultado_m2b = ajustar_modelo_ordinal(
    "Modelo 2B - Ajustado - Chuva binária",
    matriz_x_m2b
)


# =========================================================
# 6. Diagnóstico aproximado de chances proporcionais
# =========================================================

print()
print("=" * 70)
print("DIAGNÓSTICO DA SUPOSIÇÃO DE CHANCES PROPORCIONAIS")
print("=" * 70)

# Este é um diagnóstico baseado na comparação de modelos
# logísticos binários nos dois cortes cumulativos. Ele não
# substitui um teste formal como o teste de Brant.

gravidade_num = (
    dados_modelo[
        "gravidade_ordinal"
    ]
    .astype(int)
)

corte_1 = (
    gravidade_num >= 1
).astype(int)

corte_2 = (
    gravidade_num >= 2
).astype(int)

matriz_x_const = sm.add_constant(
    matriz_x_m2a
)

logit_corte1 = sm.Logit(
    corte_1,
    matriz_x_const
).fit(
    disp=False
)

logit_corte2 = sm.Logit(
    corte_2,
    matriz_x_const
).fit(
    disp=False
)

comparacao_cortes = []

for variavel in [
    v
    for v in logit_corte1.params.index
    if v != "const"
]:
    b1 = (
        logit_corte1.params.get(
            variavel,
            np.nan
        )
    )

    b2 = (
        logit_corte2.params.get(
            variavel,
            np.nan
        )
    )

    bord = (
        resultado_m2a.params.get(
            variavel,
            np.nan
        )
    )

    comparacao_cortes.append(
        {
            "variavel": variavel,
            "beta_corte1": b1,
            "or_corte1": np.exp(b1),
            "p_corte1":
                logit_corte1.pvalues.get(
                    variavel,
                    np.nan
                ),
            "beta_corte2": b2,
            "or_corte2": np.exp(b2),
            "p_corte2":
                logit_corte2.pvalues.get(
                    variavel,
                    np.nan
                ),
            "beta_ordinal": bord,
            "or_ordinal":
                np.exp(bord)
                if not np.isnan(bord)
                else np.nan,
        }
    )

df_proporcional = pd.DataFrame(
    comparacao_cortes
)

print(
    df_proporcional.round(4)
    .to_string(
        index=False
    )
)


# =========================================================
# 7. Sensibilidade ao limite espacial
# =========================================================

print()
print("=" * 70)
print("ANÁLISE DE SENSIBILIDADE DO LIMITE ESPACIAL")
print("=" * 70)

sensibilidade = []

for limite in [
    30.0,
    40.0,
    50.0,
]:
    dados_sensibilidade = preparar_dados_modelagem(
        dados,
        limite
    )

    if dados_sensibilidade.empty:
        continue

    y_sensibilidade = (
        dados_sensibilidade[
            "gravidade_ordinal"
        ]
    )

    x_mm = criar_matriz_ajustada(
        dados_sensibilidade,
        "precipitacao_mm"
    )

    x_bin = criar_matriz_ajustada(
        dados_sensibilidade,
        "chuva_binaria"
    )

    modelo_mm = OrderedModel(
        y_sensibilidade,
        x_mm,
        distr="logit"
    ).fit(
        method="bfgs",
        disp=False,
        maxiter=1000
    )

    modelo_bin = OrderedModel(
        y_sensibilidade,
        x_bin,
        distr="logit"
    ).fit(
        method="bfgs",
        disp=False,
        maxiter=1000
    )

    for nome_variavel, modelo in [
        (
            "precipitacao_mm",
            modelo_mm
        ),
        (
            "chuva_binaria",
            modelo_bin
        ),
    ]:
        coef = (
            modelo.params[
                nome_variavel
            ]
        )

        p_val = (
            modelo.pvalues[
                nome_variavel
            ]
        )

        ic = (
            modelo.conf_int()
            .loc[
                nome_variavel
            ]
        )

        sensibilidade.append(
            {
                "limite_km": limite,
                "n": len(
                    dados_sensibilidade
                ),
                "exposicao":
                    nome_variavel,
                "odds_ratio":
                    np.exp(coef),
                "ic_95_inf":
                    np.exp(ic.iloc[0]),
                "ic_95_sup":
                    np.exp(ic.iloc[1]),
                "p_valor":
                    p_val,
            }
        )

df_sensibilidade = pd.DataFrame(
    sensibilidade
)

print(
    df_sensibilidade.round(4)
    .to_string(
        index=False
    )
)


# =========================================================
# 8. Comparação de ajuste
# =========================================================

print()
print("=" * 70)
print("COMPARAÇÃO DE AJUSTE DOS MODELOS")
print("=" * 70)

tabela_metricas = pd.DataFrame(
    metricas
)

print(
    tabela_metricas.to_string(
        index=False
    )
)


# =========================================================
# 9. Salvar resultados
# =========================================================

todas_tabelas = pd.concat(
    resultados_tabelas,
    ignore_index=True
)

SAIDA_BASE_MODELAGEM.parent.mkdir(
    parents=True,
    exist_ok=True
)

dados_modelo.to_csv(
    SAIDA_BASE_MODELAGEM,
    index=False,
    encoding="utf-8-sig"
)

todas_tabelas.to_csv(
    SAIDA_RESULTADOS,
    index=False,
    encoding="utf-8-sig"
)

df_sensibilidade.to_csv(
    SAIDA_SENSIBILIDADE,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 70)
print("FINALIZAÇÃO")
print("=" * 70)

print(
    f"\nBase de modelagem salva em:\n"
    f"{SAIDA_BASE_MODELAGEM}"
)

print(
    f"\nResultados principais salvos em:\n"
    f"{SAIDA_RESULTADOS}"
)

print(
    f"\nSensibilidade espacial salva em:\n"
    f"{SAIDA_SENSIBILIDADE}"
)

print(
    "\nExecução concluída."
)
