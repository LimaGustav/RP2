from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel


# =========================================================
# Configurações de exibição e avisos
# =========================================================

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


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


# =========================================================
# Parâmetros metodológicos (limites espacial e temporal)
#
# LIMITE_DISTANCIA_KM:
# Raio máximo de distância entre o local do acidente e
# a estação meteorológica do INMET. 30 km é um corte
# representativo para microclima e precipitação convectiva.
#
# LIMITE_TEMPO_MIN:
# Janela temporal máxima entre o acidente e a medição
# meteorológica. Como o INMET registra dados horários,
# acidentes com medições acima de 60 minutos indicam
# falha ou ausência de medição horária naquela estação.
# =========================================================

LIMITE_DISTANCIA_KM = 30.0
LIMITE_TEMPO_MIN = 60.0


# =========================================================
# Carregar base integrada
# =========================================================

if not CAMINHO_BASE.exists():
    raise FileNotFoundError(
        f"Base integrada não encontrada em: {CAMINHO_BASE}\n"
        "Execute os passos anteriores (01, 02 e 03) antes de rodar a modelagem."
    )

dados = pd.read_csv(
    CAMINHO_BASE
)


# =========================================================
# Início da execução
# =========================================================

print()
print("=" * 60)
print("REGRESSÃO LOGÍSTICA ORDINAL")
print("=" * 60)

total_inicial = len(dados)

print(
    "\nQuantidade inicial de acidentes na base integrada:"
)

print(
    total_inicial
)


# =========================================================
# Critérios de exclusão e seleção amostral
# =========================================================

# 1. Status de integração e disponibilidade de precipitação
mascara_precipitacao = (
    (dados["status_integracao"] == "integrado")
    & dados["precipitacao_mm"].notna()
)
dados_etapa1 = dados[mascara_precipitacao].copy()
perda_precipitacao = total_inicial - len(dados_etapa1)

# 2. Limite espacial (distância máxima da estação)
mascara_distancia = (
    dados_etapa1["distancia_estacao_km"] <= LIMITE_DISTANCIA_KM
)
dados_etapa2 = dados_etapa1[mascara_distancia].copy()
perda_distancia = len(dados_etapa1) - len(dados_etapa2)

# 3. Limite temporal (diferença temporal máxima)
mascara_tempo = (
    dados_etapa2["diferenca_tempo_min"] <= LIMITE_TEMPO_MIN
)
dados_etapa3 = dados_etapa2[mascara_tempo].copy()
perda_tempo = len(dados_etapa2) - len(dados_etapa3)

# 4. Dados válidos nas variáveis do modelo
covariaveis_necessarias = [
    "classificacao_acidente",
    "fase_dia",
    "tipo_pista",
    "tracado_via",
]

mascara_completude = dados_etapa3[covariaveis_necessarias].notna().all(axis=1)
dados_modelo = dados_etapa3[mascara_completude].copy()
perda_covariaveis = len(dados_etapa3) - len(dados_modelo)

total_final = len(dados_modelo)


# =========================================================
# Resumo do fluxo de exclusão amostral
# =========================================================

fluxo_exclusao = pd.DataFrame(
    [
        {
            "etapa": "1. Base integrada inicial",
            "registros_restantes": total_inicial,
            "excluidos_etapa": 0,
        },
        {
            "etapa": "2. Exclusão: sem medição de precipitação",
            "registros_restantes": len(dados_etapa1),
            "excluidos_etapa": perda_precipitacao,
        },
        {
            "etapa": f"3. Exclusão: distância > {LIMITE_DISTANCIA_KM} km",
            "registros_restantes": len(dados_etapa2),
            "excluidos_etapa": perda_distancia,
        },
        {
            "etapa": f"4. Exclusão: diferença tempo > {LIMITE_TEMPO_MIN} min",
            "registros_restantes": len(dados_etapa3),
            "excluidos_etapa": perda_tempo,
        },
        {
            "etapa": "5. Exclusão: dados ausentes nas covariáveis",
            "registros_restantes": total_final,
            "excluidos_etapa": perda_covariaveis,
        },
    ]
)

fluxo_exclusao["percentual_amostra_inicial"] = (
    fluxo_exclusao["registros_restantes"]
    / total_inicial
    * 100
).round(2)

print(
    "\nFluxo dos critérios de exclusão adotados:"
)

print(
    fluxo_exclusao.to_string(
        index=False
    )
)


# =========================================================
# Preparação e engenharia de variáveis
# =========================================================

# 1. Variável Dependente: Gravidade Ordinal
# 0 = Sem Vítimas
# 1 = Com Vítimas Feridas
# 2 = Com Vítimas Fatais
mapa_gravidade = {
    "Sem Vítimas": 0,
    "Com Vítimas Feridas": 1,
    "Com Vítimas Fatais": 2,
}

if "gravidade_ordinal" not in dados_modelo.columns:
    dados_modelo["gravidade_ordinal"] = (
        dados_modelo["classificacao_acidente"]
        .map(mapa_gravidade)
    )

dados_modelo["gravidade_ordinal"] = pd.to_numeric(
    dados_modelo["gravidade_ordinal"],
    errors="coerce"
).astype(int)

dados_modelo["gravidade_ordinal"] = pd.Categorical(
    dados_modelo["gravidade_ordinal"],
    categories=[0, 1, 2],
    ordered=True
)

# 2. Variáveis de Precipitação:
# - precipitação contínua (mm)
# - chuva binária (0 = sem chuva, 1 = com chuva)
dados_modelo["precipitacao_mm"] = pd.to_numeric(
    dados_modelo["precipitacao_mm"],
    errors="coerce"
)

dados_modelo["chuva_binaria"] = (
    dados_modelo["precipitacao_mm"] > 0
).astype(int)

# 3. Covariável: Fase do Dia
# Padronização e definição de referência: "Pleno dia"
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

ordem_fase_dia = [
    "Pleno dia",
    "Plena Noite",
    "Anoitecer",
    "Amanhecer",
]

dados_modelo["fase_dia"] = pd.Categorical(
    dados_modelo["fase_dia"],
    categories=ordem_fase_dia,
    ordered=False
)

# 4. Covariável: Tipo de Pista
# Padronização e definição de referência: "Dupla"
dados_modelo["tipo_pista"] = (
    dados_modelo["tipo_pista"]
    .astype("string")
    .str.strip()
)

ordem_tipo_pista = [
    "Dupla",
    "Simples",
    "Múltipla",
]

dados_modelo["tipo_pista"] = pd.Categorical(
    dados_modelo["tipo_pista"],
    categories=ordem_tipo_pista,
    ordered=False
)

# 5. Covariável: Traçado da Via
# Como a PRF combina múltiplos atributos (ex: 'Reta;Aclive', 'Curva;Declive'),
# agregamos nas classes morfológicas principais para evitar esparsidade:
# Reta (referência), Curva, Interseção, Outros.
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


dados_modelo["tracado_via"] = (
    dados_modelo["tracado_via"]
    .apply(agrupar_tracado_via)
)

ordem_tracado = [
    "Reta",
    "Curva",
    "Interseção",
    "Outros",
]

dados_modelo["tracado_via"] = pd.Categorical(
    dados_modelo["tracado_via"],
    categories=ordem_tracado,
    ordered=False
)


# =========================================================
# Descrição da amostra final de modelagem
# =========================================================

print()
print("=" * 60)
print("AMOSTRA FINAL DE MODELAGEM")
print("=" * 60)

print(
    f"\nTamanho final da amostra (N): {len(dados_modelo):,}"
)

print(
    "\nDistribuição da gravidade dos acidentes:"
)

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

print(
    resumo_gravidade
)

if "ano" in dados_modelo.columns:

    print(
        "\nDistribuição por ano e gravidade:"
    )

    print(
        pd.crosstab(
            dados_modelo["ano"],
            dados_modelo["classificacao_acidente"],
            margins=True,
            margins_name="Total"
        )
    )

print(
    "\nDistribuição da ocorrência de chuva (precipitação > 0 mm):"
)

resumo_chuva = (
    dados_modelo["chuva_binaria"]
    .value_counts()
    .rename({0: "Sem Chuva", 1: "Com Chuva"})
    .to_frame("acidentes")
)

resumo_chuva["percentual"] = (
    resumo_chuva["acidentes"]
    / len(dados_modelo)
    * 100
).round(2)

print(
    resumo_chuva
)

print(
    "\nEstatísticas da distância (km) e diferença de tempo (min):"
)

print(
    dados_modelo[
        [
            "distancia_estacao_km",
            "diferenca_tempo_min",
            "precipitacao_mm"
        ]
    ]
    .describe()
    .round(2)
)


# =========================================================
# Matrizes de variáveis explicativas e covariáveis
#
# Para evitar dependência de fórmulas patsy que podem
# inadvertidamente criar constantes implícitas com C(),
# codificamos as variáveis dummy explicitamente com
# drop_first=True, garantindo a categoria de referência
# e ausência de termo constante no OrderedModel.
# =========================================================

variaveis_categoricas = [
    "fase_dia",
    "tipo_pista",
    "tracado_via",
]

dummies_covariaveis = pd.get_dummies(
    dados_modelo[variaveis_categoricas],
    drop_first=True,
    dtype=float
)

matriz_x_m1a = (
    dados_modelo[["precipitacao_mm"]]
    .astype(float)
)

matriz_x_m1b = (
    dados_modelo[["chuva_binaria"]]
    .astype(float)
)

matriz_x_m2a = pd.concat(
    [
        matriz_x_m1a,
        dummies_covariaveis,
    ],
    axis=1
)

matriz_x_m2b = pd.concat(
    [
        matriz_x_m1b,
        dummies_covariaveis,
    ],
    axis=1
)

y_ordinal = dados_modelo["gravidade_ordinal"]


# =========================================================
# Funções auxiliares para extração de resultados
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
    """
    Extrai coeficientes, erros-padrão, valores-p,
    Odds Ratios (razões de chances) e IC de 95%.
    """
    parametros = resultado.params
    erros = resultado.bse
    p_valores = resultado.pvalues
    intervalos = resultado.conf_int()

    linhas = []

    for variavel in parametros.index:
        # Pula pontos de corte ordinais (cutoffs/thresholds)
        if "/" in str(variavel) or str(variavel).startswith("cut"):
            continue

        coef = parametros[variavel]
        ep = erros.get(variavel, np.nan)
        z = coef / ep if ep and not np.isnan(ep) and ep > 0 else np.nan
        p_val = p_valores.get(variavel, np.nan)

        ic_inf = intervalos.loc[variavel, 0] if variavel in intervalos.index else np.nan
        ic_sup = intervalos.loc[variavel, 1] if variavel in intervalos.index else np.nan

        odds_ratio = np.exp(coef)
        or_inf = np.exp(ic_inf) if not np.isnan(ic_inf) else np.nan
        or_sup = np.exp(ic_sup) if not np.isnan(ic_sup) else np.nan

        linhas.append(
            {
                "modelo": nome_modelo,
                "variavel": variavel,
                "coeficiente": coef,
                "erro_padrao": ep,
                "z": z,
                "p_valor": p_val,
                "sig": formatar_significancia(p_val),
                "odds_ratio": odds_ratio,
                "ic_95_inf": or_inf,
                "ic_95_sup": or_sup,
            }
        )

    tabela = pd.DataFrame(linhas)

    return tabela


def calcular_metricas(resultado, nome_modelo, n_obs):
    """
    Calcula métricas de qualidade do ajuste (Log-Likelihood, AIC, BIC).
    """
    llf = resultado.llf
    aic = resultado.aic if hasattr(resultado, "aic") else np.nan
    bic = resultado.bic if hasattr(resultado, "bic") else np.nan

    return {
        "modelo": nome_modelo,
        "n_obs": n_obs,
        "log_likelihood": llf,
        "aic": aic,
        "bic": bic,
    }


# =========================================================
# MODELO 1A: Associação Bruta (Precipitação contínua)
# =========================================================

print()
print("=" * 60)
print("MODELO 1A: ASSOCIAÇÃO BRUTA (PRECIPITAÇÃO CONTÍNUA)")
print("=" * 60)

modelo_m1a = OrderedModel(
    y_ordinal,
    matriz_x_m1a,
    distr="logit"
)

resultado_m1a = modelo_m1a.fit(
    method="bfgs",
    disp=False,
    maxiter=1000
)

print(
    resultado_m1a.summary()
)

tabela_m1a = extrair_tabela_modelo(
    resultado_m1a,
    "Modelo 1A (Bruto - Precipitação mm)"
)

metricas_m1a = calcular_metricas(
    resultado_m1a,
    "Modelo 1A (Bruto - mm)",
    len(dados_modelo)
)

print(
    "\nRazões de Chances (Odds Ratios) - Modelo 1A:"
)

print(
    tabela_m1a[
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


# =========================================================
# MODELO 1B: Associação Bruta (Chuva binária)
# =========================================================

print()
print("=" * 60)
print("MODELO 1B: ASSOCIAÇÃO BRUTA (CHUVA BINÁRIA)")
print("=" * 60)

modelo_m1b = OrderedModel(
    y_ordinal,
    matriz_x_m1b,
    distr="logit"
)

resultado_m1b = modelo_m1b.fit(
    method="bfgs",
    disp=False,
    maxiter=1000
)

print(
    resultado_m1b.summary()
)

tabela_m1b = extrair_tabela_modelo(
    resultado_m1b,
    "Modelo 1B (Bruto - Chuva Binária)"
)

metricas_m1b = calcular_metricas(
    resultado_m1b,
    "Modelo 1B (Bruto - Binária)",
    len(dados_modelo)
)

print(
    "\nRazões de Chances (Odds Ratios) - Modelo 1B:"
)

print(
    tabela_m1b[
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


# =========================================================
# MODELO 2A: Associação Ajustada (Precipitação contínua)
# =========================================================

print()
print("=" * 60)
print("MODELO 2A: ASSOCIAÇÃO AJUSTADA (PRECIPITAÇÃO CONTÍNUA)")
print("=" * 60)

modelo_m2a = OrderedModel(
    y_ordinal,
    matriz_x_m2a,
    distr="logit"
)

resultado_m2a = modelo_m2a.fit(
    method="bfgs",
    disp=False,
    maxiter=1000
)

print(
    resultado_m2a.summary()
)

tabela_m2a = extrair_tabela_modelo(
    resultado_m2a,
    "Modelo 2A (Ajustado - Precipitação mm)"
)

metricas_m2a = calcular_metricas(
    resultado_m2a,
    "Modelo 2A (Ajustado - mm)",
    len(dados_modelo)
)

print(
    "\nRazões de Chances (Odds Ratios) - Modelo 2A:"
)

print(
    tabela_m2a[
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


# =========================================================
# MODELO 2B: Associação Ajustada (Chuva binária)
# =========================================================

print()
print("=" * 60)
print("MODELO 2B: ASSOCIAÇÃO AJUSTADA (CHUVA BINÁRIA)")
print("=" * 60)

modelo_m2b = OrderedModel(
    y_ordinal,
    matriz_x_m2b,
    distr="logit"
)

resultado_m2b = modelo_m2b.fit(
    method="bfgs",
    disp=False,
    maxiter=1000
)

print(
    resultado_m2b.summary()
)

tabela_m2b = extrair_tabela_modelo(
    resultado_m2b,
    "Modelo 2B (Ajustado - Chuva Binária)"
)

metricas_m2b = calcular_metricas(
    resultado_m2b,
    "Modelo 2B (Ajustado - Binária)",
    len(dados_modelo)
)

print(
    "\nRazões de Chances (Odds Ratios) - Modelo 2B:"
)

print(
    tabela_m2b[
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


# =========================================================
# Avaliação da Suposição de Chances Proporcionais
#
# Para testar a premissa de linhas paralelas (proportional odds),
# ajustamos modelos de regressão logística binária separados nos
# cortes cumulativos da escala ordinal:
#
# Corte 1: Y >= 1 vs Y = 0 (Com Vítimas vs Sem Vítimas)
# Corte 2: Y >= 2 vs Y <= 1 (Vítimas Fatais vs Vítimas Feridas ou Sem Vítimas)
#
# Se os coeficientes forem semelhantes entre os cortes,
# a suposição de chances proporcionais é razoável.
# =========================================================

print()
print("=" * 60)
print("AVALIAÇÃO DA SUPOSIÇÃO DE CHANCES PROPORCIONAIS")
print("=" * 60)

corte_1 = (
    dados_modelo["gravidade_ordinal"].astype(int) >= 1
).astype(int)

corte_2 = (
    dados_modelo["gravidade_ordinal"].astype(int) >= 2
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

variaveis_comparar = [
    v for v in logit_corte1.params.index
    if v != "const"
]

for var in variaveis_comparar:
    b_c1 = logit_corte1.params.get(var, np.nan)
    or_c1 = np.exp(b_c1)
    p_c1 = logit_corte1.pvalues.get(var, np.nan)

    b_c2 = logit_corte2.params.get(var, np.nan)
    or_c2 = np.exp(b_c2)
    p_c2 = logit_corte2.pvalues.get(var, np.nan)

    b_ord = resultado_m2a.params.get(var, np.nan)
    or_ord = np.exp(b_ord) if not np.isnan(b_ord) else np.nan

    comparacao_cortes.append(
        {
            "variavel": var,
            "beta_corte1 (Y>=1)": round(b_c1, 4),
            "or_corte1": round(or_c1, 3),
            "p_corte1": round(p_c1, 4),
            "beta_corte2 (Y>=2)": round(b_c2, 4),
            "or_corte2": round(or_c2, 3),
            "p_corte2": round(p_c2, 4),
            "beta_ordinal": round(b_ord, 4),
            "or_ordinal": round(or_ord, 3),
        }
    )

df_proporcional = pd.DataFrame(comparacao_cortes)

print(
    "\nComparação dos coeficientes nos cortes cumulativos (Y >= 1 vs Y >= 2):"
)

print(
    df_proporcional.to_string(
        index=False
    )
)


# =========================================================
# Comparação geral dos modelos e métricas de ajuste
# =========================================================

print()
print("=" * 60)
print("COMPARAÇÃO DE AJUSTE DOS MODELOS")
print("=" * 60)

tabela_metricas = pd.DataFrame(
    [
        metricas_m1a,
        metricas_m1b,
        metricas_m2a,
        metricas_m2b,
    ]
)

print(
    tabela_metricas.to_string(
        index=False
    )
)


# =========================================================
# Consolidação e salvamento dos resultados
# =========================================================

todas_tabelas = pd.concat(
    [
        tabela_m1a,
        tabela_m1b,
        tabela_m2a,
        tabela_m2b,
    ],
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

print()
print("=" * 60)
print("FINALIZAÇÃO")
print("=" * 60)

print(
    f"\nBase limpa para modelagem salva em:\n{SAIDA_BASE_MODELAGEM}"
)

print(
    f"\nTabela consolidada de Odds Ratios salva em:\n{SAIDA_RESULTADOS}"
)

print(
    "\nRegressão Logística Ordinal concluída com sucesso."
)