"""Comparação simétrica da presença de chuva registrada pela PRF e pelo INMET."""

import numpy as np
import pandas as pd
from scipy.stats import chi2, pearsonr


LIMITES_CONCORDANCIA_KM = (10, 20, 30)
SEMENTE_BOOTSTRAP = 20261118
N_REAMOSTRAGENS = 5000


def coeficiente_phi(tabelas):
    """Recebe contagens na ordem 00, 01, 10 e 11."""
    a, b, c, d = np.moveaxis(tabelas.astype(float), -1, 0)
    denominador = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return np.divide(
        a * d - b * c, denominador, out=np.full_like(denominador, np.nan),
        where=denominador > 0,
    )


def calcular_concordancia(dados, indicador_prf):
    resultados = []
    rng = np.random.default_rng(SEMENTE_BOOTSTRAP)
    validos = dados["precipitacao_mm"].notna() & indicador_prf.notna()
    for limite in LIMITES_CONCORDANCIA_KM:
        mascara = validos & dados["distancia_estacao_km"].le(limite)
        prf = indicador_prf.loc[mascara].astype(int)
        inmet = dados.loc[mascara, "precipitacao_mm"].gt(0).astype(int)
        classes = 2 * prf + inmet
        contagens = classes.value_counts().reindex(range(4), fill_value=0).to_numpy()
        phi = float(coeficiente_phi(contagens))
        if not np.isfinite(phi):
            raise ValueError(f"Os dois indicadores precisam variar no recorte de {limite} km.")
        r = pearsonr(prf, inmet).statistic
        if not np.isclose(r, phi, atol=1e-12, rtol=0):
            raise ValueError("O coeficiente phi não coincide com a correlação dos indicadores.")

        por_estacao = pd.crosstab(dados.loc[mascara, "codigo_wmo"], classes)
        por_estacao = por_estacao.reindex(columns=range(4), fill_value=0).to_numpy()
        n_estacoes = len(por_estacao)
        if n_estacoes < 2:
            raise ValueError(f"O bootstrap por estação requer pelo menos duas em {limite} km.")
        sorteios = rng.integers(n_estacoes, size=(N_REAMOSTRAGENS, n_estacoes))
        reamostras = coeficiente_phi(por_estacao[sorteios].sum(axis=1))
        ic_inf, ic_sup = np.nanquantile(reamostras, [0.025, 0.975])

        a, b, c, d = (int(valor) for valor in contagens)
        n = len(prf)
        concordancia = (a + d) / n
        esperada = ((a + b) * (a + c) + (c + d) * (b + d)) / n**2
        resultados.append({
            "limite_km": limite,
            "n": n,
            "n_estacoes": n_estacoes,
            "prf0_inmet0": a,
            "prf0_inmet1": b,
            "prf1_inmet0": c,
            "prf1_inmet1": d,
            "pearson_phi": float(r),
            "p_nominal": chi2.sf(n * r**2, 1),
            "phi_ic_95_inf": ic_inf,
            "phi_ic_95_sup": ic_sup,
            "kappa": (concordancia - esperada) / (1 - esperada),
            "concordancia_total": concordancia,
            "discordancia": (b + c) / n,
            "concordancia_chuva": 2 * d / (2 * d + b + c),
            "concordancia_ausencia_chuva": 2 * a / (2 * a + b + c),
        })
    return pd.DataFrame(resultados)
