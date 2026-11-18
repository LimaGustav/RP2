"""Executa, separadamente, as sensibilidades e a comparação PRF–INMET."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from analises.concordancia import calcular_concordancia
from analises.pareamento import comparar_estacoes, comparar_horarios, indexar_precipitacao
from analises.qualidade_integracao import classificar_chuva_prf
from config import ARQUIVO_BASE_INTEGRADA, ARQUIVO_INMET, PASTA_DADOS_PROCESSADOS
from modelos.ordinal import ajustar_modelo_ordinal
from modelos.preparacao import criar_matriz_ajustada
from utils.validacao import preparar_tipos


PASTA_SAIDA = PASTA_DADOS_PROCESSADOS / "analises_adicionais"
COLUNAS_EFEITO = ["odds_ratio", "ic_95_inf", "ic_95_sup", "p_valor"]


def efeito(dados, variavel, ajustar=True):
    matriz = (
        criar_matriz_ajustada(dados, variavel)
        if ajustar else dados[[variavel]].astype(float)
    )
    if variavel not in matriz or matriz[variavel].nunique() < 2:
        raise ValueError(f"A exposição {variavel} precisa variar na amostra.")
    resultado, tabela = ajustar_modelo_ordinal(dados["gravidade_ordinal"], matriz, variavel)
    linha = tabela.loc[tabela["variavel"].eq(variavel), COLUNAS_EFEITO].iloc[0]
    if not resultado.mle_retvals["converged"] or not np.isfinite(linha).all():
        raise ValueError(f"O ajuste ordinal de {variavel} não produziu estimativas válidas.")
    return linha.to_dict()


def efeitos_inmet(dados):
    return {
        f"{prefixo}_{metrica}": valor
        for prefixo, termo in [("mm", "precipitacao_mm"), ("chuva", "chuva_binaria")]
        for metrica, valor in efeito(dados, termo).items()
    }


def tabela_temporal(amostras):
    chuva_original = amostras["A"]["chuva_binaria"]
    linhas = []
    for regra, dados in amostras.items():
        chuva = dados["chuva_binaria"]
        ganhou = int((chuva_original.eq(0) & chuva.eq(1)).sum())
        perdeu = int((chuva_original.eq(1) & chuva.eq(0)).sum())
        linhas.append({
            "regra": regra,
            "n": len(dados),
            "com_chuva": int(chuva.sum()),
            "sem_chuva": int(chuva.eq(0).sum()),
            "reclassificados_0_para_1": ganhou,
            "reclassificados_1_para_0": perdeu,
            "percentual_reclassificado": 100 * (ganhou + perdeu) / len(dados),
            **efeitos_inmet(dados),
        })
    return pd.DataFrame(linhas)


def tabela_estacoes(integrada, cenarios):
    principal, _ = cenarios["principal"]
    linhas = []
    for cenario, (dados, sem_medicao) in cenarios.items():
        if not principal.index.difference(dados.index).empty:
            raise ValueError("A busca alternativa deixou de fora um caso da amostra principal.")
        comuns = dados.loc[principal.index]
        novas = dados.loc[dados.index.difference(principal.index)]
        originais = integrada.loc[novas.index]
        aumento = novas["distancia_estacao_km"] - originais["distancia_estacao_km"]
        tempo = novas["diferenca_tempo_min"]
        linhas.append({
            "cenario": cenario,
            "n": len(dados),
            "com_chuva": int(dados["chuva_binaria"].sum()),
            "sem_medicao_no_raio": sem_medicao,
            "recuperados": len(novas),
            "recuperados_com_chuva": int(novas["chuva_binaria"].sum()),
            "mudaram_de_estacao": int(novas["codigo_wmo"].ne(originais["codigo_wmo"]).sum()),
            "reclassificacoes_casos_comuns": int(
                comuns["chuva_binaria"].ne(principal["chuva_binaria"]).sum()
            ),
            "alteracoes_mm_casos_comuns": int(
                comuns["precipitacao_mm"].ne(principal["precipitacao_mm"]).sum()
            ),
            "distancia_mediana_km": dados["distancia_estacao_km"].median(),
            "distancia_original_recuperados_mediana_km": originais["distancia_estacao_km"].median(),
            "distancia_nova_recuperados_mediana_km": novas["distancia_estacao_km"].median(),
            "aumento_distancia_mediana_km": aumento.median(),
            "aumento_distancia_p95_km": aumento.quantile(0.95),
            "aumento_distancia_max_km": aumento.max(),
            "diferenca_tempo_mediana_min": tempo.median(),
            "diferenca_tempo_p95_min": tempo.quantile(0.95),
            "diferenca_tempo_max_min": tempo.max(),
            **efeitos_inmet(dados),
        })
    return pd.DataFrame(linhas)


def modelos_por_fonte(dados, indicador_prf):
    pares = dados.assign(chuva_prf=indicador_prf).dropna(subset=["chuva_prf"])
    return pd.DataFrame([
        {
            "fonte": fonte,
            "modelo": "ajustado" if ajustar else "bruto",
            "n": len(pares),
            **efeito(pares, variavel, ajustar),
        }
        for ajustar in [False, True]
        for fonte, variavel in [("INMET", "chuva_binaria"), ("PRF", "chuva_prf")]
    ])


def main():
    for caminho in [ARQUIVO_BASE_INTEGRADA, ARQUIVO_INMET]:
        if not caminho.is_file():
            raise FileNotFoundError(
                f"Arquivo ausente: {caminho}. Execute primeiro python executar_pipeline.py."
            )
    integrada, inmet = preparar_tipos(
        pd.read_csv(ARQUIVO_BASE_INTEGRADA, parse_dates=["data_hora_medicao"]),
        pd.read_csv(ARQUIVO_INMET),
    )
    medidas = indexar_precipitacao(inmet)
    chaves = pd.MultiIndex.from_frame(integrada[["data_hora_medicao", "codigo_wmo"]])
    precipitacao_original = medidas.reindex(chaves).to_numpy()
    if not np.array_equal(precipitacao_original, integrada["precipitacao_mm"], equal_nan=True):
        raise ValueError("Os CSVs integrado e INMET não correspondem. Regere o pipeline principal.")

    cenarios = comparar_estacoes(integrada, inmet, medidas)
    principal, _ = cenarios["principal"]
    horarios = comparar_horarios(integrada, medidas, principal)
    indicador_prf = classificar_chuva_prf(principal)
    tabelas = {
        "sensibilidade_temporal.csv": tabela_temporal(horarios),
        "sensibilidade_estacao.csv": tabela_estacoes(integrada, cenarios),
        "concordancia_prf_inmet.csv": calcular_concordancia(principal, indicador_prf),
        "modelos_prf_inmet.csv": modelos_por_fonte(principal, indicador_prf),
    }

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=PASTA_SAIDA) as temporaria:
        pasta = Path(temporaria)
        for nome, tabela in tabelas.items():
            tabela.to_csv(pasta / nome, index=False, encoding="utf-8-sig")
        for nome in tabelas:
            destino = PASTA_SAIDA / nome
            (pasta / nome).replace(destino)
            print(f"Arquivo criado: {destino}")


if __name__ == "__main__":
    main()
