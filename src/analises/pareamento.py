"""Sensibilidades que partem dos pareamentos oficiais já processados."""

import numpy as np
import pandas as pd

from config import LIMITE_DISTANCIA_KM, LIMITE_TEMPO_MIN
from modelos.preparacao import preparar_dados_modelagem
from utils.integracao import encontrar_estacao_mais_proxima, preparar_estacoes


def indexar_precipitacao(inmet):
    """Indexa os acumulados por horário e estação, inclusive os valores de 0 mm."""
    medidas = inmet.set_index(
        ["data_hora", "codigo_wmo"], verify_integrity=True
    )["precipitacao_mm"].sort_index()
    return medidas.where(np.isfinite(medidas) & medidas.ge(0))


def preparar_modelagem(dados):
    return preparar_dados_modelagem(dados, LIMITE_DISTANCIA_KM, LIMITE_TEMPO_MIN)


def comparar_horarios(dados, medidas, principal):
    """Compara três horários nos mesmos acidentes, mantendo a estação original."""
    amostras = {"A": principal}
    horarios = {
        # Nas horas cheias, o próprio registro encerra o intervalo selecionado.
        "B": dados["data_hora"].dt.ceil("h"),
        "C": dados["data_hora"].dt.floor("h"),
    }
    for regra, horario in horarios.items():
        chaves = pd.MultiIndex.from_arrays([horario, dados["codigo_wmo"]])
        precipitacao = medidas.reindex(chaves).to_numpy()
        nova_base = dados.assign(
            data_hora_medicao=horario,
            diferenca_tempo_min=(horario - dados["data_hora"]).abs().dt.total_seconds() / 60,
            precipitacao_mm=precipitacao,
            status_integracao=np.where(pd.notna(precipitacao), "integrado", "precipitacao_ausente"),
        )
        amostras[regra] = preparar_modelagem(nova_base)

    comum = principal.index.intersection(amostras["B"].index).intersection(amostras["C"].index)
    return {regra: amostra.loc[comum] for regra, amostra in amostras.items()}


def recuperar_na_mesma_hora(dados, inmet, medidas):
    """Busca a estação válida mais próxima apenas para casos sem precipitação."""
    estacoes = preparar_estacoes(inmet)
    medida_valida = medidas.dropna()
    horarios_validos = medida_valida.index.get_level_values("data_hora").unique()
    candidatos = dados.loc[
        dados["precipitacao_mm"].isna()
        & dados["distancia_estacao_km"].le(LIMITE_DISTANCIA_KM)
        & dados["diferenca_tempo_min"].le(LIMITE_TEMPO_MIN)
    ]
    recuperada = dados.copy()
    for horario, acidentes in candidatos.groupby("data_hora_medicao", sort=False):
        if horario not in horarios_validos:
            continue
        disponiveis = medida_valida.loc[horario]
        estacoes_validas = estacoes[estacoes["codigo_wmo"].isin(disponiveis.index)]
        for indice, acidente in acidentes.iterrows():
            estacao, distancia = encontrar_estacao_mais_proxima(acidente, estacoes_validas)
            if estacao is not None and distancia <= LIMITE_DISTANCIA_KM:
                codigo = estacao["codigo_wmo"]
                recuperada.at[indice, "codigo_wmo"] = codigo
                recuperada.at[indice, "estacao_inmet"] = estacao["estacao"]
                recuperada.at[indice, "distancia_estacao_km"] = distancia
                recuperada.at[indice, "precipitacao_mm"] = disponiveis.at[codigo]
                recuperada.at[indice, "status_integracao"] = "integrado"
    return recuperada


def comparar_estacoes(dados, inmet, medidas):
    alternativa = recuperar_na_mesma_hora(dados, inmet, medidas)
    # O denominador de disponibilidade inclui todos os acidentes dentro do raio.
    amostras = {}
    for nome, base in [("principal", dados), ("alternativa", alternativa)]:
        no_raio = base["distancia_estacao_km"].le(LIMITE_DISTANCIA_KM)
        sem_medicao = int((no_raio & base["precipitacao_mm"].isna()).sum())
        amostras[nome] = (preparar_modelagem(base), sem_medicao)
    return amostras
