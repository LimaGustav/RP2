# Associação entre Precipitação e Gravidade dos Acidentes em Rodovias Federais no Estado de São Paulo

Este projeto investiga a associação entre a precipitação registrada por estações meteorológicas e a gravidade dos acidentes ocorridos em rodovias federais no estado de São Paulo.

A análise utiliza dados da Polícia Rodoviária Federal (PRF) e do Instituto Nacional de Meteorologia (INMET), integrados por proximidade espacial e temporal.

## Estrutura dos dados

### PRF

- `datatran2026.csv`  
  Base agrupada por ocorrência. Cada linha representa um acidente. Esta é a principal base utilizada no estudo.

- `acidentes2026.csv`  
  Base agrupada por pessoa envolvida no acidente. É mantida como fonte complementar, mas não é utilizada como base principal da análise.

### INMET

Os arquivos do INMET correspondem às estações meteorológicas automáticas e contêm registros horários de diversas variáveis meteorológicas.

Neste projeto serão utilizadas principalmente:

- identificação da estação;
- latitude e longitude da estação;
- data;
- hora;
- precipitação total horária em milímetros.

## Metodologia

O projeto é dividido nas seguintes etapas:

1. Preparação e filtragem dos acidentes da PRF para o estado de São Paulo;
2. Preparação e unificação dos arquivos das estações meteorológicas do INMET;
3. Integração espacial e temporal entre acidentes e estações meteorológicas;
4. Análise descritiva dos dados;
5. Avaliação da associação entre precipitação e gravidade por meio de Regressão Logística Ordinal;
6. Interpretação dos resultados.

A gravidade dos acidentes será representada pela classificação da PRF:

- sem vítimas;
- com vítimas feridas;
- com vítimas fatais.

## Estrutura do projeto

```text
RP2/
├── articles/
├── datasets/
│   ├── raw/
│   │   ├── prf/
│   │   └── inmet/
│   └── processed/
├── src/
│   ├── 01_preparar_prf.py
│   ├── 02_preparar_inmet.py
│   ├── 03_integrar_bases.py
│   ├── 04_analise_descritiva.py
│   └── 05_regressao_ordinal.py
├── README.md
└── requirements.txt