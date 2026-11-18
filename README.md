# Associação entre Precipitação e Gravidade dos Acidentes em Rodovias Federais no Estado de São Paulo

Este projeto investiga a associação entre a precipitação registrada por estações meteorológicas e a gravidade dos acidentes ocorridos em rodovias federais no estado de São Paulo.

A análise utiliza dados da Polícia Rodoviária Federal (PRF) e do Instituto Nacional de Meteorologia (INMET), integrados por proximidade espacial e temporal.

## Estrutura dos dados

### PRF

- `datatran2025.csv` e `datatran2026.csv` são as bases agrupadas por ocorrência. Cada linha representa um acidente.

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
│   │   │   ├── datatran2025.csv
│   │   │   └── datatran2026.csv
│   │   └── inmet/
│   │       ├── 2025/
│   │       └── 2026/
│   └── processed/
├── src/
│   ├── 01_preparar_prf.py
│   ├── 02_preparar_inmet.py
│   ├── 03_integrar_bases.py
│   ├── 04_analise_descritiva.py
│   ├── 05_regressao_ordinal.py
│   ├── config.py
│   ├── analises/
│   ├── modelos/
│   └── utils/
├── executar_pipeline.py
├── RP2_Analise_PRF_INMET.ipynb
├── README.md
├── requirements-notebook.txt
└── requirements.txt
```

## Execução local no Windows

Os CSVs da PRF já estão no repositório. Extraia os CSVs brutos do INMET em
`datasets/raw/inmet/2025/` e `datasets/raw/inmet/2026/`, mantendo os nomes
originais dos arquivos.
A validação usou 40 arquivos de estações de São Paulo por ano.
Os arquivos de 2025 cobrem até 31/12/2025, e os de 2026, até 31/08/2026.

Na raiz do repositório, use o PowerShell. O pipeline foi validado com Python 3.13.3.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Para executar o pipeline completo na raiz do repositório, use o comando abaixo.
O executor informa a etapa em andamento e para na primeira falha.

```powershell
python executar_pipeline.py
```

Para executar ou depurar uma etapa por vez, use os comandos abaixo na ordem indicada.

```powershell
python src\01_preparar_prf.py
python src\02_preparar_inmet.py
python src\03_integrar_bases.py
python src\04_analise_descritiva.py
python src\05_regressao_ordinal.py
```

Também é possível executar os comandos com `.\.venv\Scripts\python.exe` sem ativar o ambiente.
As etapas de preparação e integração geram os CSVs intermediários em
`datasets/processed/`, incluindo `base_integrada.csv`.
A etapa 04 exibe a análise descritiva no console e a etapa 05 grava os CSVs de modelagem
na mesma pasta.

## Notebook de análise

O [notebook de análise](RP2_Analise_PRF_INMET.ipynb) apresenta a cobertura dos dados,
a seleção da amostra, as visualizações e os resultados estatísticos. Ele usa os CSVs
em `datasets/processed/`. Execute o pipeline primeiro se precisar gerar esses arquivos.

Com a `.venv` ativada, instale as dependências do notebook e execute-o a partir da raiz.

```powershell
python -m pip install -r requirements-notebook.txt
python -m nbconvert --to notebook --execute --inplace RP2_Analise_PRF_INMET.ipynb
```

Para navegar pelas seções, abra o arquivo em um editor compatível com Jupyter e selecione a
`.venv` como ambiente. Matplotlib, o kernel Jupyter e o executor do notebook são instalados
separadamente das dependências do pipeline.
