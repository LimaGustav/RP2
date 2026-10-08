Auditoria estática concluída: não executei scripts do projeto, não alterei arquivos e não criei commits.

## 1. Resumo executivo

O pipeline está bem organizado e os outputs preservados no notebook são, em geral, coerentes com o código. A preparação da PRF parece ter produzido 6.515 acidentes de SP, sem datas, horários, coordenadas, gravidade ordinal ou IDs inválidos nos resultados registrados.

Entretanto, há riscos metodológicos importantes que impedem tratar o modelo ordinal atual como resultado final:

- A estação é escolhida apenas pela proximidade espacial, antes de verificar se ela possui dado de precipitação válido no horário. Isso pode descartar um acidente mesmo havendo outra estação próxima com observação válida.
- O script do INMET converte texto em número, mas não valida sentinelas, valores negativos, duplicidades estação–tempo, linhas corrompidas ou extremos fisicamente implausíveis.
- A integração temporal usa a medição com menor diferença absoluta; ela pode escolher uma observação posterior ao acidente. Para chuva horária, isso pode representar informação ainda não disponível no instante do acidente.
- A hipótese de chances proporcionais não foi testada formalmente. O diagnóstico exibido sugere violação relevante para covariáveis como `fase_dia`; portanto, o modelo ordinal ajustado não deve ser considerado automaticamente válido como modelo principal.
- A amostra final cai de 6.515 para 3.730 acidentes com limite de 30 km: perde-se cerca de 43% da base inicial. Isso torna a análise de seleção/missingness essencial.
- O resultado atual não fornece evidência robusta de associação entre chuva e maior gravidade. A associação muda conforme o limite espacial; isso é evidência de sensibilidade à regra de integração, não evidência conclusiva.

Não foi possível verificar com os arquivos disponíveis a qualidade linha a linha dos CSVs brutos do INMET, suas sentinelas, seus valores extremos, duplicidades ou falhas de cobertura, pois os arquivos INMET e as bases processadas não estão presentes no repositório local.

## 2. Mapa do pipeline atual

| Etapa | Arquivo/função | Entrada → saída | Observação |
|---|---|---|---|
| PRF raw | `src/01_preparar_prf.py` | `datatran2025.csv`, `datatran2026.csv` → `acidentes_sp.csv` | Lê, concatena, filtra `uf == SP`, converte data/hora/coordenadas e cria gravidade. |
| Definição do desfecho | `01_preparar_prf.py`, `mapa_gravidade` | classificação PRF → `gravidade_ordinal` | 0 sem vítimas; 1 feridos; 2 fatais. |
| INMET raw | `src/02_preparar_inmet.py`, `processar_arquivo` | CSVs INMET → `inmet_sp.csv` | Lê 8 linhas de metadados, converte UTC para São Paulo e precipitação para número. |
| Estações | `02_preparar_inmet.py` | metadados → latitude/longitude/WMO | Filtra arquivos/estações de SP. |
| Integração espacial | `utils/integracao.py`, `encontrar_estacao_mais_proxima` | acidente + estações → estação mais próxima | Haversine; não usa limite nesta etapa. |
| Integração temporal | `encontrar_medicao_mais_proxima` | acidente + série da estação → observação mais próxima | Menor diferença absoluta; aceita antes ou depois do acidente. |
| Base integrada | `src/03_integrar_bases.py` | PRF preparada + INMET preparado → `base_integrada.csv` | Mantém distância, diferença temporal, precipitação e status. |
| Descritiva | `src/04_analise_descritiva.py` | `base_integrada.csv` | Resume chuva, distância, extremos e comparação PRF–INMET. |
| Modelagem | `src/05_regressao_ordinal.py` | `base_integrada.csv` → resultados | Filtra 30 km/60 min, ajusta `OrderedModel`, multinomial e sensibilidades 30/40/50 km. |

Trajeto de uma linha: acidente PRF → filtro SP → data/hora e coordenadas → estação INMET geograficamente mais próxima → observação dessa estação temporalmente mais próxima → precipitação/status → filtros de modelagem → regressão.

## 3. Problemas críticos encontrados

### [CRÍTICO] Seleção espacial sem considerar disponibilidade temporal/meteorológica

**Local:** `src/03_integrar_bases.py`; `utils/integracao.py`.

**Problema:** o código escolhe primeiro a estação geograficamente mais próxima e só depois procura a medição naquela estação. Não procura outra estação próxima quando a selecionada tem precipitação ausente.

**Por que importa:** parte dos 854 acidentes sem precipitação pode ter uma alternativa meteorológica válida em estação ligeiramente mais distante. A ausência passa a depender da estação escolhida, não necessariamente da inexistência de observação na região.

**Evidência:** `encontrar_estacao_mais_proxima()` é chamado antes de `encontrar_medicao_mais_proxima()`; o código marca `precipitacao_ausente` se a medição da estação mais próxima for `NaN`.

**Recomendação:** avaliar conjuntamente espaço, tempo e validade da precipitação. Por exemplo, selecionar entre estações candidatas dentro de uma regra espacial/temporal pré-especificada.

**Impacto esperado:** menor perda evitável de dados e menor viés por cobertura desigual das estações.

### [CRÍTICO] Hipótese de chances proporcionais não está adequadamente demonstrada

**Local:** `src/05_regressao_ordinal.py`, seção de diagnóstico.

**Problema:** o código compara dois logits binários, mas declara explicitamente que isso “não substitui um teste formal”. Ainda assim, a regressão ordinal é usada como análise principal.

**Evidência:** no notebook, `fase_dia_Plena Noite` teve OR de 0,58 para “gravidade ≥ 1” e 2,60 para “gravidade ≥ 2”; há diferença substancial entre os cortes. Para `tracado_via_Curva`, os ORs também divergem.

**Por que importa:** o modelo ordinal presume que cada covariável tem o mesmo efeito nos dois limiares de gravidade. Se isso falha para covariáveis do modelo, os coeficientes ordinais ajustados podem não resumir adequadamente os dados.

**Recomendação:** aplicar teste formal de proporcionalidade ou usar modelo de odds proporcionais parciais/multinomial como análise principal se a hipótese falhar.

**Impacto esperado:** interpretação estatística compatível com a estrutura observada do desfecho.

### [CRÍTICO] Limpeza insuficiente da precipitação do INMET

**Local:** `src/02_preparar_inmet.py`.

**Problema:** valores são transformados com `pd.to_numeric(..., errors="coerce")`, mas não há verificação de sentinelas, negativos, valores impossíveis, duplicidade estação–data–hora, arquivos incompletos, linhas vazias ou metadados anômalos.

**Por que importa:** um sentinela numérico, como códigos negativos de falha, permaneceria como precipitação “válida” e entraria no modelo. Isso é diferente de `NaN` e diferente de 0 mm.

**Evidência:** não existem filtros posteriores para `precipitacao_mm < 0`, limites físicos, duplicidades ou cobertura por estação.

**Recomendação:** criar uma auditoria explícita da variável antes da integração, preservando estados distintos: 0 mm, ausente, inválido e não observado.

**Impacto esperado:** evita classificar falha instrumental como ausência de chuva ou como chuva negativa.

## 4. Problemas importantes

### [ALTO] A medição temporal pode ser posterior ao acidente

**Local:** `utils/integracao.py`, `encontrar_medicao_mais_proxima`.

A diferença temporal é calculada com valor absoluto. Isso pode selecionar uma medição futura. O notebook mostra máximo de 30 minutos e mediana de 15 minutos, coerente com a escolha da observação horária mais próxima, mas não resolve a ambiguidade causal/temporal.

A precipitação horária deve ser interpretada segundo a convenção oficial do timestamp do INMET: início, fim ou total do intervalo. Sem isso, não é possível afirmar que a chuva medida antecede o acidente.

### [ALTO] Efeito de precipitação é modelado linearmente em mm

**Local:** `src/05_regressao_ordinal.py`.

Com muitos zeros e poucos valores positivos, um OR por 1 mm mistura “começar a chover” com “aumentar intensidade entre episódios chuvosos”. O resultado por mm pode ser dominado pela massa em zero e por poucos extremos.

A análise binária ajuda, mas ainda é insuficiente para caracterizar não linearidade. A relação deve ser explorada como: sem chuva vs chuva; intensidade entre eventos chuvosos; categorias físicas; spline/GAM.

### [ALTO] Possível viés de seleção por casos completos

**Evidência observada:** 854/6.515 acidentes ficaram sem precipitação válida: 13,1%. Em 2025 foram 701/4.683; em 2026, 153/1.832.

O código ajusta um modelo auxiliar para a ausência, o que é positivo, mas não prova MCAR, MAR ou MNAR. A ausência pode depender de estação, região, estação do ano, distância, manutenção do equipamento e condição meteorológica.

### [MÉDIO] Supressão global de avisos

**Local:** `src/05_regressao_ordinal.py`, `warnings.filterwarnings("ignore")`.

Alertas de convergência, separação, matriz mal condicionada ou Hessiana singular podem ser ocultados. Isso dificulta auditar estabilidade do modelo.

### [MÉDIO] Sem diagnóstico explícito de colinearidade, influência ou agrupamento

Não há VIF/diagnóstico equivalente, resíduos/influência, erros robustos, agrupamento por rodovia/estação ou ajuste para dependência espacial. Acidentes associados à mesma estação ou rodovia não são necessariamente independentes.

## 5. Limpeza dos dados

### PRF

Os outputs preservados registram:

- 6.515 acidentes de SP;
- 4.683 em 2025 e 1.832 em 2026;
- 0 datas, horários, coordenadas ou gravidades inválidas;
- 0 IDs duplicados;
- categorias de gravidade: 971 sem vítimas, 5.273 com feridos e 271 fatais.

Ponto pendente: o script apenas imprime IDs duplicados; não verifica explicitamente linhas duplicadas completas nem exclui registros inválidos se fossem encontrados. A consistência entre `classificacao_acidente`, `mortos`, `feridos_graves` e `feridos_leves` também não é auditada.

### INMET

O notebook registra:

- 40 estações;
- 583.680 observações;
- 350.400 em 2025 e 233.160 em 2026;
- 85.496 precipitações ausentes em 2025 e 57.084 em 2026;
- 0 timestamps/coordernadas inválidos após a conversão;
- período local de `2024-12-31 21:00` a `2026-08-31 20:00`.

A presença de 120 observações classificadas localmente em 2024 decorre da conversão de `2025-01-01 00:00 UTC` para horário de São Paulo. Isso é esperado, mas a definição do período analítico deve ser feita por instante, não apenas por ano local.

Não foi possível verificar com os arquivos disponíveis se os `NaN` representam campos vazios, sentinelas, falhas de leitura ou ausência efetiva de medição.

## 6. Integração PRF–INMET e recomendação de distância

O notebook confirma:

| Estatística | Distância |
|---|---:|
| Média | 26,23 km |
| Q1 | 12,09 km |
| Mediana | 21,65 km |
| Q3 | 39,16 km |
| Máximo | 81,84 km |

Não é metodologicamente defensável supor que chuva em estação a 80 km represente necessariamente a chuva no local do acidente. A precipitação é altamente heterogênea no espaço, sobretudo em chuva convectiva.

| Limite | Leitura metodológica |
|---|---|
| 5 km | Alta representatividade; provável perda de amostra muito grande. |
| 10 km | Conservador e defensável para validação de alta qualidade; pode concentrar a amostra em áreas com maior cobertura. |
| 15 km | Compromisso ainda conservador; deve ser documentado por região/rodovia. |
| 20 km | Próximo à mediana observada; ainda pode errar eventos localizados. |
| 30 km | Limite principal atual; mantém 3.730 acidentes finais, mas já admite erro de exposição relevante. |
| 40 km | Maior amostra: 4.339; maior risco de erro espacial. |
| 50 km | 5.015; risco elevado de atribuição incorreta de chuva. |

Não houve, nos outputs disponíveis, quantidade para 5, 10, 15 ou 20 km. Não se deve inventar esses valores.

A abordagem mais defensável para o artigo é apresentar uma análise principal pré-especificada com limite conservador e análises de sensibilidade em múltiplos limites, em vez de escolher o corte pelo menor p-valor. Idealmente, comparar estação mais próxima, múltiplas estações candidatas e uma ponderação por distância; IDW só é apropriado se houver estações suficientes e cobertura espacial adequada.

## 7. Janela temporal

A diferença temporal foi média de 14,77 min, mediana de 15 min e máximo de 30 min. Portanto, o filtro de 60 min não exclui ninguém no resultado atual; é redundante para esse algoritmo de “mais próxima”.

Recomendação:

- análise principal usando observação anterior ao acidente, se a definição horária do INMET permitir;
- sensibilidade com 30 e 60 minutos;
- medida de precipitação acumulada anterior, por exemplo 1 h, 3 h e 6 h;
- comparar “chuva no intervalo do acidente” com “pista potencialmente molhada por chuva recente”.

Usar só a observação mais próxima pode perder o efeito de umidade residual na pista.

## 8. Missing data

A exclusão completa pode ser aceitável apenas como análise principal condicionada a diagnóstico e sensibilidades; ela não deve ser descrita como ausência aleatória sem evidência.

- **MCAR:** improvável de provar; exigiria ausência sem associação com espaço, tempo, estação, gravidade e demais variáveis observadas.
- **MAR:** plausível se a ausência for explicada por estação, período, região e variáveis observadas.
- **MNAR:** permanece possível se falhas forem relacionadas à própria chuva intensa ou ao funcionamento do sensor sob condições adversas.

Imputar precipitação pontual é geralmente fraco para uma exposição meteorológica local. Melhor alternativa: procurar observação válida em estação candidata; depois fazer análise de casos completos e uma sensibilidade com categoria/indicador de ausência, sem interpretar “missing” como “sem chuva”.

## 9. Regressão logística ordinal: explicação e auditoria

O desfecho é ordinal porque “sem vítimas”, “feridos” e “fatais” têm ordem de gravidade. A distância entre 0, 1 e 2, porém, não é numérica nem equivalente; por isso regressão linear é inadequada.

O modelo ordinal estima logits cumulativos, por exemplo:

- probabilidade de gravidade ser pelo menos “feridos”;
- probabilidade de gravidade ser “fatal”.

Com chances proporcionais, o coeficiente de `precipitacao_mm` é o mesmo nos dois cortes. Se β for positivo, maior precipitação estaria associada a maiores chances de estar em categoria mais grave; `exp(β)` é o OR. Se β for negativo, ocorre o oposto.

A implementação usa corretamente `OrderedModel(..., distr="logit")`, codifica a resposta como ordenada e não adiciona constante indevida. As referências aparentes são:

- `fase_dia`: Pleno dia;
- `tipo_pista`: Dupla;
- `tracado_via`: Reta;
- ano: 2025;
- mês: janeiro.

Limitações da implementação:

- não há teste formal de chances proporcionais;
- não há diagnóstico formal de separação;
- não há verificação formal de colinearidade;
- ICs e erros-padrão são não robustos;
- não há ajuste por cluster de rodovia, estação ou município;
- o modelo usa precipitação linear sem justificar essa forma funcional;
- faltam intervalos/efeitos em escalas interpretáveis, como 5 mm ou chuva sim/não.

O resultado multinomial preservado não mostra associação estatisticamente convincente:

- feridos vs sem vítimas, por mm: OR 0,973; IC95% 0,944–1,003;
- fatais vs sem vítimas, por mm: OR 0,989; IC95% 0,924–1,058;
- chuva binária, fatal vs sem vítimas: OR 0,554; IC95% 0,265–1,159.

Esses números não permitem concluir que a chuva reduza gravidade; os intervalos são compatíveis com ausência de associação e com efeitos em ambas as direções.

## 10. Modelos recomendados

| Modelo | Papel recomendado |
|---|---|
| Ordinal logístico | Principal apenas se proporcionalidade for aceitável. |
| Odds proporcionais parciais | Principal alternativo se somente algumas covariáveis violarem proporcionalidade. |
| Multinomial | Complementar/alternativo se a estrutura ordinal não se sustentar. |
| GAM ordinal ou spline em precipitação | Complementar para investigar não linearidade com interpretação inferencial. |
| Modelo misto/hierárquico | Complementar se houver clustering relevante por estação, rodovia, município ou região. |
| Random Forest/XGBoost | Exploração preditiva, não evidência principal de associação. |
| SVM | Não recomendado: pouca interpretabilidade e baixa aderência ao objetivo inferencial. |
| Árvores simples | Exploratórias; úteis para detectar interações, não para efeito principal causal/inferencial. |

Modelos de ML podem indicar padrões e interações, mas não resolvem confundimento, erro de exposição, missingness ou causalidade. Não devem substituir a regressão inferencial no artigo.

## 11. Confundimento, vieses e controles

Controles plausíveis: sazonalidade, ano, fase do dia, tipo de pista, traçado, rodovia, município/região e possivelmente estação meteorológica.

Possíveis problemas:

- `fase_dia` pode ser confundidor, mas também proxy de volume de tráfego, luminosidade e comportamento;
- `mês` é importante para sazonalidade, mas pode não capturar eventos meteorológicos regionais;
- rodovia/município/região podem capturar geografia, infraestrutura e cobertura meteorológica;
- condição meteorológica registrada pela PRF não deve ser usada automaticamente como controle: pode ser uma medida concorrente da exposição e ter erro diferencial;
- não há controle para fluxo de tráfego, velocidade, tipo de veículo, ocupantes, tipo de acidente, iluminação, condição do pavimento ou resposta de emergência; portanto, não se deve fazer alegações causais.

Além disso, a base condiciona a análise à ocorrência de acidente. O estudo responde à associação entre chuva medida e gravidade entre acidentes registrados, não ao risco de acidente na população exposta.

## 12. Análises de sensibilidade mínimas

1. Limites espaciais: 5, 10, 15, 20, 30, 40 e 50 km.
2. Regra temporal: observação anterior; 30/60 min; acumulados anteriores.
3. Estação mais próxima versus estação próxima com dado válido.
4. Casos completos versus estratégia explícita para ausência.
5. Chuva contínua, binária, categórica e spline.
6. Inclusão/exclusão justificada de extremos, sem remoção automática.
7. Ordinal, odds proporcionais parciais e multinomial.
8. Modelos com e sem controles espaciais/rodovia/estação.
9. Resultados estratificados por litoral/interior, ano e faixas de distância.

O resultado já mostra sensibilidade espacial: para chuva binária, o OR ordinal foi 0,831 em 30 km, 0,788 em 40 km e 0,775 em 50 km; apenas em 50 km atingiu p < 0,05. Isso reforça que não se deve selecionar a regra pelo resultado estatístico.

## 13. Checklists

### Base final

- [ ] Sem duplicidades de acidente.
- [ ] Coordenadas PRF válidas e dentro da área plausível.
- [ ] Timestamp do acidente validado.
- [ ] Uma observação INMET por estação–timestamp, sem duplicidade.
- [ ] Sentinelas, negativos e valores inválidos tratados separadamente de zero.
- [ ] Cobertura por estação, ano e mês documentada.
- [ ] Distância e diferença temporal preservadas.
- [ ] Regra de escolha da estação documentada antes da análise.

### Antes do modelo final

- [ ] Fluxograma completo de exclusões.
- [ ] Distribuição da gravidade e número de eventos fatais suficiente.
- [ ] Forma funcional da precipitação justificada.
- [ ] Proporcionalidade formalmente avaliada.
- [ ] Diagnósticos de estabilidade, influência e colinearidade.
- [ ] Erros-padrão/clusterização considerados.
- [ ] Sensibilidades espaciais, temporais e de missingness executadas.
- [ ] Interpretação em associação, não causalidade.

### Escrita do artigo

- [ ] Descrever que a chuva é medida em estação, não diretamente no acidente.
- [ ] Informar distância e diferença temporal completas, não só médias.
- [ ] Distinguir claramente 0 mm de dado ausente.
- [ ] Explicar exclusões e perda de amostra.
- [ ] Relatar incerteza com IC95%, não só p-valor.
- [ ] Não afirmar que chuva “causa” gravidade.
- [ ] Relatar limitações de cobertura espacial, temporal e de seleção.

## 14. Plano priorizado

1. Auditar integralmente os arquivos INMET brutos: sentinelas, duplicidades, cobertura e extremos.
2. Redesenhar e documentar a regra conjunta espaço–tempo–validade.
3. Pré-especificar limites espaciais/temporais e análises de sensibilidade.
4. Investigar mecanismo de ausência e comparabilidade entre incluídos/excluídos.
5. Testar formalmente chances proporcionais; definir ordinal parcial ou multinomial conforme o resultado.
6. Modelar chuva com forma não linear/clinicamente-meteorologicamente interpretável.
7. Incluir estrutura espacial/rodovia/estação ou discutir explicitamente sua ausência.
8. Consolidar a redação apenas após confirmar que artigo, notebook e outputs usam a mesma versão do pipeline.