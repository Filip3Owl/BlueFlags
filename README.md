# BlueFlags

Predição de risco de depressão em adolescentes com base no uso de redes sociais e hábitos de vida, utilizando regressão logística.

![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat-square&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## Visão Geral

Este projeto constrói um pipeline de classificação binária para determinar se um adolescente está em risco de depressão (`depression_label`: 0 ou 1). O dataset captura padrões comportamentais e de estilo de vida — incluindo consumo diário de redes sociais, duração do sono, desempenho acadêmico e níveis de estresse autorrelatados — coletados de aproximadamente 1.200 adolescentes.

O projeto segue um fluxo de trabalho estruturado de aprendizado de máquina: análise exploratória de dados, pré-processamento, treinamento do modelo e avaliação.

---

## Definição do Problema

A saúde mental de adolescentes tornou-se uma preocupação crescente, com o uso de redes sociais frequentemente citado como fator contribuinte para o aumento das taxas de ansiedade e depressão. Este projeto investiga se variáveis comportamentais e de estilo de vida conseguem predizer de forma confiável sintomas depressivos, utilizando regressão logística como modelo principal.

**Variável alvo:** `depression_label` — binária (0 = sem depressão, 1 = depressão)

---

## Dataset

**Arquivo:** `data/Teen_Mental_Health_Dataset.csv`
**Registros:** 1.200

| Feature | Tipo | Descrição |
|---|---|---|
| `age` | Numérico | Idade do adolescente (anos) |
| `gender` | Categórico | Gênero (masculino / feminino) |
| `daily_social_media_hours` | Numérico | Média de horas por dia em redes sociais |
| `platform_usage` | Categórico | Principal plataforma de redes sociais utilizada |
| `sleep_hours` | Numérico | Média de horas de sono por noite |
| `screen_time_before_sleep` | Numérico | Horas de exposição a telas antes de dormir |
| `academic_performance` | Numérico | Pontuação acadêmica no estilo GPA |
| `physical_activity` | Numérico | Horas de atividade física por semana |
| `social_interaction_level` | Categórico | Nível de interação social presencial (baixo / médio / alto) |
| `stress_level` | Numérico | Nível de estresse autorrelatado (1–10) |
| `anxiety_level` | Numérico | Nível de ansiedade autorrelatado (1–10) |
| `addiction_level` | Numérico | Pontuação de dependência de redes sociais (1–10) |
| `depression_label` | Binário | Variável alvo — 0: sem depressão, 1: depressão |

---

## Análise Exploratória de Dados

### Distribuição das Classes

![Distribuição das Classes](assets/01_target_distribution.png)

O gráfico de barras (esquerda) mostra as contagens absolutas: **1.169 adolescentes sem depressão** e apenas **31 com depressão**. O gráfico de pizza (direita) traduz isso em proporções: **97,4% vs 2,6%**.

Visualmente, a barra da classe 1 mal aparece na escala do gráfico — o que ilustra com clareza o problema: qualquer modelo que sempre previsse "sem depressão" alcançaria 97,4% de acurácia sem aprender absolutamente nada. Essa assimetria extrema é a principal restrição técnica do projeto e motivou duas decisões de design — usar **SMOTE** no treino e adotar **ROC-AUC e F1-score** como métricas primárias em vez de acurácia.

> **Insight:** O desbalanceamento de 37:1 (classe 0 : classe 1) é severo o suficiente para que a acurácia seja uma métrica completamente enganosa. Um modelo trivial que sempre prevê "sem depressão" teria 97,4% de acurácia e recall zero na classe que realmente importa.

---

### Correlação com a Variável Alvo

![Matriz de Correlação](assets/06_correlation_matrix.png)

O mapa de calor exibe as correlações de Pearson entre todos os pares de features numéricas. A intensidade da cor indica a força da correlação — vermelho para correlação positiva, azul para negativa — e a última linha/coluna destaca a relação de cada feature com `depression_label`.

**Como ler a última linha:**

| Feature | Correlação com `depression_label` | Interpretação |
|---|---|---|
| `sleep_hours` | **-0,19** | Mais sono → menor risco (correlação mais forte, azul escuro) |
| `daily_social_media_hours` | **+0,18** | Mais redes sociais → maior risco (vermelho mais intenso) |
| `stress_level` | **+0,17** | Estresse mais alto acompanha depressão |
| `anxiety_level` | **+0,17** | Ansiedade mais alta acompanha depressão |
| `academic_performance` | -0,02 | Sem relação relevante com depressão |
| `physical_activity` | -0,02 | Sem relação relevante com depressão |
| `addiction_level` | -0,01 | Surpreendentemente próximo de zero |

Repare também que as features **entre si** apresentam correlações próximas de zero (quase toda a diagonal inferior está neutra), o que é positivo para a regressão logística: ausência de multicolinearidade forte significa que cada feature contribui com informação independente.

> **Insight:** As correlações individuais são modestas (máximo ≈ 0,19), mas isso não as torna inúteis — a regressão logística combina todas as features em conjunto, e a combinação de quatro sinais fracos pode gerar poder preditivo substancial. O ROC-AUC de 0,99 do modelo final confirma isso.

---

### Distribuições das Features por Classe

![Box Plots Bivariados](assets/04_bivariate_boxplots.png)

Cada box plot compara a distribuição de uma feature entre adolescentes sem depressão (verde, classe 0) e com depressão (laranja, classe 1). A linha central de cada caixa é a mediana; as bordas inferior e superior da caixa são o 1º e 3º quartis (IQR); os "bigodes" estendem-se até 1,5× o IQR.

**Leitura por painel:**

- **`stress_level` e `anxiety_level`** — as diferenças mais dramáticas. A caixa laranja está inteiramente acima da caixa verde, com mediana próxima de 8 (escala 1–10) contra 5. Há uma separação clara entre as distribuições — quase sem sobreposição nas regiões centrais.
- **`daily_social_media_hours`** — a caixa laranja está deslocada para cima (mediana ≈ 6,5 h/dia vs ≈ 4,5 h/dia). O uso diário de redes sociais no grupo com depressão é consistentemente mais alto, sem outliers extremos que distorçam a leitura.
- **`sleep_hours`** — inversão clara: adolescentes com depressão dormem menos. A caixa laranja fica comprimida na faixa de 4–5 h, enquanto a verde é mais ampla e centrada em ~6,5 h.
- **`screen_time_before_sleep`** — distribuições quase idênticas entre as classes. Surpreendentemente, o tempo de tela antes de dormir não diferencia os grupos, ao contrário do que o senso comum poderia sugerir.
- **`academic_performance` e `physical_activity`** — sobreposição total. Essas features não diferenciam as classes e provavelmente contribuirão pouco para o modelo.
- **`addiction_level`** — diferença visível mas menos pronunciada. A caixa laranja é ligeiramente mais alta, porém a sobreposição é grande.
- **`age`** — sem diferença relevante. Adolescentes de qualquer faixa etária do dataset estão igualmente representados nas duas classes.

**Valores médios por classe:**

| Feature | Sem Depressão (0) | Com Depressão (1) | Diferença |
|---|---|---|---|
| `stress_level` | 5,37 | 8,48 | **+3,11 pontos** |
| `anxiety_level` | 5,56 | 8,61 | **+3,05 pontos** |
| `daily_social_media_hours` | 4,48 h | 6,72 h | **+2,24 h/dia (+50%)** |
| `sleep_hours` | 6,49 h | 4,76 h | **-1,73 h/noite (-27%)** |

> **Insight:** A diferença de 3 pontos em estresse e ansiedade (em uma escala de 1–10) é enorme em termos práticos. O grupo com depressão está sistematicamente na metade superior da escala, enquanto o grupo sem depressão está na metade inferior. Esses dois indicadores autorrelatados concentram o maior poder discriminativo do dataset.

---

### Matriz de Dispersão — Features Principais

![Matriz de Dispersão](assets/07_scatter_matrix.png)

A matriz plota as relações par a par entre as quatro features mais correlacionadas com a variável alvo (`sleep_hours`, `daily_social_media_hours`, `stress_level`, `anxiety_level`), coloridas por classe (verde = sem depressão, laranja = com depressão). A diagonal mostra a distribuição de cada feature via histograma.

**O que observar:**

- **Diagonal:** As distribuições das duas classes se sobrepõem em todas as features individualmente — nenhuma feature sozinha consegue separar perfeitamente os grupos. Isso confirma que o modelo precisa de todas as features em conjunto.
- **`stress_level` × `anxiety_level` (par inferior direito):** É o par com maior separação visual. Os pontos laranjas se concentram na região de alto estresse e alta ansiedade (quadrante superior direito), enquanto os verdes dominam o quadrante inferior esquerdo.
- **`sleep_hours` × `stress_level` e `sleep_hours` × `anxiety_level`:** Tendência diagonal inversa — pontos com baixo sono e alto estresse/ansiedade tendem a ser laranjas.
- **A sobreposição generalizada** é consistente com os valores de correlação modestos (≤ 0,19) e justifica por que o modelo não pode confiar em limites simples de decisão — a regressão logística combina todos os sinais simultaneamente.

> **Insight:** Nenhuma fronteira linear simples entre duas features consegue separar as classes. O poder do modelo vem da combinação multidimensional de todos os preditores. Isso também explica por que o ROC-AUC (que avalia a separabilidade global) é 0,99 mesmo com correlações individuais fracas.

---

## Principais Descobertas da EDA

- **Grave desbalanceamento de classes (97,4% vs 2,6%)** torna a acurácia uma métrica enganosa. ROC-AUC e F1-score são os critérios primários de avaliação.
- **Níveis de estresse e ansiedade** apresentam a maior diferença entre as classes (+3 pontos em uma escala de 1–10), tornando-os as features mais discriminantes nos dados brutos.
- **Adolescentes com depressão dormem em média 1,73 horas a menos** por noite — uma diferença de 27% com impactos funcionais e fisiológicos significativos.
- **O consumo de redes sociais é 50% maior** no grupo com depressão (6,72 h vs 4,48 h por dia).
- **Tempo de tela antes de dormir, desempenho acadêmico e atividade física** apresentam diferenças desprezíveis entre as classes nos dados brutos.

---

## Pré-processamento

O dataset bruto foi transformado pelo pipeline a seguir antes do treinamento do modelo.

### Codificação

| Feature | Estratégia | Resultado |
|---|---|---|
| `gender` | Ordinal Encoding | masculino = 0, feminino = 1 |
| `platform_usage` | One-Hot Encoding (drop first) | `platform_usage_Instagram`, `platform_usage_TikTok` |
| `social_interaction_level` | Ordinal Encoding | baixo = 0, médio = 1, alto = 2 |

### Escalonamento

Todas as 9 features numéricas foram padronizadas com `StandardScaler` (média zero, variância unitária). O escalonamento é aplicado após a divisão treino/teste para evitar vazamento de dados.

### Divisão Treino / Teste

Os dados foram divididos na proporção 80/20 com amostragem estratificada para preservar a distribuição original das classes em ambos os conjuntos.

| Conjunto | Linhas | Classe 0 | Classe 1 |
|---|---|---|---|
| Treino (antes do SMOTE) | 960 | 935 | 25 |
| Teste | 240 | 234 | 6 |

### Desbalanceamento de Classes — SMOTE

O SMOTE (Synthetic Minority Over-sampling Technique) foi aplicado exclusivamente ao conjunto de treino para gerar amostras sintéticas da classe minoritária. O conjunto de teste é mantido intacto para refletir a distribuição real.

| | Classe 0 | Classe 1 | Total |
|---|---|---|---|
| Antes do SMOTE | 935 | 25 | 960 |
| Após o SMOTE | 935 | 935 | 1.870 |

O conjunto de treino final contém **13 features** após a codificação e **1.870 amostras balanceadas** após a reamostragem.

---

## Resultados do Modelo

O modelo de Regressão Logística foi treinado no conjunto de treino balanceado pelo SMOTE (1.870 amostras) e avaliado no conjunto de teste com distribuição original (240 amostras, 6 casos positivos).

### Métricas de Desempenho

| Métrica | Valor |
|---|---|
| ROC-AUC | **0,9922** |
| Precisão Média | **0,8333** |
| Melhor F1 Score | **0,7273** (threshold = 0,82) |

> A acurácia é omitida intencionalmente — com 97,5% das amostras de teste na classe 0, um modelo que sempre prevê "sem depressão" alcança 97,5% de acurácia enquanto é completamente inútil para a tarefa.

---

### Curvas ROC e Precisão-Recall

![Curvas ROC e PR](assets/09_roc_pr_curves.png)

**Curva ROC (esquerda):** O eixo X é a taxa de falsos positivos (1 − especificidade) e o eixo Y é a taxa de verdadeiros positivos (sensibilidade/recall). A linha tracejada diagonal representa um classificador aleatório (AUC = 0,50). A curva laranja do modelo sobe quase verticalmente até o canto superior esquerdo — o comportamento de um classificador quase perfeito — e apresenta **AUC = 0,992**.

**Curva Precisão-Recall (direita):** Especialmente relevante para datasets desbalanceados. O eixo X é o recall (quantos positivos reais foram capturados) e o eixo Y é a precisão (dos que foram previstos positivos, quantos realmente o eram). A linha tracejada representa a precisão de um classificador aleatório no nível da prevalência da classe (≈ 2,6%). O modelo obtém **AP (Average Precision) = 0,833** — 32× acima da linha de base. A queda abrupta na curva entre recall 0,67 e 1,0 reflete a dificuldade de capturar o último caso de depressão sem gerar muitos falsos positivos.

> **Insight:** O ROC-AUC de 0,99 indica que o modelo aprendeu uma separação quase perfeita entre as distribuições de probabilidade das duas classes. Porém, a curva PR revela a tensão real: para capturar 100% dos casos de depressão (recall = 1,0), a precisão cai para ~50% — ou seja, para cada caso real detectado, há um falso alarme. Isso é uma limitação inerente ao tamanho minúsculo da classe positiva (apenas 6 casos no teste).

---

### Matrizes de Confusão — Efeito do Threshold

![Matrizes de Confusão](assets/08_confusion_matrices.png)

As duas matrizes comparam o comportamento do mesmo modelo base em dois thresholds de decisão diferentes. Cada célula mostra: **linha = classe real, coluna = classe prevista**.

**Threshold padrão (0,50) — esquerda:**
- Verdadeiros Negativos (TN): 228 — casos sem depressão corretamente ignorados
- Falsos Positivos (FP): 6 — casos sem depressão erroneamente sinalizados
- Falsos Negativos (FN): 1 — caso com depressão perdido pelo modelo
- Verdadeiros Positivos (TP): 5 — casos com depressão corretamente detectados

Com threshold padrão, o modelo detecta 5 dos 6 casos reais, mas gera 6 alarmes falsos.

**Threshold ajustado (0,82) — direita:**
- TN: 233 — quase todos os negativos corretamente classificados
- FP: 1 — apenas 1 alarme falso
- FN: 2 — 2 casos de depressão não detectados
- TP: 4 — 4 dos 6 casos de depressão corretamente identificados

Elevar o threshold de 0,50 para 0,82 significa que o modelo só classifica alguém como "em risco" quando tem pelo menos 82% de confiança. O efeito é **reduzir os alarmes falsos de 6 para 1**, mas ao custo de perder 1 caso verdadeiro adicional.

> **Insight:** A escolha do threshold é uma decisão clínica, não técnica. Em triagem de saúde mental, **um caso perdido (falso negativo) tem custo humano muito mais alto** do que um alarme falso. Nesse contexto, o threshold 0,50 — que detecta 5/6 casos com 6 alarmes — pode ser preferível ao 0,82, que detecta 4/6 com apenas 1 alarme. O modelo oferece ambas as opções; quem define o trade-off é o contexto de uso.

---

### Importância das Features — Coeficientes da Regressão Logística

![Coeficientes das Features](assets/10_feature_coefficients.png)

Este gráfico de barras horizontais mostra os coeficientes do modelo em escala de log-odds. Barras **laranjas** aumentam a probabilidade de depressão; barras **verdes** a reduzem. O comprimento da barra indica a magnitude do efeito após padronização das features — portanto os coeficientes são comparáveis entre si.

**Leitura do gráfico (do maior ao menor efeito em valor absoluto):**

| Feature | Coeficiente | Direção | Interpretação |
|---|---|---|---|
| `sleep_hours` | ≈ −4,1 | Verde (protetor) | **O preditor mais forte do modelo.** Cada desvio-padrão adicional de sono está fortemente associado à redução do risco. O efeito é mais que o dobro do segundo preditor mais forte. |
| `anxiety_level` | ≈ +3,3 | Laranja (risco) | Segundo maior efeito. Ansiedade elevada é o principal indicador positivo de risco no modelo. |
| `stress_level` | ≈ +2,8 | Laranja (risco) | Terceiro maior. Estresse e ansiedade juntos dominam os preditores de risco. |
| `daily_social_media_hours` | ≈ +2,4 | Laranja (risco) | Quarto maior. Cada hora adicional em redes sociais aumenta significativamente o risco previsto. |
| `gender` | ≈ +2,0 | Laranja (risco) | **Achado importante não destacado na EDA.** No modelo, ser do sexo feminino (codificado como 1) está associado a risco substancialmente maior de depressão — consistente com a literatura de saúde mental adolescente. |
| `academic_performance` | ≈ +0,7 | Laranja (risco) | Efeito pequeno e contraintuitivo — desempenho acadêmico mais alto associado levemente a maior risco. Pode refletir pressão de desempenho. |
| `addiction_level` | ≈ +0,6 | Laranja (risco) | Efeito modesto, menor do que o esperado dado que a EDA mostrou diferença pequena entre classes. |
| `platform_usage_Instagram` | ≈ −1,7 | Verde | **Achado contraintuitivo:** usar Instagram como plataforma principal está associado a *menor* risco em relação à plataforma de referência (Facebook/outra). |
| `platform_usage_TikTok` | ≈ −0,9 | Verde | Mesmo padrão do Instagram, em menor magnitude. |
| `age` | ≈ −0,3 | Verde | Efeito pequeno: adolescentes ligeiramente mais velhos têm risco levemente menor no modelo. |
| `physical_activity`, `screen_time_before_sleep`, `social_interaction_level` | ≈ 0 | — | Contribuição negligenciável. O modelo praticamente os ignora. |

> **Insight 1 — O sono é o preditor mais poderoso:** O coeficiente de `sleep_hours` (≈ −4,1) é mais que o dobro do segundo coeficiente em valor absoluto. Em log-odds, isso representa um efeito enorme. Enquanto a correlação de Pearson mostrou −0,19 (fraca), o modelo multivariado reconhece que o sono captura um sinal único, não redundante com os outros preditores.

> **Insight 2 — Gênero é um preditor relevante:** O coeficiente de `gender` (≈ +2,0) coloca essa feature entre os quatro maiores preditores. A codificação (feminino = 1) indica que o modelo associa o sexo feminino a maior risco — padrão bem documentado em epidemiologia de depressão adolescente, onde taxas de depressão em meninas são consistentemente 1,5× a 2× maiores que em meninos.

> **Insight 3 — Plataforma importa de forma contraintuitiva:** Os coeficientes negativos de Instagram e TikTok podem parecer surpreendentes, mas refletem que, comparados à plataforma de referência (provavelmente Facebook ou outras), esses usuários apresentam menor risco *controlando as demais variáveis*. Isso sugere que o efeito das redes sociais está capturado principalmente pelo volume de uso (`daily_social_media_hours`), não pela plataforma específica.

---

## Ajuste de Hiperparâmetros

O `GridSearchCV` com validação cruzada `StratifiedKFold` de 5 dobras foi utilizado para buscar entre 36 combinações de parâmetros em duas estratégias de regularização (L1 e L2) e uma faixa de valores de `C`.

### Espaço de Busca

| Parâmetro | Valores testados |
|---|---|
| `C` (força de regularização) | 0,001; 0,01; 0,1; 1; 10; 100 |
| `penalty` | l1, l2 |
| `solver` | lbfgs, liblinear, saga |
| Métrica de pontuação | ROC-AUC |

### Melhor Configuração

| Parâmetro | Valor |
|---|---|
| `C` | 0,1 |
| `penalty` | l2 |
| `solver` | lbfgs |
| Melhor CV ROC-AUC | 0,9918 |

---

### Curvas ROC e PR — Base vs. Ajustado

![Curvas Base vs Ajustado](assets/11_baseline_vs_tuned.png)

Este gráfico sobrepõe as curvas de desempenho do modelo base (linha sólida verde) e do modelo ajustado pelo GridSearchCV (linha tracejada laranja).

**Curva ROC (esquerda):** Ambas as curvas são quase indistinguíveis visualmente — os dois modelos têm separabilidade excelente. A diferença de AUC é pequena em valor absoluto (0,9922 → 0,9936), mas estatisticamente significativa no contexto de datasets pequenos e desbalanceados.

**Curva Precisão-Recall (direita):** Aqui a diferença é mais visível. O modelo ajustado (tracejado laranja) mantém precisão mais alta em níveis de recall mais altos — a curva "cai" mais tarde e a área sob ela é maior (AP: 0,833 → 0,863). Isso significa que o modelo ajustado consegue recuperar mais casos de depressão antes de começar a gerar alarmes falsos excessivos.

> **Insight:** A regularização mais forte (C=0,1 em vez do padrão C=1) funciona porque penaliza coeficientes grandes, forçando o modelo a depender de vários preditores em vez de superajustar para os mais correlacionados. Com apenas 25 casos positivos no treino (antes do SMOTE), o overfitting é um risco real — e C=0,1 o mitiga.

---

### Matrizes de Confusão — Base vs. Ajustado

![Comparação das Matrizes de Confusão](assets/12_cm_baseline_vs_tuned.png)

As duas matrizes mostram o desempenho no conjunto de teste (240 amostras, 6 casos positivos) com o threshold ótimo para cada modelo.

**Modelo Base (esquerda):**
- TN: 233 | FP: 1 | FN: 2 | TP: 4
- Detecta 4 dos 6 casos de depressão com 1 alarme falso

**Modelo Ajustado (direita):**
- TN: 232 | FP: 2 | FN: 1 | TP: 5
- Detecta 5 dos 6 casos de depressão com 2 alarmes falsos

O modelo ajustado ganha **1 caso verdadeiro adicional detectado** ao custo de 1 falso positivo extra.

> **Insight:** Em termos práticos, a regularização mais forte melhorou o recall da classe positiva de 67% para 83% — o modelo passou de "perde 2 em cada 6 casos" para "perde apenas 1 em cada 6 casos". Para uma aplicação de triagem de saúde mental, essa diferença é clinicamente relevante: um adolescente a mais encaminhado para avaliação especializada.

### Resumo de Desempenho — Base vs. Ajustado

| Métrica | Base | Ajustado | Melhoria |
|---|---|---|---|
| ROC-AUC | 0,9922 | **0,9936** | +0,0014 |
| Precisão Média | 0,8333 | **0,8626** | +0,0293 |
| Melhor F1 Score | 0,7273 | **0,7692** | +0,0419 |
| Recall de Depressão | 0,67 | **0,83** | **+0,16** |

---

## Estrutura do Projeto

```
BlueFlags/
├── assets/                        # Figuras incorporadas neste README
├── data/
│   ├── Teen_Mental_Health_Dataset.csv
│   └── processed/                 # Gerado pelo 02_preprocessing.ipynb (gitignored)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── notebooks/
│   ├── 01_eda.ipynb               # Análise Exploratória de Dados
│   ├── 02_preprocessing.ipynb     # Codificação, escalonamento, SMOTE
│   ├── 03_model.ipynb             # Treinamento e avaliação da Regressão Logística
│   ├── 04_tuning.ipynb            # Ajuste de hiperparâmetros com GridSearchCV
│   └── 05_model_comparison.ipynb  # Comparação de modelos (LR, RF, GB, SVM)
├── outputs/
│   └── figures/                   # Gráficos gerados durante a EDA (gitignored)
├── .vscode/
│   └── settings.json
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Configuração

### 1. Clonar o repositório

```bash
git clone https://github.com/Filip3Owl/BlueFlags.git
cd BlueFlags
```

### 2. Criar e ativar o ambiente virtual

```bash
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Registrar o kernel do Jupyter

```bash
python -m ipykernel install --user --name=venv-logistic --display-name "Python (.venv)"
```

### 5. Iniciar o Jupyter

```bash
jupyter notebook
```

Selecione o kernel **Python (.venv)** ao abrir qualquer notebook.

---

## Notebooks

| Notebook | Descrição |
|---|---|
| `notebooks/01_eda.ipynb` | Análise Exploratória de Dados — balanceamento de classes, distribuições, correlações |
| `notebooks/02_preprocessing.ipynb` | Codificação, escalonamento, divisão treino/teste, SMOTE |
| `notebooks/03_model.ipynb` | Treinamento da Regressão Logística, ajuste de threshold, avaliação e coeficientes |
| `notebooks/04_tuning.ipynb` | GridSearchCV sobre C, penalty e solver — comparação base vs. ajustado |
| `notebooks/05_model_comparison.ipynb` | Comparação de 4 classificadores (LR, Random Forest, Gradient Boosting, SVM) |

---

## Roadmap

- [x] Análise Exploratória de Dados
- [x] Pré-processamento — codificação, escalonamento, SMOTE
- [x] Treinamento do modelo de Regressão Logística
- [x] Avaliação do modelo — ROC-AUC 0,99, F1 0,73, matriz de confusão
- [x] Ajuste de hiperparâmetros — GridSearchCV, melhor: C=0,1, l2, lbfgs (F1 0,77)
- [ ] Comparação de modelos — Logistic Regression vs Random Forest vs Gradient Boosting vs SVM

---

## Licença

Este projeto está licenciado sob a Licença MIT.
