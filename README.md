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

O dataset apresenta um severo **desbalanceamento de classes**: 97,4% dos registros pertencem à classe 0 (sem depressão) e apenas 2,6% à classe 1 (depressão). Esse desbalanceamento é uma restrição crítica para a modelagem e deve ser tratado por meio de técnicas de reamostragem antes do treinamento.

![Distribuição das Classes](assets/01_target_distribution.png)

---

### Correlação com a Variável Alvo

O mapa de calor abaixo exibe as correlações de Pearson par a par entre todas as features numéricas. A última linha destaca a relação de cada feature com `depression_label`.

![Matriz de Correlação](assets/06_correlation_matrix.png)

Os quatro correlatos mais fortes com depressão são:

| Feature | Correlação | Direção |
|---|---|---|
| `sleep_hours` | -0,191 | Menos sono está associado a maior risco de depressão |
| `daily_social_media_hours` | +0,175 | Mais tempo em redes sociais se correlaciona com depressão |
| `stress_level` | +0,171 | Pontuações de estresse mais altas acompanham de perto a depressão |
| `anxiety_level` | +0,170 | Pontuações de ansiedade mais altas acompanham de perto a depressão |

---

### Distribuições das Features por Classe

Box plots revelam diferenças expressivas nas distribuições das features entre adolescentes com e sem depressão. As diferenças mais pronunciadas são observadas em `stress_level`, `anxiety_level`, `sleep_hours` e `daily_social_media_hours`.

![Box Plots Bivariados](assets/04_bivariate_boxplots.png)

**Valores médios por classe:**

| Feature | Sem Depressão (0) | Com Depressão (1) | Diferença |
|---|---|---|---|
| `stress_level` | 5,37 | 8,48 | +3,11 |
| `anxiety_level` | 5,56 | 8,61 | +3,05 |
| `daily_social_media_hours` | 4,48 h | 6,72 h | +2,24 h |
| `sleep_hours` | 6,49 h | 4,76 h | -1,73 h |

---

### Matriz de Dispersão — Features Principais

A matriz de dispersão abaixo plota as relações par a par entre as quatro features mais correlacionadas, coloridas por classe. A sobreposição entre as classes é consistente com os valores de correlação fraco a moderado, confirmando que nenhuma feature isolada é suficiente para a classificação.

![Matriz de Dispersão](assets/07_scatter_matrix.png)

---

## Principais Descobertas

- **Grave desbalanceamento de classes (97,4% vs 2,6%)** torna a acurácia uma métrica enganosa. ROC-AUC e F1-score serão os critérios primários de avaliação.
- **Níveis de estresse e ansiedade** apresentam a maior diferença de média entre as classes (+3 pontos em uma escala de 1–10), tornando-os as features mais discriminantes.
- **Adolescentes com depressão dormem em média 1,73 horas a menos** por noite do que os sem depressão.
- **O consumo de redes sociais é 50% maior** no grupo com depressão (6,72 h vs 4,48 h por dia).
- **Desempenho acadêmico, atividade física e nível de dependência** apresentam diferenças desprezíveis entre as classes e provavelmente não contribuirão de forma significativa para o modelo.

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

### Curvas ROC e Precisão-Recall

![Curvas ROC e PR](assets/09_roc_pr_curves.png)

O ROC-AUC de **0,99** indica separabilidade quase perfeita entre as classes. A curva Precisão-Recall mostra uma Precisão Média de **0,83**, substancialmente acima da linha de base aleatória de 2,6% (prevalência da classe).

### Matrizes de Confusão

![Matrizes de Confusão](assets/08_confusion_matrices.png)

No threshold ajustado (0,82), o modelo identifica corretamente **4 dos 6 casos de depressão** no conjunto de teste, com apenas 3 falsos positivos entre as 234 amostras sem depressão.

### Importância das Features

![Coeficientes das Features](assets/10_feature_coefficients.png)

Os coeficientes confirmam os achados da EDA: **nível de estresse**, **nível de ansiedade** e **horas diárias em redes sociais** são os preditores positivos mais fortes da depressão, enquanto **horas de sono** é o preditor negativo mais forte.

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

### Modelo Base vs. Modelo Ajustado

![Curvas Base vs Ajustado](assets/11_baseline_vs_tuned.png)

![Comparação das Matrizes de Confusão](assets/12_cm_baseline_vs_tuned.png)

| Métrica | Base | Ajustado | Melhoria |
|---|---|---|---|
| ROC-AUC | 0,9922 | **0,9936** | +0,0014 |
| Precisão Média | 0,8333 | **0,8626** | +0,0293 |
| Melhor F1 Score | 0,7273 | **0,7692** | +0,0419 |
| Recall de Depressão | 0,67 | **0,83** | +0,16 |

O modelo ajustado (C=0,1, L2) identifica corretamente **5 dos 6 casos de depressão** no conjunto de teste, contra 4 no modelo base. A regularização mais forte (C=0,1 vs padrão C=1) reduz o overfitting e melhora a generalização para a classe minoritária.

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
│   └── 04_tuning.ipynb            # Ajuste de hiperparâmetros com GridSearchCV
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

---

## Roadmap

- [x] Análise Exploratória de Dados
- [x] Pré-processamento — codificação, escalonamento, SMOTE
- [x] Treinamento do modelo de Regressão Logística
- [x] Avaliação do modelo — ROC-AUC 0,99, F1 0,73, matriz de confusão
- [x] Ajuste de hiperparâmetros — GridSearchCV, melhor: C=0,1, l2, lbfgs (F1 0,77)

---

## Licença

Este projeto está licenciado sob a Licença MIT.
