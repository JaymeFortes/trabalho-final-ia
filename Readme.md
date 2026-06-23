# Trabalho Final — Inteligência Artificial (PUCRS)

## Modelos utilizados

| Dataset | Modelo obrigatório | Modelo escolhido |
|---|---|---|
| Wine Quality | KNN | Árvore de Decisão |
| Students Dropout | Regressão Logística | Árvore de Decisão |

---

## Estrutura do projeto

```
trabalho-final-ia/
├── analise.py          ← script principal
├── Readme.md           ← este arquivo
└── data/               ← coloque os datasets aqui (ver Passo 1)
```

---

## Passo 1 — Baixar os datasets

Crie a pasta `data/` e salve os arquivos **exatamente** com os nomes abaixo:

| Arquivo esperado | Link |
|---|---|
| `winequality-red.csv` | https://archive.ics.uci.edu/dataset/186/wine+quality |
| `winequality-white.csv` | mesmo link acima (o pacote contém os dois) |
| `students.csv` | https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |

> Os arquivos devem usar `;` como separador — é o formato padrão do UCI.  
> Se o arquivo de dropout vier com outro nome, renomeie para `students.csv`.

Após baixar:

```
data/
├── winequality-red.csv
├── winequality-white.csv
└── students.csv
```

---

## Passo 2 — Instalar dependências

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

> **`pip: command not found`?** Use `python -m pip` no lugar de `pip`.  
> Para corrigir permanentemente no Windows, adicione ao PATH:  
> `C:\Users\<seu-usuario>\AppData\Local\Python\pythoncore-3.x-64\Scripts`

---

## Passo 3 — Executar

```bash
python analise.py
```

---

## O que o script gera

### Figuras (`figuras/`)

| Arquivo | Descrição |
|---|---|
| `wine_dist.png` | Distribuição das notas de qualidade do vinho |
| `wine_corr.png` | Matriz de correlação dos atributos |
| `wine_knn_k.png` | Impacto de k na acurácia do KNN |
| `wine_tree_depth.png` | Overfitting da Árvore (treino vs teste por profundidade) |
| `wine_tree_importance.png` | Importância dos atributos — Árvore Wine |
| `wine_tree_viz.png` | Visualização da árvore podada (max_depth=3) |
| `wine_confusion.png` | Matrizes de confusão — KNN e Árvore (Wine) |
| `drop_dist.png` | Distribuição das classes Dropout/Enrolled/Graduate |
| `drop_lr_C.png` | Impacto do parâmetro C na Regressão Logística |
| `drop_tree_depth.png` | Impacto da profundidade na Árvore (Dropout) |
| `drop_tree_leaf.png` | Impacto do min_samples_leaf na Árvore (Dropout) |
| `drop_tree_criterion.png` | Gini vs Entropy na Árvore (Dropout) |
| `drop_tree_importance.png` | Top 15 atributos mais importantes — Árvore Dropout |
| `drop_tree_viz.png` | Visualização da árvore podada (max_depth=3) |
| `drop_confusion.png` | Matrizes de confusão — Reg. Logística e Árvore (Dropout) |
| `comparacao_geral.png` | Comparação de acurácia e F1 entre os 4 modelos |

### Métricas (`resultados.json`)

Todas as métricas calculadas: acurácia, F1, precisão, recall e variações de hiperparâmetros de todos os modelos.

Ambos são criados automaticamente — não é necessário criar nada manualmente.

---

## Problemas comuns

| Erro | Causa / Solução |
|---|---|
| `FileNotFoundError: data/winequality-red.csv` | Os CSVs não estão na pasta `data/` |
| `FileNotFoundError: data/students.csv` | Renomeie o arquivo de dropout para `students.csv` |
| Colunas estranhas ou erro de leitura | Confirme que os CSVs usam `;` como separador |
| `pip: command not found` | Use `python -m pip` (ver Passo 2) |
