# Explicação dos Resultados — Console Output

---

## Dataset 1 — Wine Quality

### Visão geral

| Info | Valor |
|---|---|
| Total de amostras | 6.497 |
| Atributos | 14 (físico-químicos + tipo + target) |
| Classe **Bom** (nota ≥ 6) | 4.113 (63%) |
| Classe **Ruim** (nota < 6) | 2.384 (37%) |

O dataset é levemente desbalanceado — há quase o dobro de vinhos "Bons".

---

### KNN — Variação de k

Cada linha mostra a acurácia para um valor de k (número de vizinhos consultados):

| k | Treino | Teste | CV (5-fold) | Observação |
|---|---|---|---|---|
| 1 | 1.000 | 0.772 | 0.770 | **Overfitting claro** — acerta tudo no treino, pouco no teste |
| 3 | 0.882 | 0.739 | 0.750 | Ainda instável |
| 5–7 | ~0.82–0.84 | ~0.739–0.748 | ~0.750 | Melhorando |
| 11 | 0.798 | 0.756 | 0.747 | Melhor equilíbrio na região |
| 21–51 | ~0.76–0.77 | ~0.742–0.743 | ~0.749–0.752 | Estabiliza — modelo mais simples |

> **O que é overfitting?** Quando `train` é muito maior que `test`, o modelo "memorizou" os dados de treino e não generaliza bem para dados novos. No k=1, cada ponto consulta apenas ele mesmo — acerto perfeito no treino, mas fraco no teste.

> **CV (validação cruzada)** divide o treino em 5 partes e treina/testa 5 vezes, rotacionando qual parte é o teste. A média é uma estimativa mais confiável da performance real do que um único teste.

---

### KNN — Variação de `weights` e `metric` (k=21 fixo)

| Configuração | Acurácia | F1 |
|---|---|---|
| uniform / euclidean | 0.742 | 0.808 |
| uniform / manhattan | 0.756 | 0.818 |
| **distance / euclidean** | 0.812 | 0.860 |
| **distance / manhattan** | **0.820** | **0.865** |

- **`weights=distance`** dá mais peso a vizinhos mais próximos, o que melhora bastante (de ~0.74 para ~0.82).
- **`metric=manhattan`** (soma das diferenças absolutas) superou a euclidiana neste dataset.

**Melhor configuração encontrada por CV:** `k=15, weights=distance, metric=manhattan` → CV = **0.8083**

---

### Árvore de Decisão — Variação de `max_depth`

| Profundidade | Treino | Teste | Observação |
|---|---|---|---|
| 2–3 | 0.738 | 0.734 | Modelo muito simples (underfitting) |
| 4–8 | 0.752–0.841 | 0.741–0.743 | Ganho no treino, teste estagna |
| 12 | 0.939 | 0.767 | Começa a overfittar |
| None (sem limite) | **1.000** | 0.767 | Overfitting total — decora o treino |

**Melhor profundidade por CV:** `max_depth=12` → CV = **0.7408**

> A árvore sem limite de profundidade atinge acurácia perfeita no treino (1.000), mas não melhora no teste — evidência clássica de overfitting. Limitar a profundidade funciona como **regularização**.

---

## Dataset 2 — Students Dropout

### Visão geral

| Info | Valor |
|---|---|
| Total de amostras | 4.424 |
| Atributos | 37 |
| **Graduate** (formados) | 2.209 (50%) |
| **Dropout** (desistiram) | 1.421 (32%) |
| **Enrolled** (matriculados) | 794 (18%) |

Problema multiclasse com 3 categorias. A classe **Enrolled** é a menor e mais difícil de prever.

---

### Regressão Logística

**Acurácia geral: 77%**

| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| Dropout | 0.81 | 0.75 | 0.78 | 355 |
| **Enrolled** | **0.56** | **0.37** | **0.44** | 199 |
| Graduate | 0.79 | 0.93 | 0.85 | 552 |
| **Média macro** | 0.72 | 0.68 | **0.69** | — |
| Média ponderada | 0.76 | 0.77 | 0.76 | — |

> **Enrolled** tem o pior desempenho (F1=0.44) por ser a classe com menos amostras e mais ambígua — alunos matriculados podem futuramente se formar ou desistir.

> **Recall 0.93 em Graduate** significa que o modelo acerta 93% dos alunos que realmente se formaram.

**Variação de C (regularização):**

| C | Acurácia |
|---|---|
| 0.01 | 0.7676 |
| **0.1** | **0.7712** |
| 1.0 | 0.7694 |
| 10.0 | 0.7694 |

C controla a força da regularização: **C menor = mais regularização** (modelo mais simples). A diferença entre os valores é mínima (~0.4%), indicando que o modelo não é muito sensível a esse hiperparâmetro neste dataset.

---

### Random Forest — Variação de hiperparâmetros

| n_estimators | max_depth | min_samples_leaf | Treino | Teste | F1 macro |
|---|---|---|---|---|---|
| 50 | None | 1 | 1.000 | 0.769 | 0.699 |
| 100 | None | 1 | 1.000 | 0.769 | 0.691 |
| 300 | None | 1 | 1.000 | 0.770 | 0.695 |
| **300** | **10** | **1** | 0.918 | **0.772** | 0.688 |
| 300 | 10 | 5 | 0.864 | 0.767 | 0.675 |
| 300 | 6 | 5 | 0.793 | 0.759 | 0.649 |

**Melhor configuração:** `n_estimators=300, max_depth=10, min_samples_leaf=1` → Teste = **0.7722**

> As três primeiras linhas têm `train=1.000` — overfitting, pois sem limite de profundidade a floresta decora os dados. Limitar `max_depth=10` reduz o treino para 0.918 e **melhora** ligeiramente o teste, confirmando o efeito positivo da regularização.

> Aumentar `min_samples_leaf` de 1 para 5 força folhas maiores (mais conservadoras), mas neste caso penalizou o F1 macro — a regularização foi longe demais para as classes menores.

---

## Resumo final

| Modelo | Dataset | Acurácia (teste) | F1 |
|---|---|---|---|
| KNN (k=15, distance, manhattan) | Wine | ~0.82 | ~0.865 |
| Árvore (max_depth=12) | Wine | 0.767 | — |
| Regressão Logística (C=0.1) | Dropout | 0.771 | 0.69 (macro) |
| Random Forest (300 árvores, depth=10) | Dropout | 0.772 | 0.688 (macro) |

O **KNN com `weights=distance`** foi o grande destaque — o simples ajuste de peso por distância elevou a acurácia de ~0.74 para ~0.82 no dataset de vinhos. No Dropout, Regressão Logística e Random Forest ficaram empatados em acurácia, mas a Regressão foi melhor em F1 macro, sendo mais equilibrada entre as 3 classes.
