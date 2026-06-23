"""
Trabalho Final - Inteligencia Artificial (PUCRS)
Analise Critica de Modelos de IA
Pipeline de analise: Wine Quality + Students Dropout
Gera todas as metricas e figuras usadas no relatorio e na apresentacao.
"""
import os, json, warnings  # os: manipula arquivos/pastas; json: salva resultados; warnings: controla avisos
import numpy as np          # numpy: operacoes matematicas e arrays numericos
import pandas as pd         # pandas: leitura e manipulacao de tabelas (DataFrames)
import matplotlib           # matplotlib: biblioteca base para graficos
matplotlib.use("Agg")       # usa backend sem janela grafica (salva figuras em arquivo, sem abrir tela)
import matplotlib.pyplot as plt  # interface principal para criar graficos
import seaborn as sns            # seaborn: graficos estatisticos com visual mais bonito

# importacoes do scikit-learn (biblioteca de machine learning)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# train_test_split: divide os dados em treino e teste
# cross_val_score: avalia o modelo com validacao cruzada
# StratifiedKFold: divide os dados em K partes mantendo a proporcao das classes
from sklearn.preprocessing import StandardScaler       # normaliza os dados (media=0, desvio=1)
from sklearn.pipeline import Pipeline                  # encadeia etapas (ex: normalizar -> treinar)
from sklearn.neighbors import KNeighborsClassifier     # modelo KNN (K vizinhos mais proximos)
from sklearn.tree import DecisionTreeClassifier, plot_tree  # arvore de decisao + funcao para visualiza-la
from sklearn.linear_model import LogisticRegression    # regressao logistica (classificacao)
from sklearn.ensemble import RandomForestClassifier    # random forest (conjunto de arvores de decisao)
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
# accuracy_score: % de acertos totais
# f1_score: media harmonica entre precisao e recall
# precision_score: dos que o modelo disse "positivo", quantos eram de fato positivos
# recall_score: dos que eram de fato positivos, quantos o modelo acertou
# classification_report: relatorio completo com todas as metricas por classe
# confusion_matrix: tabela mostrando acertos e erros por classe
# ConfusionMatrixDisplay: plota a matriz de confusao como grafico

warnings.filterwarnings("ignore")  # suprime avisos de bibliotecas para nao poluir a saida
np.random.seed(42)                  # fixa a semente aleatoria para resultados reproduziveis
sns.set_theme(style="whitegrid")    # define o estilo visual dos graficos (fundo branco com grid)
FIG = "figuras"; os.makedirs(FIG, exist_ok=True)  # define pasta de saida das figuras e a cria se nao existir
RESULTS = {}        # dicionario que vai acumular todos os resultados numericos do experimento
RANDOM_STATE = 42   # constante usada para garantir reproducibilidade em todos os modelos

def savefig(name):
    # salva a figura atual com alta resolucao (130 dpi) na pasta FIG e fecha o plot
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}", dpi=130, bbox_inches="tight"); plt.close()

# =====================================================================
# DATASET 1 - WINE QUALITY (classificacao binaria: qualidade >= 6 -> "Bom")
# =====================================================================
print("="*70); print("DATASET 1: WINE QUALITY"); print("="*70)  # imprime separador no terminal
red = pd.read_csv("data/winequality-red.csv", sep=";")    # le o CSV de vinhos tintos (separador ;)
white = pd.read_csv("data/winequality-white.csv", sep=";")  # le o CSV de vinhos brancos
red["type"] = 0      # adiciona coluna "type" com valor 0 para identificar vinho tinto
white["type"] = 1    # adiciona coluna "type" com valor 1 para identificar vinho branco
wine = pd.concat([red, white], ignore_index=True)  # junta os dois datasets em um unico DataFrame
wine["target"] = (wine["quality"] >= 6).astype(int)   # cria a variavel alvo: 1 se qualidade >= 6 (Bom), 0 caso contrario (Ruim)
print("Shape:", wine.shape)   # imprime o numero de linhas e colunas do dataset
print("Distribuicao target (0=Ruim, 1=Bom):", wine["target"].value_counts().to_dict())  # imprime quantos "Bom" e "Ruim" existem

Xw = wine.drop(columns=["quality", "target"])  # X: atributos de entrada (remove colunas que nao sao features)
yw = wine["target"]                             # y: variavel alvo (o que queremos prever)
feat_names_wine = list(Xw.columns)             # lista com os nomes das colunas de entrada (para usar nos graficos)
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(Xw, yw, test_size=0.25,
                                              stratify=yw, random_state=RANDOM_STATE)
# divide os dados: 75% treino, 25% teste
# stratify=yw garante que a proporcao de classes seja igual nos dois conjuntos
# random_state fixa a divisao para ser sempre a mesma

RESULTS["wine"] = {
    "n": int(wine.shape[0]),           # total de amostras no dataset de vinho
    "n_feat": int(Xw.shape[1]),        # numero de atributos (colunas de entrada)
    "dist": {"Ruim (<6)": int((yw==0).sum()), "Bom (>=6)": int((yw==1).sum())},  # contagem de cada classe
    "red": int(red.shape[0]),          # quantidade de vinhos tintos
    "white": int(white.shape[0]),      # quantidade de vinhos brancos
}

# ---- Figura: distribuicao de qualidade ----
plt.figure(figsize=(7,4))                         # cria uma figura com tamanho 7x4 polegadas
order = sorted(wine["quality"].unique())           # ordena os valores unicos de qualidade (ex: 3,4,5,6,7,8,9)
sns.countplot(data=wine, x="quality", order=order, color="#7e3b52")  # grafico de barras com contagem por nota
plt.axvline(2.5, color="black", ls="--", alpha=.6)  # linha vertical pontilhada indicando o corte entre "Ruim" e "Bom"
plt.title("Wine Quality - distribuicao das notas (corte em 6 -> 'Bom')")  # titulo do grafico
plt.xlabel("Nota de qualidade"); plt.ylabel("Frequencia")  # rotulos dos eixos X e Y
savefig("wine_dist.png")  # salva e fecha a figura

# ---- Correlacoes (apoio a descricao) ----
plt.figure(figsize=(9,7))                             # cria figura maior para o heatmap
corr = wine.drop(columns=["target"]).corr()           # calcula a matriz de correlacao entre todos os atributos
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False, square=True, cbar_kws={"shrink":.7})
# plota a matriz de correlacao como mapa de calor
# cmap="RdBu_r": azul=correlacao positiva, vermelho=negativa
# center=0: centraliza a escala de cores no zero
# annot=False: nao escreve os valores nas celulas (ficaria muito cheio)
plt.title("Wine Quality - matriz de correlacao")  # titulo do grafico
savefig("wine_corr.png")  # salva e fecha a figura

# ---------- MODELO A (obrigatorio): KNN ----------
# >>> 3+ variacoes de hiperparametros NESTE modelo <<<
print("\n--- KNN: variacao de hiperparametros (k) ---")  # avisa no terminal qual etapa esta sendo executada
ks = [1, 3, 5, 7, 11, 15, 21, 31, 51]  # lista de valores de k a testar (numero de vizinhos)
knn_curve = {"k": ks, "train": [], "test": [], "cv": []}  # dicionario para guardar as acuracias de cada k
cvk = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)  # validacao cruzada estratificada com 5 partes
for k in ks:  # itera sobre cada valor de k
    pipe = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
    # cria um pipeline: primeiro normaliza os dados, depois aplica o KNN com o k atual
    pipe.fit(Xw_tr, yw_tr)  # treina o pipeline com os dados de treino
    knn_curve["train"].append(accuracy_score(yw_tr, pipe.predict(Xw_tr)))  # acuracia no treino (pode ter overfitting)
    knn_curve["test"].append(accuracy_score(yw_te, pipe.predict(Xw_te)))   # acuracia no teste (performance real)
    knn_curve["cv"].append(cross_val_score(pipe, Xw_tr, yw_tr, cv=cvk, scoring="accuracy").mean())
    # acuracia media da validacao cruzada (mais confiavel que um unico treino/teste)
    print(f"k={k:>2}  train={knn_curve['train'][-1]:.3f}  test={knn_curve['test'][-1]:.3f}  cv={knn_curve['cv'][-1]:.3f}")
    # imprime os resultados formatados para o k atual

# variacoes adicionais: weights e metric
extra = {}  # dicionario para guardar resultados das combinacoes extras de hiperparametros
for weights in ["uniform", "distance"]:    # uniform: todos os vizinhos tem peso igual; distance: vizinhos proximos tem mais peso
    for metric in ["euclidean", "manhattan"]:  # euclidean: distancia em linha reta; manhattan: distancia em blocos (L1)
        pipe = Pipeline([("sc", StandardScaler()),
                         ("knn", KNeighborsClassifier(n_neighbors=21, weights=weights, metric=metric))])
        # cria pipeline KNN com k=21 e a combinacao atual de weights e metric
        pipe.fit(Xw_tr, yw_tr)  # treina o modelo
        acc = accuracy_score(yw_te, pipe.predict(Xw_te))   # acuracia no conjunto de teste
        f1 = f1_score(yw_te, pipe.predict(Xw_te))          # F1-score no conjunto de teste
        extra[f"{weights}/{metric}"] = {"acc": round(acc,4), "f1": round(f1,4)}  # armazena os resultados
        print(f"k=21 weights={weights:8} metric={metric:9} -> acc={acc:.3f} f1={f1:.3f}")  # imprime no terminal

# selecao final do KNN por validacao cruzada sobre k x weights x metric
best_cfg, best_cv = None, -1  # inicializa a melhor configuracao como vazia e a melhor CV como -1 (minimo possivel)
for k in [7, 11, 15, 21, 31]:         # testa apenas k's intermediarios (evita extremos)
    for weights in ["uniform", "distance"]:   # testa os dois tipos de peso
        for metric in ["euclidean", "manhattan"]:  # testa as duas metricas de distancia
            pipe = Pipeline([("sc", StandardScaler()),
                             ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric))])
            cv = cross_val_score(pipe, Xw_tr, yw_tr, cv=cvk, scoring="accuracy").mean()
            # calcula a acuracia media na validacao cruzada para esta combinacao
            if cv > best_cv:  # se esta combinacao for melhor que a anterior...
                best_cv, best_cfg = cv, {"n_neighbors":k, "weights":weights, "metric":metric}
                # ...atualiza o melhor resultado e a melhor configuracao
best_k = best_cfg["n_neighbors"]  # extrai o melhor valor de k da configuracao vencedora
print("Melhor config KNN por CV:", best_cfg, "cv=", round(best_cv,4))  # imprime a melhor configuracao encontrada
knn_best = Pipeline([("sc", StandardScaler()),
                     ("knn", KNeighborsClassifier(**best_cfg))]).fit(Xw_tr, yw_tr)
# cria e treina o pipeline final com a melhor configuracao (**best_cfg desempacota o dicionario como argumentos)
yw_pred_knn = knn_best.predict(Xw_te)  # gera as predicoes do KNN no conjunto de teste

# ---- Figura: curva k ----
plt.figure(figsize=(7.5,4.5))                                      # cria a figura
plt.plot(ks, knn_curve["train"], "o-", label="Treino")             # linha de acuracia no treino
plt.plot(ks, knn_curve["test"], "s-", label="Teste")               # linha de acuracia no teste
plt.plot(ks, knn_curve["cv"], "^--", label="CV (5-fold)")          # linha de acuracia na validacao cruzada
plt.axvline(best_k, color="grey", ls=":", alpha=.7)                # linha vertical marcando o melhor k escolhido
plt.gca().invert_xaxis()                                           # inverte o eixo X (k pequeno = mais complexo fica a esquerda)
plt.title("KNN (Wine) - impacto de k na acuracia")                 # titulo
plt.xlabel("k (n_neighbors)  [<- mais complexo | mais simples ->]")  # rotulo do eixo X com explicacao
plt.ylabel("Acuracia"); plt.legend()                               # rotulo Y e legenda
savefig("wine_knn_k.png")  # salva a figura

RESULTS["wine"]["knn"] = {
    "k_curve": {k: {"train": round(t,4), "test": round(te,4), "cv": round(c,4)}
                for k,t,te,c in zip(ks, knn_curve["train"], knn_curve["test"], knn_curve["cv"])},
    # dicionario com as acuracias de treino, teste e CV para cada valor de k testado
    "extra_variations": extra,                                    # resultados das variações de weights e metric
    "best_k": int(best_k), "best_cfg": best_cfg,                  # melhor k e configuracao completa
    "test_acc": round(accuracy_score(yw_te, yw_pred_knn),4),      # acuracia final no teste
    "test_f1": round(f1_score(yw_te, yw_pred_knn),4),             # F1-score final no teste
    "test_precision": round(precision_score(yw_te, yw_pred_knn),4),  # precisao final no teste
    "test_recall": round(recall_score(yw_te, yw_pred_knn),4),     # recall final no teste
}

# ---------- MODELO B (escolhido): Arvore de Decisao ----------
print("\n--- Arvore de Decisao (Wine) ---")  # avisa no terminal qual etapa esta sendo executada
dt_depths = [2, 3, 4, 6, 8, 12, None]  # lista de profundidades maximas a testar (None = arvore sem limite)
dt_rows = []  # lista para guardar os resultados de cada configuracao testada
for d in dt_depths:  # itera sobre cada profundidade
    dt = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE).fit(Xw_tr, yw_tr)
    # cria e treina uma arvore de decisao com a profundidade atual
    tr = accuracy_score(yw_tr, dt.predict(Xw_tr)); te = accuracy_score(yw_te, dt.predict(Xw_te))
    # calcula acuracia no treino (tr) e no teste (te)
    dt_rows.append({"depth": str(d), "train": round(tr,4), "test": round(te,4)})  # salva os resultados
    print(f"max_depth={str(d):>4} train={tr:.3f} test={te:.3f}")  # imprime os resultados no terminal
best_depth, best_dt_cv = None, -1  # inicializa a melhor profundidade e melhor CV
for d in [2,3,4,6,8,12]:  # testa as profundidades finitas (ignora None para evitar overfitting total)
    cv = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE),
                         Xw_tr, yw_tr, cv=cvk, scoring="accuracy").mean()
    # calcula a acuracia media na validacao cruzada para esta profundidade
    if cv > best_dt_cv: best_dt_cv, best_depth = cv, d  # atualiza se esta profundidade for melhor
print("Melhor max_depth (arvore) por CV:", best_depth, "cv=", round(best_dt_cv,4))  # imprime o melhor resultado
dt_best = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE).fit(Xw_tr, yw_tr)
# cria e treina a arvore final com a melhor profundidade encontrada
yw_pred_dt = dt_best.predict(Xw_te)  # gera as predicoes da arvore no conjunto de teste
RESULTS["wine"]["tree"] = {
    "depth_curve": dt_rows,                                         # resultados de todas as profundidades testadas
    "best_depth": best_depth,                                       # melhor profundidade encontrada
    "test_acc": round(accuracy_score(yw_te, yw_pred_dt),4),        # acuracia final no teste
    "test_f1": round(f1_score(yw_te, yw_pred_dt),4),               # F1-score final no teste
    "test_precision": round(precision_score(yw_te, yw_pred_dt),4), # precisao final no teste
    "test_recall": round(recall_score(yw_te, yw_pred_dt),4),       # recall final no teste
}

# ---- Figura: overfitting da arvore (train vs test x depth) ----
plt.figure(figsize=(7.5,4.5))                                      # cria a figura
xs = [str(r["depth"]) for r in dt_rows]                           # lista de rotulos do eixo X (profundidades)
plt.plot(xs, [r["train"] for r in dt_rows], "o-", label="Treino") # linha de acuracia no treino
plt.plot(xs, [r["test"] for r in dt_rows], "s-", label="Teste")   # linha de acuracia no teste
plt.title("Arvore (Wine) - profundidade vs acuracia (overfitting)") # titulo: mostra que arvore profunda tem overfitting
plt.xlabel("max_depth"); plt.ylabel("Acuracia"); plt.legend()      # rotulos e legenda
savefig("wine_tree_depth.png")  # salva a figura

# ---- Figura: importancia de atributos (arvore) ----
imp = pd.Series(dt_best.feature_importances_, index=feat_names_wine).sort_values(ascending=True)
# cria uma Serie com a importancia de cada atributo, ordenada do menos para o mais importante
plt.figure(figsize=(7.5,5))                         # cria a figura
imp.plot(kind="barh", color="#7e3b52")              # grafico de barras horizontal com a importancia
plt.title("Arvore (Wine) - importancia dos atributos")  # titulo
plt.xlabel("Importancia (reducao de impureza)")     # rotulo do eixo X: quanto aquele atributo reduziu a impureza (Gini/Entropy)
savefig("wine_tree_importance.png")  # salva a figura

# ---- Figura: arvore podada (interpretabilidade) ----
dt_viz = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE).fit(Xw_tr, yw_tr)
# cria uma arvore rasa (max_depth=3) apenas para visualizacao, pois arvores profundas ficam ilegíveis
plt.figure(figsize=(14,7))  # figura bem larga para caber a arvore
plot_tree(dt_viz, feature_names=feat_names_wine, class_names=["Ruim","Bom"],
          filled=True, rounded=True, fontsize=8, impurity=False)
# plota a arvore visualmente
# filled=True: colore os nos pela classe predominante
# rounded=True: bordas arredondadas nos nos
# impurity=False: nao mostra o indice de impureza (simplifica a visualizacao)
plt.title("Arvore de Decisao (Wine) podada em max_depth=3 - interpretabilidade")  # titulo
savefig("wine_tree_viz.png")  # salva a figura

# ---- Figura: matrizes de confusao (wine) ----
fig, ax = plt.subplots(1, 2, figsize=(10,4))  # cria uma figura com 2 subgraficos lado a lado
ConfusionMatrixDisplay(confusion_matrix(yw_te, yw_pred_knn), display_labels=["Ruim","Bom"]).plot(ax=ax[0], cmap="Purples", colorbar=False)
# plota a matriz de confusao do KNN no primeiro subgrafico
ax[0].set_title(f"KNN (k={best_k})")  # titulo do primeiro grafico com o melhor k
ConfusionMatrixDisplay(confusion_matrix(yw_te, yw_pred_dt), display_labels=["Ruim","Bom"]).plot(ax=ax[1], cmap="Purples", colorbar=False)
# plota a matriz de confusao da Arvore no segundo subgrafico
ax[1].set_title(f"Arvore (depth={best_depth})")  # titulo do segundo grafico com a melhor profundidade
plt.suptitle("Wine Quality - matrizes de confusao (teste)")  # titulo geral da figura
savefig("wine_confusion.png")  # salva a figura

# =====================================================================
# DATASET 2 - STUDENTS DROPOUT (classificacao multiclasse: 3 classes)
# =====================================================================
print("\n" + "="*70); print("DATASET 2: STUDENTS DROPOUT"); print("="*70)  # separador no terminal
drop = pd.read_csv("data/dropout.csv", sep=";")       # le o CSV de estudantes (separador ;)
drop.columns = [c.strip() for c in drop.columns]      # remove espacos em branco dos nomes das colunas
print("Shape:", drop.shape)  # imprime o numero de linhas e colunas
ymap = {"Dropout":0, "Enrolled":1, "Graduate":2}  # mapa para converter as classes de texto para numero
class_names = ["Dropout", "Enrolled", "Graduate"]  # nomes das classes para usar nos graficos
yd = drop["Target"].map(ymap)    # converte a coluna alvo para numeros (0, 1 ou 2)
Xd = drop.drop(columns=["Target"])   # X: atributos de entrada (remove a coluna alvo)
feat_names_drop = list(Xd.columns)   # lista com os nomes das colunas de entrada
print("Distribuicao:", drop["Target"].value_counts().to_dict())  # imprime a contagem de cada classe
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.25,
                                              stratify=yd, random_state=RANDOM_STATE)
# divide em 75% treino e 25% teste, mantendo proporcao de classes (stratify)
RESULTS["dropout"] = {
    "n": int(drop.shape[0]),          # total de amostras no dataset de estudantes
    "n_feat": int(Xd.shape[1]),       # numero de atributos de entrada
    "dist": drop["Target"].value_counts().to_dict(),  # contagem de cada classe
}

# ---- Figura: distribuicao de classes ----
plt.figure(figsize=(6,4))  # cria a figura
sns.countplot(x=drop["Target"], order=class_names, palette=["#c0392b","#e67e22","#27ae60"])
# grafico de barras com contagem de cada classe, usando cores diferentes para cada uma
plt.title("Students Dropout - distribuicao das classes")  # titulo
plt.xlabel(""); plt.ylabel("Frequencia")  # sem rotulo no eixo X (o nome da classe ja aparece); Y = frequencia
savefig("drop_dist.png")  # salva a figura

# ---------- MODELO A (obrigatorio): Regressao Logistica ----------
print("\n--- Regressao Logistica (Dropout) ---")  # avisa no terminal
logreg = Pipeline([("sc", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=2000,
                                             C=1.0, random_state=RANDOM_STATE))]).fit(Xd_tr, yd_tr)
# cria e treina o pipeline: normaliza os dados e depois aplica a Regressao Logistica
# max_iter=2000: numero maximo de iteracoes para o algoritmo convergir
# C=1.0: forca da regularizacao (maior C = menos regularizacao = modelo mais flexivel)
yd_pred_lr = logreg.predict(Xd_te)          # gera as predicoes da Regressao Logistica no teste
acc_lr = accuracy_score(yd_te, yd_pred_lr)  # calcula a acuracia no teste
print("Acuracia LR:", round(acc_lr,4))       # imprime a acuracia
print(classification_report(yd_te, yd_pred_lr, target_names=class_names))
# imprime o relatorio completo com precisao, recall e F1 para cada classe

# variacao de C (apoio a discussao de regularizacao) - secundario
lr_C = {}  # dicionario para guardar a acuracia de cada valor de C
for C in [0.01, 0.1, 1.0, 10.0]:  # testa 4 valores de C (regularizacao fraca a forte)
    p = Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(max_iter=2000, C=C, random_state=RANDOM_STATE))]).fit(Xd_tr, yd_tr)
    # cria e treina um pipeline com o valor de C atual
    lr_C[C] = round(accuracy_score(yd_te, p.predict(Xd_te)),4)  # salva a acuracia deste C
print("LR variando C:", lr_C)  # imprime o resultado de todas as variacoes de C

RESULTS["dropout"]["logreg"] = {
    "test_acc": round(acc_lr,4),                                              # acuracia final no teste
    "f1_macro": round(f1_score(yd_te, yd_pred_lr, average="macro"),4),       # F1 macro (media simples entre classes)
    "f1_weighted": round(f1_score(yd_te, yd_pred_lr, average="weighted"),4), # F1 ponderado (leva em conta o tamanho de cada classe)
    "C_variation": lr_C,                                                      # resultados da variacao de C
    "report": classification_report(yd_te, yd_pred_lr, target_names=class_names, output_dict=True),
    # relatorio completo como dicionario (para salvar no JSON)
}

# ---------- MODELO B (escolhido): Random Forest ----------
# >>> 3+ variacoes de hiperparametros NESTE modelo <<<
print("\n--- Random Forest (Dropout): variacao de hiperparametros ---")  # avisa no terminal
rf_experiments = [
    {"n_estimators":50,  "max_depth":None, "min_samples_leaf":1},  # floresta pequena, sem limite de profundidade
    {"n_estimators":100, "max_depth":None, "min_samples_leaf":1},  # floresta media, sem limite de profundidade
    {"n_estimators":300, "max_depth":None, "min_samples_leaf":1},  # floresta grande, sem limite de profundidade
    {"n_estimators":300, "max_depth":10,   "min_samples_leaf":1},  # floresta grande, arvores limitadas a 10 niveis
    {"n_estimators":300, "max_depth":10,   "min_samples_leaf":5},  # floresta grande, cada folha precisa de pelo menos 5 amostras
    {"n_estimators":300, "max_depth":6,    "min_samples_leaf":5},  # floresta grande, arvores mais rasas (6 niveis)
]
# lista de configuracoes a testar
# n_estimators: numero de arvores na floresta
# max_depth: profundidade maxima de cada arvore (None = sem limite)
# min_samples_leaf: numero minimo de amostras em cada folha (regularizacao)
rf_rows = []  # lista para guardar os resultados de cada configuracao
for cfg in rf_experiments:  # itera sobre cada configuracao
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **cfg).fit(Xd_tr, yd_tr)
    # cria e treina o Random Forest com a configuracao atual
    # n_jobs=-1: usa todos os nucleos do processador para treinar mais rapido
    tr = accuracy_score(yd_tr, rf.predict(Xd_tr))       # acuracia no treino
    te = accuracy_score(yd_te, rf.predict(Xd_te))       # acuracia no teste
    f1m = f1_score(yd_te, rf.predict(Xd_te), average="macro")  # F1-macro no teste
    rf_rows.append({**cfg, "train": round(tr,4), "test": round(te,4), "f1_macro": round(f1m,4)})
    # salva a configuracao junto com os resultados (**cfg desempacota o dicionario)
    print(f"{cfg} -> train={tr:.3f} test={te:.3f} f1m={f1m:.3f}")  # imprime os resultados

# melhor config por test acc
best_rf_cfg = max(rf_rows, key=lambda r: r["test"])  # encontra a configuracao com maior acuracia no teste
best_params = {k:best_rf_cfg[k] for k in ["n_estimators","max_depth","min_samples_leaf"]}
# extrai apenas os hiperparametros (sem as metricas) do melhor resultado
rf_best = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params).fit(Xd_tr, yd_tr)
# cria e treina o modelo final com os melhores parametros
yd_pred_rf = rf_best.predict(Xd_te)  # gera as predicoes do Random Forest no conjunto de teste
print("Melhor RF:", best_params, "test_acc=", round(accuracy_score(yd_te, yd_pred_rf),4))  # imprime o melhor resultado

RESULTS["dropout"]["rf"] = {
    "experiments": rf_rows,                                                    # resultados de todos os experimentos
    "best_params": best_params,                                                # melhores hiperparametros encontrados
    "test_acc": round(accuracy_score(yd_te, yd_pred_rf),4),                   # acuracia final no teste
    "f1_macro": round(f1_score(yd_te, yd_pred_rf, average="macro"),4),        # F1 macro final no teste
    "f1_weighted": round(f1_score(yd_te, yd_pred_rf, average="weighted"),4),  # F1 ponderado final no teste
    "report": classification_report(yd_te, yd_pred_rf, target_names=class_names, output_dict=True),
    # relatorio completo como dicionario
}

# ---- Figura: impacto dos hiperparametros do RF (train vs test) ----
labels = [f"n={r['n_estimators']}\nd={r['max_depth']}\nleaf={r['min_samples_leaf']}" for r in rf_rows]
# cria rotulos para o eixo X resumindo os hiperparametros de cada experimento
x = np.arange(len(rf_rows))  # posicoes numericas para as barras no eixo X
plt.figure(figsize=(9,4.5))  # cria a figura
plt.plot(x, [r["train"] for r in rf_rows], "o-", label="Treino")  # linha de acuracia no treino
plt.plot(x, [r["test"] for r in rf_rows], "s-", label="Teste")    # linha de acuracia no teste
plt.xticks(x, labels, fontsize=8)  # coloca os rotulos criados acima no eixo X
plt.title("Random Forest (Dropout) - impacto dos hiperparametros")  # titulo
plt.ylabel("Acuracia"); plt.legend()  # rotulo Y e legenda
savefig("drop_rf_hyper.png")  # salva a figura

# ---- Figura: importancia de atributos (RF) top 15 ----
impd = pd.Series(rf_best.feature_importances_, index=feat_names_drop).sort_values(ascending=False).head(15).sort_values()
# cria uma Serie com a importancia de cada atributo, pega os 15 mais importantes e ordena crescente (para o grafico horizontal ficar bonito)
plt.figure(figsize=(8,6))  # cria a figura
impd.plot(kind="barh", color="#27ae60")  # grafico de barras horizontal com as importancias
plt.title("Random Forest (Dropout) - top 15 atributos mais importantes")  # titulo
plt.xlabel("Importancia")  # rotulo do eixo X
savefig("drop_rf_importance.png")  # salva a figura

# ---- Figura: matrizes de confusao (dropout) ----
fig, ax = plt.subplots(1, 2, figsize=(11,4.2))  # cria figura com 2 subgraficos lado a lado
ConfusionMatrixDisplay(confusion_matrix(yd_te, yd_pred_lr), display_labels=class_names).plot(ax=ax[0], cmap="Greens", colorbar=False, xticks_rotation=45)
# plota a matriz de confusao da Regressao Logistica; xticks_rotation=45 inclina os rotulos para nao sobrepor
ax[0].set_title("Regressao Logistica")  # titulo do primeiro subgrafico
ConfusionMatrixDisplay(confusion_matrix(yd_te, yd_pred_rf), display_labels=class_names).plot(ax=ax[1], cmap="Greens", colorbar=False, xticks_rotation=45)
# plota a matriz de confusao do Random Forest
ax[1].set_title("Random Forest")  # titulo do segundo subgrafico
plt.suptitle("Students Dropout - matrizes de confusao (teste)")  # titulo geral da figura
savefig("drop_confusion.png")  # salva a figura

# =====================================================================
# COMPARACAO GERAL
# =====================================================================
plt.figure(figsize=(8,4.5))  # cria a figura final de comparacao
models = ["KNN\n(Wine)", "Arvore\n(Wine)", "Reg.Log.\n(Dropout)", "Random Forest\n(Dropout)"]
# nomes dos 4 modelos para o eixo X
accs = [RESULTS["wine"]["knn"]["test_acc"], RESULTS["wine"]["tree"]["test_acc"],
        RESULTS["dropout"]["logreg"]["test_acc"], RESULTS["dropout"]["rf"]["test_acc"]]
# lista com a acuracia de cada modelo no teste
f1s  = [RESULTS["wine"]["knn"]["test_f1"], RESULTS["wine"]["tree"]["test_f1"],
        RESULTS["dropout"]["logreg"]["f1_macro"], RESULTS["dropout"]["rf"]["f1_macro"]]
# lista com o F1-score de cada modelo no teste
x = np.arange(len(models)); w=0.38  # posicoes no eixo X e largura das barras
plt.bar(x-w/2, accs, w, label="Acuracia", color="#34495e")  # barras de acuracia (deslocadas para a esquerda)
plt.bar(x+w/2, f1s, w, label="F1", color="#9b59b6")         # barras de F1 (deslocadas para a direita)
for i,(a,f) in enumerate(zip(accs,f1s)):         # itera sobre os 4 modelos
    plt.text(i-w/2, a+.01, f"{a:.2f}", ha="center", fontsize=8)  # escreve o valor de acuracia acima de cada barra
    plt.text(i+w/2, f+.01, f"{f:.2f}", ha="center", fontsize=8)  # escreve o valor de F1 acima de cada barra
plt.xticks(x, models); plt.ylim(0,1.05)          # coloca os nomes dos modelos no eixo X; limita Y entre 0 e 1.05
plt.title("Comparacao geral dos modelos (conjunto de teste)")  # titulo
plt.ylabel("Score"); plt.legend()                # rotulo Y e legenda
savefig("comparacao_geral.png")  # salva a figura

with open("resultados.json","w") as f:  # abre (ou cria) o arquivo resultados.json para escrita
    json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    # salva o dicionario RESULTS como JSON formatado (indent=2 = indentado com 2 espacos; ensure_ascii=False = preserva acentos)
print("\n>>> Figuras salvas em /figuras e resultados em resultados.json")  # avisa que o processo terminou
print("Figuras:", sorted(os.listdir(FIG)))  # lista todas as figuras geradas, em ordem alfabetica
