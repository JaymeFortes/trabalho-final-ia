"""
Trabalho Final - Inteligencia Artificial (PUCRS)
Analise Critica de Modelos de IA
Pipeline de analise: Wine Quality + Students Dropout
Gera todas as metricas e figuras usadas no relatorio e na apresentacao.

Modelos:
  Wine Quality       -> KNN (obrigatorio) + Arvore de Decisao (escolhido)
  Students Dropout   -> Regressao Logistica (obrigatorio) + Arvore de Decisao (escolhido)
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
np.random.seed(42)
sns.set_theme(style="whitegrid")

# cria a proxima pasta results_N disponivel (results_1, results_2, ...)
_n = 1
while os.path.exists(f"results_{_n}"):
    _n += 1
FIG = f"results_{_n}"
os.makedirs(FIG)
print(f"Salvando figuras em: {FIG}/")

RESULTS = {}
RANDOM_STATE = 42

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{FIG}/{name}", dpi=130, bbox_inches="tight")
    plt.close()

# =====================================================================
# DATASET 1 - WINE QUALITY (classificacao binaria: qualidade >= 6 -> "Bom")
# =====================================================================
print("="*70)
print("DATASET 1: WINE QUALITY")
print("="*70)

red   = pd.read_csv("data/winequality-red.csv",   sep=";")
white = pd.read_csv("data/winequality-white.csv",  sep=";")
red["type"]   = 0   # tinto
white["type"] = 1   # branco
wine = pd.concat([red, white], ignore_index=True)
wine["target"] = (wine["quality"] >= 6).astype(int)   # 1 = Bom, 0 = Ruim
print("Shape:", wine.shape)
print("Distribuicao target (0=Ruim, 1=Bom):", wine["target"].value_counts().to_dict())

Xw = wine.drop(columns=["quality", "target"])
yw = wine["target"]
feat_names_wine = list(Xw.columns)
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(
    Xw, yw, test_size=0.25, stratify=yw, random_state=RANDOM_STATE)

RESULTS["wine"] = {
    "n": int(wine.shape[0]), "n_feat": int(Xw.shape[1]),
    "dist": {"Ruim (<6)": int((yw==0).sum()), "Bom (>=6)": int((yw==1).sum())},
    "red": int(red.shape[0]), "white": int(white.shape[0]),
}

# ---- Figura: distribuicao de qualidade ----
plt.figure(figsize=(7,4))
order = sorted(wine["quality"].unique())
sns.countplot(data=wine, x="quality", order=order, color="#7e3b52")
plt.axvline(2.5, color="black", ls="--", alpha=.6)
plt.title("Wine Quality - distribuicao das notas (corte em 6 -> 'Bom')")
plt.xlabel("Nota de qualidade"); plt.ylabel("Frequencia")
savefig("wine_dist.png")

# ---- Correlacoes (apoio a descricao) ----
plt.figure(figsize=(9,7))
corr = wine.drop(columns=["target"]).corr()
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False, square=True, cbar_kws={"shrink":.7})
plt.title("Wine Quality - matriz de correlacao")
savefig("wine_corr.png")

# ---------- MODELO A (obrigatorio): KNN ----------
# >>> 3+ variacoes de hiperparametros NESTE modelo <<<
print("\n--- KNN: variacao de hiperparametros (k) ---")
ks = [1, 3, 5, 7, 11, 15, 21, 31, 51]
knn_curve = {"k": ks, "train": [], "test": [], "cv": []}
cvk = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for k in ks:
    pipe = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
    pipe.fit(Xw_tr, yw_tr)
    knn_curve["train"].append(accuracy_score(yw_tr, pipe.predict(Xw_tr)))
    knn_curve["test"].append(accuracy_score(yw_te, pipe.predict(Xw_te)))
    knn_curve["cv"].append(cross_val_score(pipe, Xw_tr, yw_tr, cv=cvk, scoring="accuracy").mean())
    print(f"k={k:>2}  train={knn_curve['train'][-1]:.3f}  "
          f"test={knn_curve['test'][-1]:.3f}  cv={knn_curve['cv'][-1]:.3f}")

# variacoes adicionais: weights e metric
extra = {}
for weights in ["uniform", "distance"]:
    for metric in ["euclidean", "manhattan"]:
        pipe = Pipeline([("sc", StandardScaler()),
                         ("knn", KNeighborsClassifier(n_neighbors=21, weights=weights, metric=metric))])
        pipe.fit(Xw_tr, yw_tr)
        acc = accuracy_score(yw_te, pipe.predict(Xw_te))
        f1  = f1_score(yw_te, pipe.predict(Xw_te))
        extra[f"{weights}/{metric}"] = {"acc": round(acc,4), "f1": round(f1,4)}
        print(f"k=21 weights={weights:8} metric={metric:9} -> acc={acc:.3f} f1={f1:.3f}")

# selecao final do KNN por validacao cruzada
best_cfg, best_cv = None, -1
for k in [7, 11, 15, 21, 31]:
    for weights in ["uniform", "distance"]:
        for metric in ["euclidean", "manhattan"]:
            pipe = Pipeline([("sc", StandardScaler()),
                             ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric))])
            cv = cross_val_score(pipe, Xw_tr, yw_tr, cv=cvk, scoring="accuracy").mean()
            if cv > best_cv:
                best_cv, best_cfg = cv, {"n_neighbors":k, "weights":weights, "metric":metric}
best_k = best_cfg["n_neighbors"]
print("Melhor config KNN por CV:", best_cfg, "cv=", round(best_cv,4))
knn_best = Pipeline([("sc", StandardScaler()),
                     ("knn", KNeighborsClassifier(**best_cfg))]).fit(Xw_tr, yw_tr)
yw_pred_knn = knn_best.predict(Xw_te)

# ---- Figura: curva k ----
plt.figure(figsize=(7.5,4.5))
plt.plot(ks, knn_curve["train"], "o-", label="Treino")
plt.plot(ks, knn_curve["test"],  "s-", label="Teste")
plt.plot(ks, knn_curve["cv"],    "^--", label="CV (5-fold)")
plt.axvline(best_k, color="grey", ls=":", alpha=.7)
plt.gca().invert_xaxis()
plt.title("KNN (Wine) - impacto de k na acuracia")
plt.xlabel("k (n_neighbors)  [<- mais complexo | mais simples ->]")
plt.ylabel("Acuracia"); plt.legend()
savefig("wine_knn_k.png")

RESULTS["wine"]["knn"] = {
    "k_curve": {k: {"train": round(t,4), "test": round(te,4), "cv": round(c,4)}
                for k,t,te,c in zip(ks, knn_curve["train"], knn_curve["test"], knn_curve["cv"])},
    "extra_variations": extra, "best_k": int(best_k), "best_cfg": best_cfg,
    "test_acc":       round(accuracy_score(yw_te, yw_pred_knn),4),
    "test_f1":        round(f1_score(yw_te, yw_pred_knn),4),
    "test_precision": round(precision_score(yw_te, yw_pred_knn),4),
    "test_recall":    round(recall_score(yw_te, yw_pred_knn),4),
}

# ---------- MODELO B (escolhido): Arvore de Decisao ----------
print("\n--- Arvore de Decisao (Wine) ---")
dt_depths = [2, 3, 4, 6, 8, 12, None]
dt_rows_wine = []
for d in dt_depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE).fit(Xw_tr, yw_tr)
    tr = accuracy_score(yw_tr, dt.predict(Xw_tr))
    te = accuracy_score(yw_te, dt.predict(Xw_te))
    dt_rows_wine.append({"depth": str(d), "train": round(tr,4), "test": round(te,4)})
    print(f"max_depth={str(d):>4} train={tr:.3f} test={te:.3f}")

best_depth_wine, best_dt_cv_wine = None, -1
for d in [2,3,4,6,8,12]:
    cv = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE),
                         Xw_tr, yw_tr, cv=cvk, scoring="accuracy").mean()
    if cv > best_dt_cv_wine:
        best_dt_cv_wine, best_depth_wine = cv, d
print("Melhor max_depth (arvore Wine) por CV:", best_depth_wine, "cv=", round(best_dt_cv_wine,4))
dt_best_wine = DecisionTreeClassifier(max_depth=best_depth_wine, random_state=RANDOM_STATE).fit(Xw_tr, yw_tr)
yw_pred_dt = dt_best_wine.predict(Xw_te)

RESULTS["wine"]["tree"] = {
    "depth_curve": dt_rows_wine, "best_depth": best_depth_wine,
    "test_acc":       round(accuracy_score(yw_te, yw_pred_dt),4),
    "test_f1":        round(f1_score(yw_te, yw_pred_dt),4),
    "test_precision": round(precision_score(yw_te, yw_pred_dt),4),
    "test_recall":    round(recall_score(yw_te, yw_pred_dt),4),
}

# ---- Figura: overfitting da arvore wine (train vs test x depth) ----
plt.figure(figsize=(7.5,4.5))
xs = [r["depth"] for r in dt_rows_wine]
plt.plot(xs, [r["train"] for r in dt_rows_wine], "o-", label="Treino")
plt.plot(xs, [r["test"]  for r in dt_rows_wine], "s-", label="Teste")
plt.title("Arvore (Wine) - profundidade vs acuracia")
plt.xlabel("max_depth"); plt.ylabel("Acuracia"); plt.legend()
savefig("wine_tree_depth.png")

# ---- Figura: importancia de atributos (arvore wine) ----
imp_wine = pd.Series(dt_best_wine.feature_importances_, index=feat_names_wine).sort_values(ascending=True)
plt.figure(figsize=(7.5,5))
imp_wine.plot(kind="barh", color="#7e3b52")
plt.title("Arvore (Wine) - importancia dos atributos")
plt.xlabel("Importancia (reducao de impureza)")
savefig("wine_tree_importance.png")

# ---- Figura: arvore podada para visualizacao (interpretabilidade) ----
dt_viz = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE).fit(Xw_tr, yw_tr)
plt.figure(figsize=(14,7))
plot_tree(dt_viz, feature_names=feat_names_wine, class_names=["Ruim","Bom"],
          filled=True, rounded=True, fontsize=8, impurity=False)
plt.title("Arvore de Decisao (Wine) podada em max_depth=3 - interpretabilidade")
savefig("wine_tree_viz.png")

# ---- Figura: matrizes de confusao (wine) ----
fig, ax = plt.subplots(1, 2, figsize=(10,4))
ConfusionMatrixDisplay(confusion_matrix(yw_te, yw_pred_knn),
                       display_labels=["Ruim","Bom"]).plot(ax=ax[0], cmap="Purples", colorbar=False)
ax[0].set_title(f"KNN (k={best_k}, distance, manhattan)")
ConfusionMatrixDisplay(confusion_matrix(yw_te, yw_pred_dt),
                       display_labels=["Ruim","Bom"]).plot(ax=ax[1], cmap="Purples", colorbar=False)
ax[1].set_title(f"Arvore (depth={best_depth_wine})")
plt.suptitle("Wine Quality - matrizes de confusao (teste)")
savefig("wine_confusion.png")

# =====================================================================
# DATASET 2 - STUDENTS DROPOUT (classificacao multiclasse: 3 classes)
# =====================================================================
print("\n" + "="*70)
print("DATASET 2: STUDENTS DROPOUT")
print("="*70)

drop = pd.read_csv("data/students.csv", sep=";")  # arquivo renomeado para students.csv
drop.columns = [c.strip() for c in drop.columns]
print("Shape:", drop.shape)

ymap = {"Dropout":0, "Enrolled":1, "Graduate":2}
class_names = ["Dropout", "Enrolled", "Graduate"]
yd = drop["Target"].map(ymap)
Xd = drop.drop(columns=["Target"])
feat_names_drop = list(Xd.columns)
print("Distribuicao:", drop["Target"].value_counts().to_dict())

Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
    Xd, yd, test_size=0.25, stratify=yd, random_state=RANDOM_STATE)

RESULTS["dropout"] = {
    "n": int(drop.shape[0]), "n_feat": int(Xd.shape[1]),
    "dist": drop["Target"].value_counts().to_dict(),
}

# ---- Figura: distribuicao de classes ----
plt.figure(figsize=(6,4))
sns.countplot(x=drop["Target"], order=class_names,
              palette=["#c0392b","#e67e22","#27ae60"])
plt.title("Students Dropout - distribuicao das classes")
plt.xlabel(""); plt.ylabel("Frequencia")
savefig("drop_dist.png")

# ---------- MODELO A (obrigatorio): Regressao Logistica ----------
print("\n--- Regressao Logistica (Dropout) ---")
logreg = Pipeline([("sc", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=2000,
                                             C=1.0, random_state=RANDOM_STATE))]).fit(Xd_tr, yd_tr)
yd_pred_lr = logreg.predict(Xd_te)
acc_lr = accuracy_score(yd_te, yd_pred_lr)
print("Acuracia LR:", round(acc_lr,4))
print(classification_report(yd_te, yd_pred_lr, target_names=class_names))

# variacao de C (regularizacao da Regressao Logistica)
lr_C = {}
for C in [0.01, 0.1, 1.0, 10.0]:
    p = Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(max_iter=2000, C=C, random_state=RANDOM_STATE))]).fit(Xd_tr, yd_tr)
    lr_C[C] = round(accuracy_score(yd_te, p.predict(Xd_te)),4)
print("LR variando C:", lr_C)

RESULTS["dropout"]["logreg"] = {
    "test_acc":    round(acc_lr,4),
    "f1_macro":    round(f1_score(yd_te, yd_pred_lr, average="macro"),4),
    "f1_weighted": round(f1_score(yd_te, yd_pred_lr, average="weighted"),4),
    "C_variation": lr_C,
    "report": classification_report(yd_te, yd_pred_lr, target_names=class_names, output_dict=True),
}

# ---- Figura: variacao de C na Regressao Logistica ----
plt.figure(figsize=(6,4))
plt.plot(list(lr_C.keys()), list(lr_C.values()), "o-", color="#2980b9")
plt.xscale("log")
plt.title("Regressao Logistica (Dropout) - impacto do parametro C")
plt.xlabel("C (regularizacao)"); plt.ylabel("Acuracia (teste)")
savefig("drop_lr_C.png")

# ---------- MODELO B (escolhido): Arvore de Decisao ----------
# >>> 3 hiperparametros variados: max_depth, min_samples_leaf, criterion <<<
print("\n--- Arvore de Decisao (Dropout): variacao de hiperparametros ---")

dt_experiments_drop = [
    # variacao 1: max_depth (profundidade maxima da arvore)
    {"max_depth": 2,    "min_samples_leaf": 1,  "criterion": "gini"},
    {"max_depth": 4,    "min_samples_leaf": 1,  "criterion": "gini"},
    {"max_depth": 6,    "min_samples_leaf": 1,  "criterion": "gini"},
    {"max_depth": 10,   "min_samples_leaf": 1,  "criterion": "gini"},
    {"max_depth": 15,   "min_samples_leaf": 1,  "criterion": "gini"},
    {"max_depth": None, "min_samples_leaf": 1,  "criterion": "gini"},  # sem limite -> overfitting esperado
    # variacao 2: min_samples_leaf (minimo de amostras por folha = regularizacao)
    {"max_depth": 10,   "min_samples_leaf": 5,  "criterion": "gini"},
    {"max_depth": 10,   "min_samples_leaf": 10, "criterion": "gini"},
    # variacao 3: criterion (metrica de impureza usada para decidir os cortes)
    {"max_depth": 10,   "min_samples_leaf": 1,  "criterion": "entropy"},  # entropy vs gini
]

dt_rows_drop = []
for cfg in dt_experiments_drop:
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE, **cfg).fit(Xd_tr, yd_tr)
    tr  = accuracy_score(yd_tr, dt.predict(Xd_tr))
    te  = accuracy_score(yd_te, dt.predict(Xd_te))
    f1m = f1_score(yd_te, dt.predict(Xd_te), average="macro")
    dt_rows_drop.append({**cfg, "train": round(tr,4), "test": round(te,4), "f1_macro": round(f1m,4)})
    print(f"{cfg} -> train={tr:.3f} test={te:.3f} f1m={f1m:.3f}")

# escolhe a melhor combinacao de hiperparametros usando validacao cruzada
best_dt_drop_cfg, best_dt_drop_cv = None, -1
cvi = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for cfg in dt_experiments_drop:
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE, **cfg)
    cv = cross_val_score(dt, Xd_tr, yd_tr, cv=cvi, scoring="accuracy").mean()
    if cv > best_dt_drop_cv:
        best_dt_drop_cv, best_dt_drop_cfg = cv, cfg

print("Melhor Arvore Dropout por CV:", best_dt_drop_cfg, "cv=", round(best_dt_drop_cv,4))
dt_best_drop = DecisionTreeClassifier(random_state=RANDOM_STATE, **best_dt_drop_cfg).fit(Xd_tr, yd_tr)
yd_pred_dt = dt_best_drop.predict(Xd_te)
print("Acuracia teste:", round(accuracy_score(yd_te, yd_pred_dt),4))
print(classification_report(yd_te, yd_pred_dt, target_names=class_names))

RESULTS["dropout"]["tree"] = {
    "experiments":  dt_rows_drop,
    "best_cfg":     best_dt_drop_cfg,
    "test_acc":     round(accuracy_score(yd_te, yd_pred_dt),4),
    "f1_macro":     round(f1_score(yd_te, yd_pred_dt, average="macro"),4),
    "f1_weighted":  round(f1_score(yd_te, yd_pred_dt, average="weighted"),4),
    "report": classification_report(yd_te, yd_pred_dt, target_names=class_names, output_dict=True),
}

# ---- Figura: impacto da profundidade (train vs test) ----
# filtra apenas as linhas onde so variou max_depth (leaf=1, gini fixos)
depth_only = [r for r in dt_rows_drop if r["min_samples_leaf"]==1 and r["criterion"]=="gini"]
xs = [str(r["max_depth"]) for r in depth_only]
plt.figure(figsize=(8,4.5))
plt.plot(xs, [r["train"] for r in depth_only], "o-", label="Treino")
plt.plot(xs, [r["test"]  for r in depth_only], "s-", label="Teste")
plt.title("Arvore de Decisao (Dropout) - profundidade vs acuracia")
plt.xlabel("max_depth"); plt.ylabel("Acuracia"); plt.legend()
savefig("drop_tree_depth.png")

# ---- Figura: impacto do min_samples_leaf (depth=10, gini) ----
# filtra apenas as linhas onde so variou min_samples_leaf (depth=10, gini fixos)
leaf_only = [r for r in dt_rows_drop if r["max_depth"]==10 and r["criterion"]=="gini"]
xs_leaf = [str(r["min_samples_leaf"]) for r in leaf_only]
plt.figure(figsize=(5,4))
plt.bar(xs_leaf, [r["test"] for r in leaf_only], color=["#2980b9","#27ae60","#e67e22"])
plt.title("Arvore (Dropout) - min_samples_leaf vs acuracia (depth=10)")
plt.xlabel("min_samples_leaf"); plt.ylabel("Acuracia (teste)")
for i,r in enumerate(leaf_only):
    plt.text(i, r["test"]+0.002, f"{r['test']:.3f}", ha="center", fontsize=9)
savefig("drop_tree_leaf.png")

# ---- Figura: criterion (gini vs entropy, depth=10, leaf=1) ----
# filtra apenas as linhas onde so variou o criterio de impureza (depth=10, leaf=1 fixos)
crit_only = [r for r in dt_rows_drop if r["max_depth"]==10 and r["min_samples_leaf"]==1]
plt.figure(figsize=(5,4))
plt.bar([r["criterion"] for r in crit_only], [r["test"] for r in crit_only],
        color=["#8e44ad","#16a085"])
plt.title("Arvore (Dropout) - criterio de impureza vs acuracia (depth=10)")
plt.xlabel("Criterio"); plt.ylabel("Acuracia (teste)")
for i,r in enumerate(crit_only):
    plt.text(i, r["test"]+0.002, f"{r['test']:.3f}", ha="center", fontsize=9)
savefig("drop_tree_criterion.png")

# ---- Figura: importancia dos atributos (arvore dropout) top 15 ----
imp_drop = pd.Series(dt_best_drop.feature_importances_,
                     index=feat_names_drop).sort_values(ascending=False).head(15).sort_values()
plt.figure(figsize=(8,6))
imp_drop.plot(kind="barh", color="#27ae60")
plt.title("Arvore de Decisao (Dropout) - top 15 atributos mais importantes")
plt.xlabel("Importancia (reducao de impureza)")
savefig("drop_tree_importance.png")

# ---- Figura: arvore podada para visualizacao (interpretabilidade) ----
dt_viz_drop = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE).fit(Xd_tr, yd_tr)
plt.figure(figsize=(16,7))
plot_tree(dt_viz_drop, feature_names=feat_names_drop, class_names=class_names,
          filled=True, rounded=True, fontsize=7, impurity=False)
plt.title("Arvore de Decisao (Dropout) podada em max_depth=3 - interpretabilidade")
savefig("drop_tree_viz.png")

# ---- Figura: matrizes de confusao (dropout) ----
fig, ax = plt.subplots(1, 2, figsize=(11,4.2))
ConfusionMatrixDisplay(confusion_matrix(yd_te, yd_pred_lr),
                       display_labels=class_names).plot(ax=ax[0], cmap="Greens",
                                                        colorbar=False, xticks_rotation=45)
ax[0].set_title("Regressao Logistica")
ConfusionMatrixDisplay(confusion_matrix(yd_te, yd_pred_dt),
                       display_labels=class_names).plot(ax=ax[1], cmap="Oranges",
                                                        colorbar=False, xticks_rotation=45)
ax[1].set_title(f"Arvore de Decisao (depth={best_dt_drop_cfg['max_depth']})")
plt.suptitle("Students Dropout - matrizes de confusao (teste)")
savefig("drop_confusion.png")

# =====================================================================
# COMPARACAO GERAL
# =====================================================================
print("\n" + "="*70)
print("COMPARACAO GERAL DOS MODELOS")
print("="*70)

models = ["KNN\n(Wine)", "Arvore\n(Wine)", "Reg.Log.\n(Dropout)", "Arvore\n(Dropout)"]
accs = [
    RESULTS["wine"]["knn"]["test_acc"],
    RESULTS["wine"]["tree"]["test_acc"],
    RESULTS["dropout"]["logreg"]["test_acc"],
    RESULTS["dropout"]["tree"]["test_acc"],
]
f1s = [
    RESULTS["wine"]["knn"]["test_f1"],
    RESULTS["wine"]["tree"]["test_f1"],
    RESULTS["dropout"]["logreg"]["f1_macro"],
    RESULTS["dropout"]["tree"]["f1_macro"],
]

for m, a, f in zip(models, accs, f1s):
    print(f"{m.replace(chr(10),' '):25} acc={a:.4f}  f1={f:.4f}")

plt.figure(figsize=(8,4.5))
x = np.arange(len(models)); w = 0.38
plt.bar(x-w/2, accs, w, label="Acuracia", color="#34495e")
plt.bar(x+w/2, f1s,  w, label="F1",       color="#9b59b6")
for i,(a,f) in enumerate(zip(accs,f1s)):
    plt.text(i-w/2, a+.005, f"{a:.2f}", ha="center", fontsize=8)
    plt.text(i+w/2, f+.005, f"{f:.2f}", ha="center", fontsize=8)
plt.xticks(x, models); plt.ylim(0, 1.05)
plt.title("Comparacao geral dos modelos (conjunto de teste)")
plt.ylabel("Score"); plt.legend()
savefig("comparacao_geral.png")

# =====================================================================
# SALVAR RESULTADOS
# =====================================================================
with open(f"{FIG}/resultados.json","w") as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False)

print(f"\n>>> Figuras e resultados salvos em {FIG}/")
print("Figuras:", sorted(os.listdir(FIG)))