# Trabalho Final — Inteligência Artificial (PUCRS)

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

| Arquivo | Link |
|---|---|
| `winequality-red.csv` | https://archive.ics.uci.edu/dataset/186/wine+quality |
| `winequality-white.csv` | mesmo link acima (o pacote contém os dois) |
| `dropout.csv` | https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |

> Os arquivos devem usar `;` como separador — é o formato padrão do UCI. Se o arquivo vier com outro nome, renomeie para `dropout.csv`.

Após baixar:

```
data/
├── winequality-red.csv
├── winequality-white.csv
└── dropout.csv
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

| Saída | Conteúdo |
|---|---|
| `figuras/` | imagens `.png` usadas no relatório e na apresentação |
| `resultados.json` | métricas completas (acurácia, F1, precisão, recall, variações de hiperparâmetros) |

Ambos são criados automaticamente — não é necessário criar nada manualmente.

---

## Problemas comuns

| Erro | Causa / Solução |
|---|---|
| `FileNotFoundError: data/winequality-red.csv` | Os CSVs não estão na pasta `data/` |
| Colunas estranhas ou erro de leitura | Confirme que os CSVs usam `;` como separador |
| `pip: command not found` | Use `python -m pip` (ver Passo 2) |
