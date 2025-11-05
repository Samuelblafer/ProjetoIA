# ProjetoIA

Este repositório contém um script para treinar uma CNN simples com validação K-Fold ou hold-out (20/80, 30/70).

Arquivos principais
- `train_cnn_kfold.py`: script principal para treinar e avaliar o modelo. Gera gráficos em `results/`.
- `dataset/`: pasta esperada com subpastas por classe (ex.: `apartamento`, `sobrado`, `terreo`).
- `.vscode/settings.json`: aponta o interpretador Python usado no workspace.
- `requirements.txt`: dependências do projeto.

Como usar
1. Instale dependências (recomendo criar um venv):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Ajuste `dataset_path` em `train_cnn_kfold.py` se necessário e escolha K-Fold ou hold-out editando `USE_KFOLD` e `VALIDATION_SPLIT`.

3. Rode o script:

```powershell
python train_cnn_kfold.py
```

4. Os gráficos gerados aparecerão em `results/`.

Como subir para o GitHub
- Inicialize o repositório local e faça commit. Use os comandos (veja abaixo) e crie um repositório no GitHub (página web) para pegar a URL remota.

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<seu-usuario>/<seu-repo>.git
git push -u origin main
```

Substitua `<seu-usuario>` e `<seu-repo>` pela sua conta e nome do repositório.
