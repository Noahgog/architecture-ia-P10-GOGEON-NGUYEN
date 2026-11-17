# P10 - IA as a Service avec FastAPI

Squelette reproductible pour étudier le passage d'un modèle léger à une API d'inférence.

## Architecture

```mermaid
flowchart LR
    C[Client / benchmark] --> A[FastAPI /predict]
    A --> P[Validation + preprocessing]
    P --> M[Logistic Regression]
    M --> Q[Postprocessing]
    Q --> R[JSON response]
```

## Démarrage rapide

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Puis ouvrir `http://127.0.0.1:8000/docs`.

Les métriques cumulées du processus sont disponibles sur `GET /metrics` :

- nombre de requêtes `/predict` ;
- nombre d'erreurs de prédiction ;
- latence totale et moyenne du modèle.

## Exemple de requête

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"instances":[[5.1,3.5,1.4,0.2]]}'
```

## Benchmark

Avec l'API démarrée dans un terminal :

```powershell
python scripts/benchmark.py --url http://127.0.0.1:8000/predict --requests 200 --batch-sizes 1,4,16
```

Le script mesure :

- latence modèle seul ;
- latence HTTP bout-en-bout ;
- overhead du service ;
- throughput ;
- taux d'erreur ;
- CPU et mémoire du processus de benchmark.

Les résultats sont écrits dans `results/benchmark_*.csv`. Le rapport à compléter est dans `docs/rapport.md`.

## Reproductibilité

Les mesures doivent préciser la machine, le système, Python, les versions des dépendances, la taille de batch et le nombre de répétitions. Le script capture automatiquement une partie de ces informations dans `results/`.

## Structure

```text
app/
  main.py          API FastAPI
  model.py         modèle et preprocessing
scripts/
  benchmark.py     charge et comparaison modèle/API
docs/
  architecture.md  schéma et choix
  rapport.md       trame du rapport final
results/           CSV générés localement
```

## Limites du squelette

Le modèle Iris est volontairement petit et sert de baseline pédagogique. Les valeurs de benchmark sont à produire sur votre machine ; elles ne doivent pas être inventées ni comparées entre machines différentes.
