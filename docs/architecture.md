# Architecture cible

```mermaid
flowchart LR
    C[Client] -->|POST /predict| A[FastAPI]
    A --> V[Validation Pydantic]
    V --> P[Preprocessing StandardScaler]
    P --> M[Logistic Regression]
    M --> Q[Postprocessing label + confiance]
    Q --> J[Réponse JSON]
```

## Baseline

- modèle : pipeline `StandardScaler` + `LogisticRegression` sur Iris ;
- service : un processus Uvicorn, endpoint synchrone `/predict` ;
- protocole : mêmes entrées, mêmes tailles de batch, répétitions identiques ;
- mesures : médiane, p95, throughput, erreurs et overhead API.

## Variante à tester

Comparer la requête unitaire avec des batchs de 4 et 16. Une extension possible est d'ajouter un mode de prédiction préchargé ou plusieurs workers Uvicorn, en gardant la machine et le protocole constants.
