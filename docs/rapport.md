# Rapport - P10 IA as a Service avec FastAPI

## Sujet retenu

Prédire la victoire de l'équipe 1 avant un match classé de Brawl Stars à partir
du mode, de la carte et des agrégats d'Elo et de niveau de pouvoir des deux
équipes. Le dataset source est
[EliF77/brawlstars-ranked](https://huggingface.co/datasets/EliF77/brawlstars-ranked).

La base utilise un échantillon CSV local pour démarrer rapidement. Les résultats
de performance doivent être produits avec un échantillon préparé par
`scripts/prepare_brawl_data.py`, et non avec l'échantillon pédagogique versionné.

## 1. Cahier des charges

Décrire l'API, le format d'entrée/sortie, les contraintes de latence, throughput, mémoire et reproductibilité.

## 2. Protocole expérimental

| Élément | Valeur |
|---|---|
| Machine / CPU | À compléter |
| GPU | À compléter |
| OS | À compléter |
| Python / dépendances | Voir `requirements.txt` |
| Nombre de requêtes | À compléter |
| Batch sizes | 1, 4, 16 |
| Précision | float64 dans la baseline |

## 3. Résultats

Copier ici le tableau produit par `scripts/benchmark.py` et ajouter un graphique latence/throughput.

## 4. Analyse de l'overhead

Calculer `latence API - latence modèle` et commenter la part relative du service.

## 5. Goulot d'étranglement et optimisation

Identifier le goulot mesuré. Comparer au moins une variante avec le même protocole.

## 6. Déploiement et supervision

Décrire l'image Docker ou le processus Uvicorn, les health checks, logs, métriques de latence p50/p95/p99, throughput, taux d'erreur, CPU, mémoire et dérive des entrées.

## 7. Conclusion d'ingénierie

- principal goulot :
- compromis accuracy / latence / mémoire / throughput :
- architecture retenue et justification :
- prochaine amélioration :
