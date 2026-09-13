# Séquences de soins — Deep Learning

Projet 2 du portfolio : transformer un historique d’événements en entrée d’un petit Transformer et prédire le **type du prochain événement**.

**Toutes les données sont synthétiques.** Elles suivent des règles inventées pour rendre l’apprentissage observable. Ce projet ne reproduit ni les parcours individuels du SNDS ni un modèle clinique validé.

Ouvrir [le rapport visuel](reports/rapport.html), [les résultats](reports/RESULTATS.md).

## Pourquoi ne pas réutiliser Open DAMIR ?

Le projet 1 utilise des données réelles agrégées de remboursements. Il n’y a pas d’identifiant patient permettant d’ordonner les actes d’une même personne. Fabriquer un identifiant à partir des catégories publiques ne reconstituerait pas un patient. Ce projet utilise donc un générateur indépendant et conserve uniquement la cohérence thématique du portfolio.

## Ce qui est livré

- 3 000 patients fictifs, avec des dates et 14 catégories d’événements.
- Événements stockés en Parquet et fenêtres de contexte en tableaux NumPy.
- Séparation **par patient** : 2 100 train, 450 validation, 450 test.
- Trois références statistiques : fréquence, Markov d’ordre 1 et d’ordre 2.
- Transformer PyTorch : embeddings 32 dimensions, positions apprises, deux couches, quatre têtes.
- Trois initialisations : graines 7, 17 et 27 ; sélection sur la perte de validation.
- Exactitude, top 3, F1 macro, perte logarithmique, matrice de confusion et bootstrap par patient.
- Checkpoints, commande de prédiction, tests, figures exportables et documentation.

Le nombre exact d’événements et les scores exécutés figurent dans `reports/results.json`. Les modèles apprennent seulement les types d’événements passés, sans âge, coût, identifiant patient ni date de l’événement cible.

## Installation

Validation locale sous Windows, Python 3.12, PyTorch CPU. Un GPU n’est pas nécessaire pour la configuration fournie.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m careseq.cli run
```

Sous Linux, utiliser `.venv/bin/python` après `python3 -m venv .venv`. Les versions exactes validées sont dans `requirements-lock.txt` ; la ligne PyTorch CPU peut nécessiter l’index officiel ci-dessus. Le fichier `pyproject.toml` fournit des plages compatibles, tandis que le verrou décrit l’environnement exécuté.

Si la construction éditable bloque, installer `setuptools` et `wheel` puis utiliser `pip install --no-build-isolation -e ".[dev]"`.

## Commandes

Depuis la racine du projet :

```bash
python -m careseq.cli run
python -m careseq.cli report
python -m careseq.cli predict --events GENERALISTE BIOLOGIE CARDIOLOGIE PHARMACIE GENERALISTE BIOLOGIE
python -m pytest -q
```

Pour exécuter depuis un autre dossier : `python -m careseq.cli --root CHEMIN_DU_PROJET run`.

`config.json` fixe les données, graines et hyperparamètres. `run` régénère les données et réentraîne les modèles : cette commande remplace les artefacts du même dossier. Pour une nouvelle expérience à conserver séparément, employer une autre racine avec `--root` et le fichier de configuration avec `run --config CHEMIN_CONFIG`.

La commande `report` reconstruit uniquement les figures et le rapport à partir des résultats enregistrés. La commande `predict` charge `models/best.pt`, conserve les 24 derniers événements au maximum et affiche trois probabilités. Ces probabilités ne sont ni calibrées pour un usage médical ni des recommandations de soins.

## Organisation

```text
config.json             protocole et hyperparamètres fixés avant évaluation
src/careseq/data.py      générateur, séparation par patient, fenêtres
src/careseq/baselines.py références statistiques apprises sur train
src/careseq/model.py     embeddings, positions, Transformer, classification
src/careseq/train.py     apprentissage, sélection sur validation, évaluation
src/careseq/metrics.py   métriques et bootstrap par patient
src/careseq/inference.py rechargement du modèle et prédiction
src/careseq/report.py    figures Matplotlib et rapport local
tests/                  vérifications du protocole et du modèle
data/                   événements fictifs, fenêtres, séparation, manifeste
models/                 poids entraînés et références statistiques
reports/                scores, courbes, matrice de confusion, exemples
docs/                   explications, protocole, fiche du modèle
```

Les fichiers de données et les poids sont exclus du dépôt Git pour garder un historique léger. Pour générer `models/best.pt` avant une prédiction, lancer `python -m careseq.cli run`. Les autres checkpoints et les données restent disponibles dans le dossier local ; ils se régénèrent avec `run`.

## Lire la comparaison correctement

Le générateur contient des motifs favorables à l’usage d’un historique de plusieurs événements. Un gain du Transformer montre qu’il peut apprendre ces motifs ; il ne démontre pas sa supériorité générale sur des données de santé réelles.

L’exactitude donne autant de poids à chaque fenêtre. Un patient avec plus d’événements produit davantage de fenêtres. Le bootstrap regroupe les fenêtres d’un patient afin de ne pas les considérer comme indépendantes. Il ne couvre pas toute l’incertitude liée au choix du générateur, de l’architecture ou des données d’entraînement.

Les 14 catégories sont définies avant la génération. Certaines spécialités initiales sont tirées au hasard après le même contexte : leur prédiction est intrinsèquement incertaine. La F1 macro et la matrice de confusion rendent visibles les différences entre catégories.

## Sources techniques

- [PyTorch — TransformerEncoder](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html)
- [PyTorch — reproductibilité](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
- [PyTorch — installation locale](https://pytorch.org/get-started/locally/)

Les graines sont fixées et les opérations déterministes activées. Cela facilite la répétition dans le même environnement ; des différences entre versions, plateformes et matériels restent possibles.
