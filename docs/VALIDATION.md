# Validation de la livraison

## Tests

Dix tests réussis sous Windows avec Python 3.12 et PyTorch CPU. Résultat machine : `reports/tests.xml`.

- Génération déterministe et dates strictement croissantes par patient.
- Séparations de patients disjointes, complètes et réalisées avant le fenêtrage.
- Alignement exact entrée/cible et exclusion des événements futurs.
- Invariance des prédictions lorsqu’on ajoute du padding correctement masqué.
- Dimensions de sortie, gradients calculables et embedding du padding non entraîné.
- Probabilités des références et repli sur contexte inconnu.
- Métriques vérifiées sur un cas parfait.
- Bootstrap apparié donnant zéro écart pour deux modèles identiques.
- Rechargement de checkpoint donnant exactement les mêmes logits.
- Mini-expérience complète : préparation, entraînement, sélection, rapport et inférence.

Les jeux utilisés par les tests sont des mini-données synthétiques temporaires, distinctes des résultats de l’expérience principale.

## Exécution principale

3 000 patients synthétiques, 81 601 événements, 75 601 fenêtres. Patients : 2 100 train, 450 validation, 450 test. Fenêtres : 53 161 train, 11 276 validation, 11 164 test.

Le contrôle `reports/data_quality.json` confirme des patients disjoints et la présence des 14 classes dans l’entraînement. L’empreinte du Parquet figure dans `data/manifest.json` et dans chaque checkpoint.

Les graines 7, 17 et 27 et le protocole de huit époques maximum ont été fixés avant l’évaluation test. Les courbes, checkpoints et scores sont enregistrés. La sélection est faite sur validation, pas sur l’exactitude test.

## Reproductibilité et réserves

L’installation éditable et `pip check` ont réussi. Le modèle a été entraîné sur CPU ; l’environnement Python est isolé de celui du projet Open DAMIR.

Le script Bash est fourni, mais n’a pas été exécuté sous Linux/WSL. Aucun GPU, déploiement, service distant ou validation médicale n’est impliqué. Les durées d’entraînement sont celles de cette exécution sur une machine également utilisée pour préparer les autres fichiers ; elles ne constituent pas un benchmark matériel contrôlé.

Les métriques démontrent un comportement sur le générateur fictif. Elles ne doivent pas être présentées comme une mesure de performance sur des patients réels.

## Vérification de la livraison

Le rapport HTML et ses trois figures ont été inspectés dans le navigateur. L’inférence CLI avec le checkpoint retenu a réussi ; son résultat est conservé dans reports/prediction_demo.json. La construction du paquet wheel a également réussi.
