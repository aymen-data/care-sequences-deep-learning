# Résultats — séquences synthétiques

3000 patients fictifs, 81601 événements.

Le Transformer retenu dépasse la référence statistique retenue dans cette expérience synthétique.

| Modèle | Exactitude | Top 3 | F1 macro | NLL |
|---|---:|---:|---:|---:|
| Frequence | 19.53% | 56.27% | 0.023 | 2.281 |
| Markov_1 | 63.04% | 86.32% | 0.379 | 1.137 |
| Markov_2 | 68.73% | 90.63% | 0.469 | 0.921 |
| Transformer | 81.79% | 96.16% | 0.652 | 0.626 |

Graine retenue sur validation : 7. Écart d'exactitude à Markov_2 : +13.06 points, intervalle bootstrap par patient [+12.27 ; +13.83].

Aucune validité clinique. Règles synthétiques inventées, test issu du même générateur.
