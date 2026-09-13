# Protocole expérimental

## Données

Générateur indépendant d’Open DAMIR, graine 2026. Chaque patient possède un profil fictif parmi trois branches. Les motifs mélangent visites, examens, suivi et quelques séquences d’urgence. Dix pour cent de bruit d’insertion sont appliqués pendant les motifs ; les paramètres exacts sont dans `data.py`. Ils ne représentent pas des fréquences médicales réelles.

Les parcours commencent par un généraliste et une biologie, puis une spécialité aléatoire liée au profil. La cible de cette troisième position est incluse avec un contexte minimal de deux événements. La spécialité n’est alors pas identifiable dans le passé : le modèle doit exprimer cette incertitude. Des motifs ultérieurs révèlent une dépendance à un historique plus long.

## Découpage et caractéristiques

Patients permutés avec une graine indépendante 31, puis séparation 70/15/15. La génération des fenêtres intervient ensuite. Aucun comptage de référence n’utilise validation ou test. Le vocabulaire fixe est défini avant la génération.

Entrées : identifiants d’événements passés et positions dans une fenêtre de 24 au maximum. Aucun identifiant patient, profil de génération, coût, date future ou événement cible n’est une caractéristique. Les identifiants patients sont utilisés uniquement pour le découpage, l’audit et le bootstrap.

Le nombre d’exemples varie selon la longueur du parcours. La fonction de coût et l’exactitude donnent autant de poids à chaque fenêtre, donc davantage de poids aux patients ayant plus d’événements.

## Modèles et sélection

Fréquence globale, Markov 1 et Markov 2 avec lissage additif 0,5 et repli sur contexte plus court. La référence de comparaison est celle de plus faible NLL sur validation.

Transformer de dimension 32, deux couches, quatre têtes, couche interne de dimension 64, dropout 0,1. AdamW, taux 0,001, weight decay 0,01, clipping du gradient à 1, batch 256, huit époques au maximum. Arrêt anticipé après trois époques sans amélioration de NLL de validation supérieure à 0,00001.

Graines 7, 17 et 27 fixées avant l’expérience. Chaque graine conserve le checkpoint de plus faible NLL de validation. La graine retenue est également choisie sur cette NLL. Les performances test de toutes les graines sont affichées pour transparence, sans choisir la meilleure sur test.

## Évaluation

Le test intervient après la fixation des checkpoints. Métriques : exactitude, top 3, F1 macro sur les 14 classes, NLL, confusion, rappel et support par classe. Bootstrap apparié par patient : 1 000 tirages, graine 123, percentiles 2,5 % et 97,5 % de l’écart d’exactitude pondéré par fenêtres.

Le diagnostic de contexte court conserve le même checkpoint et lui donne seulement les deux derniers événements. Les positions sont réinitialisées et la distribution des entrées change. Il ne s’agit pas d’une comparaison avec un Transformer réentraîné sur deux événements, ni d’une preuve causale isolant la mémoire longue.

## Reproductibilité et limites

Environnement CPU, quatre threads PyTorch, graines Python/NumPy/PyTorch fixées, opérations déterministes activées. Les versions sont enregistrées. Une répétition sur une autre version, un GPU ou une autre plateforme n’est pas supposée bit à bit identique.

Le test ne comprend ni séparation temporelle, ni décalage de population, ni source clinique externe. Les profils et motifs ont été conçus pour montrer l’intérêt potentiel des séquences. La comparaison ne doit pas être présentée comme un résultat de recherche clinique ou comme une reproduction d’un projet Inria/DREES.
