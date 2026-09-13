# Fiche du modèle

**Nom :** CareTransformer, démonstration pédagogique. **Version :** 0.1.0. **Domaine :** séquences fictives d’événements nommés comme des actes de soins.

## Usage prévu

Comprendre la préparation de séquences, les embeddings, l’attention, l’entraînement PyTorch et la comparaison à des références statistiques. Présenter un projet de portfolio avec un protocole reproductible.

## Entrées et sorties

Entrée : entre 1 et 24 catégories connues, ordonnées. L’entraînement utilise au moins deux événements ; une entrée d’un seul événement est hors de ce périmètre d’apprentissage. Les séquences plus longues sont tronquées aux 24 derniers événements.

Sortie : probabilités sur 14 catégories, puis top 3. Le padding utilise le code 0 en entrée et n’est jamais une classe de sortie.

## Données et population

Trois mille patients entièrement fictifs. Les noms des catégories évoquent des soins, mais les dates, fréquences et transitions sont inventées. Le générateur n’utilise pas de données individuelles réelles et n’est pas calibré sur Open DAMIR.

Les motifs contiennent volontairement une information située au-delà du dernier événement. Les résultats sont favorables ou défavorables à un modèle selon les règles choisies ; ils ne se généralisent pas automatiquement à des parcours médicaux.

## Architecture et entraînement

Architecture par défaut : embeddings de dimension 32, positions apprises, deux couches d’encodeur Transformer, quatre têtes d’attention, sortie linéaire vers 14 classes. Le checkpoint contient les poids, l’architecture, le vocabulaire, la graine, l’époque et l’empreinte des données.

Sélection de l’époque et de la graine sur la NLL de validation. Le test comprend des patients disjoints issus du même générateur. Les scores effectifs sont dans `reports/results.json` et la synthèse dans `reports/RESULTATS.md`.

## Limites et usages exclus

- Aucune validité clinique, prévision pour un patient réel ou aide à une décision de soins.
- Aucune recommandation de parcours, diagnostic ou estimation de risque médical.
- Pas de validation externe, temporelle, inter-hôpitaux ou entre populations.
- Pas d’apprentissage de la durée entre événements : seules les catégories et positions sont entrées.
- Pas de garantie de calibration des probabilités.
- Pas de prise en charge de catégories inconnues : la commande de prédiction les rejette.
- Pas de preuve que les poids d’attention expliquent causalement une prédiction.

## Artefacts et reprise

`models/best.pt` recharge le modèle retenu avec `torch.load(..., weights_only=True)`. Les autres graines sont sauvegardées séparément. Le code et le verrou des versions permettent de reconstruire l’expérience. Seuls les checkpoints de provenance connue doivent être utilisés.
