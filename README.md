# Analyse Entrainement

Ce projet contient un script Python permettant d'analyser des fichiers `.fit` (générés par des montres GPS/de sport) de course à pied.
Il extrait les données métriques (distance, vitesse, fréquence cardiaque, etc.) et les agrège dans un historique pour permettre le suivi de l'entraînement.

## Structure du projet

- `analyze_run.py` : Script principal pour l'analyse des fichiers FIT.
- `historique_running.json` : Base de données locale stockant l'historique des sorties.
- `dernier_fit/` : Dossier contenant le(s) dernier(s) fichier(s) FIT à analyser.
- `archives_fit/` : Dossier contenant les anciens fichiers FIT (non synchronisé).

## Prérequis

- Python 3
- Les dépendances listées dans le code (`pandas`, `numpy`, `scipy`, `fitparse`, `pydantic`).

## Utilisation

Placez le fichier zip contenant le fichier `.fit` de votre dernière course dans le dossier `dernier_fit/`, puis lancez le script `analyze_run.py`.
