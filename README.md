# Tourism Reviews XAI

## Présentation du projet

Ce projet porte sur l'analyse des avis clients dans le domaine du tourisme à l'aide de techniques de Machine Learning et d'Explainable AI (XAI).

L'objectif principal est de prédire la note attribuée à un hôtel à partir du texte d'un avis client et d'interpréter les facteurs qui influencent les prédictions du modèle.

Le problème est formulé comme une classification multiclasse avec cinq catégories correspondant aux notes de 1 à 5.

## Objectifs

- Analyser les avis clients dans le secteur touristique.
- Prédire la note d'un hôtel à partir du texte d'un avis.
- Identifier les facteurs qui influencent la satisfaction des clients.
- Comparer plusieurs modèles de Machine Learning.
- Interpréter les prédictions à l'aide de techniques d'Explainable AI.
- Identifier les limites du modèle et proposer des pistes d'amélioration.

## Dataset

Le projet utilise un dataset d'avis clients provenant de TripAdvisor.

Les données contiennent les informations suivantes :

- `Review` : texte de l'avis client en anglais.
- `Rating` : note attribuée à l'hôtel sur une échelle de 1 à 5.

Le dataset contient plus de 20 000 avis et présente un déséquilibre entre les différentes classes de notes.

Les notes 4 et 5 sont majoritaires tandis que les notes 1 et 2 sont minoritaires.

## Problématique

La variable cible `Rating` contient cinq classes :

| Rating | Signification |
|---|---|
| 1 | Très mauvais |
| 2 | Mauvais |
| 3 | Moyen |
| 4 | Bon |
| 5 | Excellent |

Il s'agit donc d'un problème de classification multiclasse.

## Méthodologie

Le projet suit plusieurs étapes :

1. Chargement et exploration des données.
2. Analyse exploratoire des données (EDA).
3. Nettoyage et prétraitement des avis textuels.
4. Vectorisation des textes avec TF-IDF.
5. Séparation des données en ensembles d'entraînement et de test.
6. Entraînement de plusieurs modèles de Machine Learning.
7. Évaluation des performances.
8. Analyse de la matrice de confusion.
9. Explicabilité des prédictions.
10. Analyse des limites et des pistes d'amélioration.

## Prétraitement et représentation des textes

Les avis sont nettoyés avant leur utilisation pour la modélisation.

La représentation TF-IDF est ensuite utilisée pour transformer les textes en variables numériques.

La vectorisation génère 5 000 features comprenant des mots individuels ainsi que des bigrammes.

La séparation entre les données d'entraînement et de test est stratifiée afin de conserver la distribution des différentes notes.

## Modèles de Machine Learning

Trois modèles ont été comparés :

- Régression logistique
- Random Forest
- XGBoost

### Résultats des modèles

La régression logistique obtient les meilleures performances globales sur l'ensemble de test.

| Modèle | Accuracy test | F1-score test |
|---|---:|---:|
| Régression logistique | 62% | 59% |
| XGBoost | 60% | 59% |
| Random Forest | 56% | 51% |

La régression logistique offre le meilleur compromis entre performance et généralisation.

## Analyse des performances

Le modèle rencontre davantage de difficultés pour les classes intermédiaires.

Les notes extrêmes sont mieux prédites :

- Note 1 : F1-score de 68%
- Note 5 : F1-score de 75%

La note 3 présente en revanche un F1-score de seulement 27%.

Cette difficulté s'explique notamment par le déséquilibre des classes et par la frontière moins nette entre les notes intermédiaires.

## Explainable AI

L'analyse XAI permet d'identifier les mots qui contribuent le plus aux prédictions du modèle.

Parmi les termes les plus importants figurent notamment :

- `excellent`
- `perfect`
- `great`
- `good`
- `clean`
- `friendly`

L'analyse montre que le modèle s'appuie principalement sur des termes positifs pour identifier les avis associés aux notes élevées.

## Principales limites

Le projet présente plusieurs limites :

- Déséquilibre entre les classes.
- Difficulté à distinguer les notes intermédiaires.
- Surapprentissage observé pour certains modèles.
- Difficulté à capturer le contexte complet des phrases.

## Pistes d'amélioration

Plusieurs améliorations peuvent être envisagées :

- Utiliser des modèles NLP plus avancés comme BERT.
- Mettre en place des techniques de rééquilibrage des classes.
- Améliorer le prétraitement du texte.
- Optimiser les hyperparamètres des modèles.
- Explorer d'autres méthodes d'explicabilité.

## Technologies utilisées

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Random Forest
- XGBoost
- Natural Language Processing (NLP)
- Explainable AI (XAI)
- Google Colab

## Résultats

Le projet montre qu'une approche basée sur le Machine Learning permet de prédire les notes d'avis clients à partir de leur contenu textuel.

La régression logistique constitue le meilleur modèle parmi les trois modèles étudiés, avec une accuracy de 62% et un F1-score de 59% sur les données de test.

L'analyse XAI permet également de mieux comprendre les facteurs textuels utilisés par le modèle pour effectuer ses prédictions.

## Auteur

Khadija Belbaraka

Data & AI | Python | SQL | Machine Learning | NLP | Explainable AI
