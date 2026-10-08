# 1 - Importation des bibliotheques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# Bibliotheques pour ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Modeles
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Metriques d'evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Bibliotheques pour XAI
import shap
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.multiclass import OneVsRestClassifier

# Configuration de l'affichage
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# 2 - Chargement et exploration des donnees
#Chargement du fichier CSV
# Si le fichier est dans l'environnement Colab
df = pd.read_csv('tripadvisor_hotel_reviews.csv')

# Affichage des premières lignes
print("\n" + "="*60)
print("APERÇU DES DONNÉES")
print("="*60)
print(df.head())

# Informations generales
print("\n" + "="*60)
print("INFORMATIONS GENERALES")
print("="*60)
print(df.info())

# Verification des valeurs manquantes
print("\n" + "="*60)
print("VALEURS MANQUANTES")
print("="*60)
print(df.isnull().sum())

# Statistiques descriptives
print("\n" + "="*60)
print("STATISTIQUES DESCRIPTIVES")
print("="*60)
print(df['Rating'].describe())

# Distribution des notes
print("\n" + "="*60)
print("DISTRIBUTION DES NOTES")
print("="*60)
rating_counts = df['Rating'].value_counts().sort_index()
for rating, count in rating_counts.items():
    print(f"Note {rating}: {count} avis ({count/len(df)*100:.1f}%)")



# 3 - Analyse exploratoire des donnees (EDA)
# 3.1 Visualisation de la distribution des notes
plt.figure(figsize=(14, 5))

# Histogramme
plt.subplot(1, 3, 1)
colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#10ac84']
rating_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Distribution des Notes', fontsize=14)
plt.xlabel('Note')
plt.ylabel("Nombre d'avis")
plt.xticks(rotation=0)

# Camembert
plt.subplot(1, 3, 2)
plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Proportion des Notes', fontsize=14)

# Analyse de l'equilibre des classes
plt.subplot(1, 3, 3)
class_balance = rating_counts / len(df) * 100
plt.bar(class_balance.index, class_balance.values, color=colors, edgecolor='black')
plt.axhline(y=20, color='red', linestyle='--', label='Equilibre parfait (20%)')
plt.title('Equilibre des Classes', fontsize=14)
plt.xlabel('Note')
plt.ylabel('Pourcentage (%)')
plt.legend()
plt.ylim(0, 35)

plt.tight_layout()
plt.show()

# 3.2 Analyse de la longueur des avis
df['review_length_words'] = df['Review'].apply(lambda x: len(str(x).split()))
df['review_length_chars'] = df['Review'].apply(lambda x: len(str(x)))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x='Rating', y='review_length_words', data=df, palette=colors)
plt.title('Longueur des avis (nombre de mots) par note', fontsize=12)
plt.xlabel('Note')
plt.ylabel('Nombre de mots')

plt.subplot(1, 2, 2)
sns.boxplot(x='Rating', y='review_length_chars', data=df, palette=colors)
plt.title('Longueur des avis (nombre de caracteres) par note', fontsize=12)
plt.xlabel('Note')
plt.ylabel('Nombre de caracteres')

plt.tight_layout()
plt.show()

# Statistiques de longueur
print("\n" + "="*60)
print("STATISTIQUES DE LONGUEUR DES AVIS")
print("="*60)
print(df.groupby('Rating')['review_length_words'].describe())

# 3.3 Nuage de mots pour visualiser les termes frequents
from wordcloud import WordCloud

# Fonction pour creer un nuage de mots
def create_wordcloud(text, title, color):
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap=color, max_words=100).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=14)
    plt.show()

# Nuage de mots pour les avis positifs (4 et 5)
positive_text = ' '.join(df[df['Rating'].isin([4,5])]['Review'].astype(str))
create_wordcloud(positive_text, 'Mots frequents dans les avis POSITIFS (notes 4-5)', 'Greens')

# Nuage de mots pour les avis negatifs (1 et 2)
negative_text = ' '.join(df[df['Rating'].isin([1,2])]['Review'].astype(str))
create_wordcloud(negative_text, 'Mots fréquents dans les avis NÉGATIFS (notes 1-2)', 'Reds')



# 4 - Pretraitement des donnees
# 4.1 Fonction de nettoyage du texte
def clean_text(text):
    """
    Nettoie le texte en :
    - Mettant en minuscules
    - Supprimant les caracteres speciaux et les chiffres
    - Supprimant les espaces multiples
    """
    if isinstance(text, str):
        # Mise en minuscules
        text = text.lower()
        # Suppression des caracteres speciaux (garde seulement lettres et espaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
    else:
        text = ""
    return text

# Application du nettoyage
df['clean_review'] = df['Review'].apply(clean_text)

# Affichage d'un exemple
print("\nExemple de nettoyage :")
print("-" * 50)
print("AVANT :", df['Review'].iloc[0][:150])
print("APRES :", df['clean_review'].iloc[0][:150])

# 4.2 Vectorisation TF-IDF
print("\n" + "="*60)
print("VECTORISATION TF-IDF")
print("="*60)

tfidf = TfidfVectorizer(
    max_features=5000,           # Garder les 5000 mots les plus frquents
    ngram_range=(1, 2),          # Unigrammes et bigrammes
    stop_words='english',        # Supprimer les mots vides anglais
    min_df=2,                    # Ignorer les mots apparaissant moins de 2 fois
    max_df=0.95                  # Ignorer les mots apparaissant dans plus de 95% des docs
)

print("Calcul de la matrice TF-IDF...")
X = tfidf.fit_transform(df['clean_review']).toarray()

# Transformation de la variable cible
y = df['Rating'] - 1  # Conversion de 1-5 a 0-4

print(f"Matrice TF-IDF créée : {X.shape[0]} échantillons, {X.shape[1]} features")
print(f"Variable cible : {len(np.unique(y))} classes (0 à 4)")

# Affichage des mots les plus importants
feature_names = tfidf.get_feature_names_out()
print(f"\nExemples de features (mots) : {feature_names[:20]}")

# 4.3 Separation en ensembles d'entrainement et de test
print("\n" + "="*60)
print("SePARATION TRAIN/TEST")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Ensemble d'entrainement : {X_train.shape[0]} echantillons")
print(f"Ensemble de test : {X_test.shape[0]} echantillons")

# Verification de la distribution dans train et test
print("\nDistribution des classes - Entraînement :")
train_dist = pd.Series(y_train).value_counts().sort_index()
for i in range(5):
    print(f"  Classe {i} (note {i+1}): {train_dist[i]} ({train_dist[i]/len(y_train)*100:.1f}%)")

print("\nDistribution des classes - Test :")
test_dist = pd.Series(y_test).value_counts().sort_index()
for i in range(5):
    print(f"  Classe {i} (note {i+1}): {test_dist[i]} ({test_dist[i]/len(y_test)*100:.1f}%)")




# 5 - Modelisation
# 5.1 def et entrainement des modeles
print("="*60)
print("ENTRAINEMENT DES MODELES")
print("="*60)

models = {
    'Regression Logistique': OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42)),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
}

results = {}

for name, model in models.items():
    print(f"\n Entranement de {name}")

    # Entrainement
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calcul des metriques
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    results[name] = {
        'model': model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_score': f1,
        'predictions': y_pred_test
    }

    print(f"  + Accuracy entrainement : {train_acc:.4f}")
    print(f"  + Accuracy test : {test_acc:.4f}")
    print(f"  + F1-score (weighted) : {f1:.4f}")

# 5.2 Comparaison des performances
print("\n" + "="*60)
print("COMPARAISON DES PERFORMANCES")
print("="*60)

comparison_df = pd.DataFrame({
    'Modèle': list(results.keys()),
    'Accuracy (Train)': [results[m]['train_accuracy'] for m in results],
    'Accuracy (Test)': [results[m]['test_accuracy'] for m in results],
    'F1-Score (Test)': [results[m]['f1_score'] for m in results]
})

print(comparison_df.to_string(index=False))

# Visualisation de la comparaison
plt.figure(figsize=(10, 6))
x = np.arange(len(comparison_df))
width = 0.25

plt.bar(x - width, comparison_df['Accuracy (Test)'], width, label='Accuracy Test', color='#3498db')
plt.bar(x, comparison_df['F1-Score (Test)'], width, label='F1-Score Test', color='#e74c3c')
plt.bar(x + width, comparison_df['Accuracy (Train)'], width, label='Accuracy Train', color='#2ecc71')

plt.xlabel('Modeles')
plt.ylabel('Score')
plt.title('Comparaison des performances des modeles')
plt.xticks(x, comparison_df['Modèle'], rotation=15)
plt.legend()
plt.tight_layout()
plt.show()

# 5.3 Selection du meilleur modele
best_model_name = comparison_df.loc[comparison_df['F1-Score (Test)'].idxmax(), 'Modèle']
best_model = results[best_model_name]['model']
print(f"\nMEILLEUR MODELE : {best_model_name}")
print(f"   F1-Score : {comparison_df.loc[comparison_df['Modèle'] == best_model_name, 'F1-Score (Test)'].values[0]:.4f}")


# 6 - Evaluation detaillee du meilleur modele
print("="*60)
print("EVALUATION DETAILLEE - MEILLEUR MODELE")
print("="*60)

# Recuperation des predictions du meilleur modele
y_pred = results[best_model_name]['predictions']

# 6.1 Rapport de classification detaille
print("\nRAPPORT DE CLASSIFICATION")
print("-" * 40)
target_names = ['Note 1 (Tres mauvais)', 'Note 2 (Mauvais)', 'Note 3 (Moyen)',
                'Note 4 (Bon)', 'Note 5 (Excellent)']
print(classification_report(y_test, y_pred, target_names=target_names))

# 6.2 Matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['1', '2', '3', '4', '5'],
            yticklabels=['1', '2', '3', '4', '5'])
plt.title(f'Matrice de confusion - {best_model_name}', fontsize=14)
plt.xlabel('Prediction', fontsize=12)
plt.ylabel('Verite terrain', fontsize=12)
plt.tight_layout()
plt.show()

# 6.3 Analyse par classe
print("\nANALYSE PAR CLASSE")
print("-" * 40)
for i in range(5):
    precision = precision_score(y_test, y_pred, labels=[i], average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, labels=[i], average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, labels=[i], average='weighted', zero_division=0)
    print(f"Classe {i+1} (Note {i+1}):")
    print(f"  Precision: {precision:.4f} | Rappel: {recall:.4f} | F1: {f1:.4f}")




# 7 - Explicabilite du modele (XAI)
print("="*60)
print("EXPLICABILITE DU MODELE (XAI)")
print("="*60)

# 7.1 Feature Importance (pour Random Forest et XGBoost)
if best_model_name in ['Random Forest', 'XGBoost']:
    print("\n7.1 FEATURE IMPORTANCE")

    # Recuperation de l'importance des features
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_

        # Top 20 des mots les plus importants
        top_idx = np.argsort(importances)[-20:][::-1]
        top_features = feature_names[top_idx]
        top_importances = importances[top_idx]

        # Visualisation
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importances, color='#2ecc71')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top 20 des mots les plus importants - {best_model_name}', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        # Affichage des resultats
        print("\nTop 10 des mots les plus importants :")
        for i, (feature, importance) in enumerate(zip(top_features[:10], top_importances[:10])):
            print(f"  {i+1}. '{feature}' : {importance:.4f}")

# 7.2 SHAP Analysis (pour XGBoost)
if best_model_name == 'XGBoost':
    print("\n7.2 ANALYSE SHAP")

    # Installation de shap
    try:
        import shap

        # Preparation d'un echantillon pour SHAP
        sample_size = min(500, X_test.shape[0])
        np.random.seed(42)
        sample_idx = np.random.choice(X_test.shape[0], sample_size, replace=False)
        X_sample = X_test[sample_idx]

        print(f"Analyse SHAP sur {sample_size} echantillons")

        # Creation de l'explainer
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_sample)

        # SHAP Summary Plot (class 4 - note 5)
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values[:,:,4], X_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary Plot - Contribution des mots (classe 5)', fontsize=14)
        plt.tight_layout()
        plt.show()

        # SHAP Bar Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[:,:,4], X_sample, feature_names=feature_names,
                          plot_type="bar", show=False)
        plt.title(f'SHAP Bar Plot - Importance globale (classe 5)', fontsize=14)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erreur lors de l'analyse SHAP : {e}")
        print(" SHAP peut necessiter une installation supplémentaire.")

# 7.3 Permutation Importance
print("\n7.3 PERMUTATION IMPORTANCE")
print("Calcul en cours (cela peut prendre quelques minutes)")

try:
    n_sample = min(500, X_test.shape[0])
    X_perm = X_test[:n_sample]
    y_perm = y_test.iloc[:n_sample]

    # Top 100 features selon l'importance du modèle
    cand_idx = np.argsort(best_model.feature_importances_)[-100:]

    # On permute seulement ces 100 colonnes
    base_score = f1_score(y_perm, best_model.predict(X_perm), average='weighted')
    rng = np.random.RandomState(42)
    importances_mean = np.zeros(X_test.shape[1])

    for j in cand_idx:
        scores = []
        for _ in range(3):
            X_tmp = X_perm.copy()
            X_tmp[:, j] = rng.permutation(X_tmp[:, j])
            scores.append(f1_score(y_perm, best_model.predict(X_tmp), average='weighted'))
        importances_mean[j] = base_score - np.mean(scores)

    class _Perm:
        pass
    perm_importance = _Perm()
    perm_importance.importances_mean = importances_mean

    # Top 15 features par permutation importance
    top_perm_idx = np.argsort(perm_importance.importances_mean)[-15:][::-1]
    top_perm_features = feature_names[top_perm_idx]
    top_perm_importances = perm_importance.importances_mean[top_perm_idx]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_perm_features)), top_perm_importances, color='#e74c3c')
    plt.yticks(range(len(top_perm_features)), top_perm_features)
    plt.xlabel('Perte de performance moyenne', fontsize=12)
    plt.title(f'Top 15 features - Permutation Importance', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    print("\nTop 10 des features par permutation importance :")
    for i, (feature, importance) in enumerate(zip(top_perm_features[:10], top_perm_importances[:10])):
        print(f"  {i+1}. '{feature}' : {importance:.4f}")

except Exception as e:
    print(f"Erreur lors du calcul de permutation importance : {e}")



# 8 - Analyse locale - Exemples individuels
print("="*60)
print("ANALYSE LOCALE - EXEMPLES INDIVIDUELS")
print("="*60)

# 8.1 Selection d'exemples
print("\n8.1 SELECTION D'EXEMPLES")
print("-" * 40)

# Exemple d'avis tres positif
positive_example = df[df['Rating'] == 5]['Review'].iloc[0]
positive_clean = df[df['Rating'] == 5]['clean_review'].iloc[0]

# Exemple d'avis tres négatif
negative_example = df[df['Rating'] == 1]['Review'].iloc[0]
negative_clean = df[df['Rating'] == 1]['clean_review'].iloc[0]

# Exemple d'avis moyen
medium_example = df[df['Rating'] == 3]['Review'].iloc[0]
medium_clean = df[df['Rating'] == 3]['clean_review'].iloc[0]

print("\n AVIS POSITIF (Note 5) :")
print(f"   Aperçu : {positive_example[:300]}...")
print(f"   Note réelle : 5")

print("\n AVIS NÉGATIF (Note 1) :")
print(f"   Aperçu : {negative_example[:300]}...")
print(f"   Note réelle : 1")

print("\n AVIS MOYEN (Note 3) :")
print(f"   Aperçu : {medium_example[:300]}...")
print(f"   Note réelle : 3")

# 8.2 Prediction et analyse des exemples
print("\n PREDICTION ET ANALYSE")
print("-" * 40)

def predict_and_analyze(text, clean_text, true_rating, model, vectorizer, feature_names):

    # Vectorisation
    vec = vectorizer.transform([clean_text]).toarray()

    # Prediction
    pred = model.predict(vec)[0] + 1  # +1 pour revenir à l'échelle 1-5
    pred_proba = model.predict_proba(vec)[0]

    print(f"\n AVIS ANALYSE :")
    print(f"   Note reelle : {true_rating}")
    print(f"   Note predite : {pred}")
    print(f"   Probabilites par classe :")
    for i, prob in enumerate(pred_proba):
        print(f"     Note {i+1}: {prob:.4f}")

    # Pour les modeles avec feature importance
    if hasattr(model, 'feature_importances_'):
        # Recuperation des features non nulles
        non_zero_idx = vec[0].nonzero()[0]
        if len(non_zero_idx) > 0:
            features_importance = [(feature_names[idx], vec[0][idx], model.feature_importances_[idx])
                                  for idx in non_zero_idx]
            features_importance.sort(key=lambda x: x[2], reverse=True)

            print(f"\n   Top 5 mots contribuant à la prediction :")
            for feat, value, imp in features_importance[:5]:
                print(f"     - '{feat}' (frequence TF-IDF: {value:.3f}, importance: {imp:.4f})")

    return pred, pred_proba

# Analyse des exemples
print("\n ANALYSE DE L'AVIS POSITIF")
print("-" * 40)
pred_pos, _ = predict_and_analyze(positive_example, positive_clean, 5, best_model, tfidf, feature_names)

print("\n ANALYSE DE L'AVIS NEGATIF")
print("-" * 40)
pred_neg, _ = predict_and_analyze(negative_example, negative_clean, 1, best_model, tfidf, feature_names)

print("\n ANALYSE DE L'AVIS MOYEN")
print("-" * 40)
pred_med, _ = predict_and_analyze(medium_example, medium_clean, 3, best_model, tfidf, feature_names)

# 8.3 Visualisation des probabilites de prediction
plt.figure(figsize=(12, 4))

examples = [('Avis Positif (Note 5 )', positive_example, 5),
            ('Avis Négatif (Note 1 )', negative_example, 1),
            ('Avis Moyen (Note 3 )', medium_example, 3)]

for idx, (title, text, true_rating) in enumerate(examples):
    clean = clean_text(text)
    vec = tfidf.transform([clean]).toarray()
    proba = best_model.predict_proba(vec)[0]

    plt.subplot(1, 3, idx+1)
    bars = plt.bar([1,2,3,4,5], proba, color=colors)
    plt.bar(true_rating, proba[true_rating-1], color='gold', edgecolor='black', linewidth=2)
    plt.title(title, fontsize=10)
    plt.xlabel('Note')
    plt.ylabel('Probabilité')
    plt.xticks(range(1,6))
    plt.ylim(0, 1)

plt.tight_layout()
plt.show()



# 9 - Synthese et visualisation finale
print("="*60)
print(" SYNTHESE FINALE DU PROJET")
print("="*60)

# 9.1 Resume des performances
print("\n RESUME DES PERFORMANCES")
print("-" * 40)

comparison_df_sorted = comparison_df.sort_values('F1-Score (Test)', ascending=False)
for idx, row in comparison_df_sorted.iterrows():
    print(f"\n {row['Modèle']}:")
    print(f"   Accuracy Test : {row['Accuracy (Test)']:.4f}")
    print(f"   F1-Score Test : {row['F1-Score (Test)']:.4f}")

# 9.2 Top mots par classe
print("\n TOP 10 MOTS PAR CLASSE")
print("-" * 40)

from collections import Counter

def get_top_words_by_class(df, tfidf, class_rating):
    """
    Extrait les mots les plus frequents pour une classe donnee
    """
    class_reviews = df[df['Rating'] == class_rating]['clean_review'].str.split()
    all_words = [word for review in class_reviews for word in review]
    word_freq = Counter(all_words)
    return word_freq.most_common(10)

for rating in [1, 2, 3, 4, 5]:
    top_words = get_top_words_by_class(df, tfidf, rating)
    print(f"\nNote {rating} :")
    for word, count in top_words:
        print(f"   - '{word}' ({count} occurrences)")

# 9.3 Graphique de synthese final
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution des notes
axes[0, 0].bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='black')
axes[0, 0].set_title('Distribution des notes', fontsize=12)
axes[0, 0].set_xlabel('Note')
axes[0, 0].set_ylabel("Nombre d'avis")

# 2. Comparaison des modeles
axes[0, 1].barh(comparison_df['Modèle'], comparison_df['F1-Score (Test)'], color='#3498db')
axes[0, 1].set_title('F1-Score par modele', fontsize=12)
axes[0, 1].set_xlabel('F1-Score')
axes[0, 1].set_xlim(0, 1)

# 3. Matrice de confusion normalisee
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title(f'Matrice de confusion normalisee - {best_model_name}', fontsize=12)
axes[1, 0].set_xlabel('Prediction')
axes[1, 0].set_ylabel('Verite terrain')

# 4. Top features
if best_model_name in ['Random Forest', 'XGBoost'] and hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    top_features_short = feature_names[top_idx]
    top_importances_short = importances[top_idx]
    axes[1, 1].barh(range(len(top_features_short)), top_importances_short, color='#2ecc71')
    axes[1, 1].set_yticks(range(len(top_features_short)))
    axes[1, 1].set_yticklabels(top_features_short)
    axes[1, 1].set_title('Top 10 des mots les plus importants', fontsize=12)
    axes[1, 1].set_xlabel('Importance')

plt.tight_layout()
plt.show()

print("\n Projet terminé avec succès !")